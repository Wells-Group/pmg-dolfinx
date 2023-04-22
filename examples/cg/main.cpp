#include "poisson.h"
#include "src/cg.hpp"
#include "src/chebyshev.hpp"
#include "src/csr.hpp"
#include "src/operators.hpp"
#include "src/vector.hpp"

#include <array>
#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <iostream>
#include <memory>
#include <mpi.h>

#include "src/amd_gpu.hpp"

using namespace dolfinx;
using T = double;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  int err;
  uint32_t num_devices;
  float peak_mem = 0.0;
  float global_peak_mem = 0.0;
  float mem = 0.0;

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(500), "number of dofs per rank")(
      "file", po::value<std::string>()->default_value(""), "mesh filename");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  const std::string filename = vm["file"].as<std::string>();

  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0)
      loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

#ifdef ROCM_SMI
    err = initialise_rocm_smi();
    num_devices = num_monitored_devices();
    std::cout << "MPI rank " << rank << " can see " << num_devices << " AMD GPUs\n";
    mem = print_amd_gpu_memory_percentage_used("Beginning");
    if (mem > peak_mem)
      peak_mem = mem;
#endif

#ifdef ROCM_TRACING
    if (rank == 0)
    {
      std::cout << "Using roctx tracing ranges for visulising trace data\n";
    }
#endif
    // Create a hexahedral mesh
#ifdef ROCM_TRACING
    add_profiling_annotation("making mesh");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("making mesh");
    if (mem > peak_mem)
      peak_mem = mem;
#endif

    std::shared_ptr<mesh::Mesh<T>> mesh;

    if (filename.size() > 0)
    {
      dolfinx::fem::CoordinateElement element(mesh::CellType::tetrahedron, 1);
      dolfinx::io::XDMFFile xdmf(MPI_COMM_WORLD, filename, "r");
      mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(
          xdmf.read_mesh(element, mesh::GhostMode::none, "Grid"));
    }
    else
    {
      const int order = 2;
      double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / order;
      std::size_t n0 = static_cast<int>(nx_approx);
      std::array<std::size_t, 3> nx = {n0, n0, n0};

      // Try to improve fit to ndofs +/- 5 in each direction
      if (n0 > 5)
      {
        std::int64_t best_misfit
            = (n0 * order + 1) * (n0 * order + 1) * (n0 * order + 1) - ndofs * size;
        best_misfit = std::abs(best_misfit);
        for (std::size_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
          for (std::size_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
            for (std::size_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
            {
              std::int64_t misfit
                  = (nx0 * order + 1) * (ny0 * order + 1) * (nz0 * order + 1) - ndofs * size;
              if (std::abs(misfit) < best_misfit)
              {
                best_misfit = std::abs(misfit);
                nx = {nx0, ny0, nz0};
              }
            }
      }
      mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
          comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]}, mesh::CellType::hexahedron));
    }

#ifdef ROCM_TRACING
    remove_profiling_annotation("making mesh");
#endif

#ifdef ROCM_TRACING
    add_profiling_annotation("making V");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("making V");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));
#ifdef ROCM_TRACING
    remove_profiling_annotation("making V");
#endif

    // Compute geometric range of mesh...
    std::array<T, 3> max_x = {-1e9, -1e9, -1e9}, min_x = {1e9, 1e9, 1e9};
    auto geom = mesh->geometry().x();
    for (int i = 0; i < geom.size(); i += 3)
      for (int j = 0; j < 3; ++j)
      {
        max_x[j] = max(geom[i + j], max_x[j]);
        min_x[j] = min(geom[i + j], min_x[j]);
      }
    std::array<T, 3> global_max_x, global_min_x;
    MPI_Allreduce(max_x.data(), global_max_x.data(), 3, dolfinx::MPI::mpi_type<T>(), MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(min_x.data(), global_min_x.data(), 3, dolfinx::MPI::mpi_type<T>(), MPI_MIN,
                  MPI_COMM_WORLD);

    std::size_t ncells = mesh->topology()->index_map(3)->size_global();
    std::size_t ndofs = V->dofmap()->index_map->size_global();
    if (rank == 0)
    {
      std::cout << "-----------------------------------\n";
      std::cout << "Number of ranks : " << size << "\n";
      std::cout << "Number of cells-global : " << ncells << "\n";
      std::cout << "Number of dofs-global : " << ndofs << "\n";
      std::cout << "Number of cells-rank : " << ncells / size << "\n";
      std::cout << "Number of dofs-rank : " << ndofs / size << "\n";
      std::cout << "Range min: " << global_min_x[0] << "," << global_min_x[1] << ","
                << global_min_x[2] << "\n";
      std::cout << "Range max: " << global_max_x[0] << "," << global_max_x[1] << ","
                << global_max_x[2] << "\n";

      std::cout << "-----------------------------------\n";
      std::cout << std::flush;
    }

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
#ifdef ROCM_TRACING
    add_profiling_annotation("making forms");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("making forms");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
#ifdef ROCM_TRACING
    remove_profiling_annotation("making forms");
#endif
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> out;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            //            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            //            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            //            out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
            out.push_back(x(0, p));
          }

          return {out, {out.size()}};
        });

#ifdef ROCM_TRACING
    add_profiling_annotation("doing topology");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("doing topology");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);
#ifdef ROCM_TRACING
    remove_profiling_annotation("doing topology");
#endif

#ifdef ROCM_TRACING
    add_profiling_annotation("doing boundary conditions");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("doing boundary conditions");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);
#ifdef ROCM_TRACING
    remove_profiling_annotation("doing boundary conditions");
#endif

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
    acc::MatrixOperator<T> op(a, {bc});
    auto map = op.index_map();

    fem::Function<T> u(V);
    la::Vector<T> b(map, 1);

#ifdef ROCM_TRACING
    add_profiling_annotation("assembling and scattering");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("assembling and scattering");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    b.set(T(0.0));
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, T>(b.mutable_array(), {bc});
#ifdef ROCM_TRACING
    remove_profiling_annotation("assembling and scattering");
#endif

#ifdef ROCM_TRACING
    add_profiling_annotation("setup device x");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("setup device x");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    DeviceVector x(map, 1);
    x.set(T{0.0});
#ifdef ROCM_TRACING
    remove_profiling_annotation("setup device x");
#endif

#ifdef ROCM_TRACING
    add_profiling_annotation("setup device y");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("setup device y");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    DeviceVector y(map, 1);
    y.copy_from_host(b); // Copy data from host vector to device vector
#ifdef ROCM_TRACING
    remove_profiling_annotation("setup device y");
#endif

#ifdef ROCM_TRACING
    add_profiling_annotation("matrix operator");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("matrix operator");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    // Create operator
    op(y, x);

    T norm = acc::norm(x);
    if (rank == 0)
    {
      std::cout << "Norm x vector initial " << norm << std::endl;
      std::cout << std::flush;
    }
#ifdef ROCM_TRACING
    remove_profiling_annotation("matrix operator");
#endif

    // Create distributed CG solver
#ifdef ROCM_TRACING
    add_profiling_annotation("creating cg solver");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("creating cg solver");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    dolfinx::acc::CGSolver<DeviceVector> cg(map, 1);
    cg.set_max_iterations(30);
    cg.set_tolerance(1e-5);
    cg.store_coefficients(true);
#ifdef ROCM_TRACING
    remove_profiling_annotation("creating cg solver");
#endif

    // Solve
#ifdef ROCM_TRACING
    add_profiling_annotation("cg solve");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("before cg solve");
    if (mem > peak_mem)
      peak_mem = mem;
#endif

    dolfinx::common::Timer tcg("ZZZ CG");
    int its = cg.solve(op, x, y, false);
    tcg.stop();

#ifdef ROCM_TRACING
    remove_profiling_annotation("cg solve");
#endif
    if (rank == 0)
    {
      std::cout << "Number of iterations " << its << std::endl;
      std::cout << std::flush;
    }

#ifdef ROCM_TRACING
    add_profiling_annotation("get eigenvalues");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("get eigenvalues");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    std::vector<T> eign = cg.compute_eigenvalues();
    std::sort(eign.begin(), eign.end());
    std::array<T, 2> eig_range = {0.3 * eign.back(), 1.2 * eign.back()};
#ifdef ROCM_TRACING
    remove_profiling_annotation("get eigenvalues");
#endif

    if (rank == 0)
      std::cout << "Eigenvalues:" << eig_range[0] << " - " << eig_range[1] << std::endl;

#ifdef ROCM_TRACING
    add_profiling_annotation("chebyshev solve");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("chebyshev solve");
    if (mem > peak_mem)
      peak_mem = mem;
#endif

    dolfinx::common::Timer tcheb("ZZZ Chebyshev");
    dolfinx::acc::Chebyshev<DeviceVector> cheb(map, 1, eig_range, 3);
    cheb.set_max_iterations(10);
#ifdef ROCM_TRACING
    remove_profiling_annotation("chebyshev solve");
#endif

    T rs = cheb.residual(op, x, y);
    if (rank == 0)
      std::cout << "Cheb resid = " << rs << std::endl;

#ifdef ROCM_TRACING
    add_profiling_annotation("chebyshev solve");
#endif
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("before chebyshev solve");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
    cheb.solve(op, x, y, true);
#ifdef ROCM_SMI
    mem = print_amd_gpu_memory_percentage_used("afterchebyshev solve");
    if (mem > peak_mem)
      peak_mem = mem;
#endif
#ifdef ROCM_TRACING
    remove_profiling_annotation("chebyshev solve");
#endif
    tcheb.stop();

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});

    MPI_Reduce(&peak_mem, &global_peak_mem, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
      std::cout
          << "peak memory used during the run (as a percentage of the total memory available): "
          << global_peak_mem << "%\n";
    }
  }
#ifdef ROCM_SMI
  err = shutdown_rocm_smi();
#endif

  MPI_Finalize();

  return 0;
}
