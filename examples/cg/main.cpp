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

#ifdef ROCM_TRACING
#include <roctx.h>
#endif
#ifdef ROCM_SMI
#include "src/amd_gpu.hpp"
#endif

using namespace dolfinx;
using T = double;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  int err;
  uint32_t num_devices;

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

#ifdef ROCM_SMI
    err = initialise_rocm_smi();
    num_devices = num_monitored_devices();
    std::cout << "MPI rank " << rank << " can see " << num_devices << " AMD GPUs\n";
    print_amd_gpu_memory_used("Beginning");
#endif

#ifdef ROCM_TRACING
    if (rank == 0)
    {
      std::cout << "Using roctx tracing ranges for visulising trace data\n";
    }
#endif
    // Create a hexahedral mesh
#ifdef ROCM_TRACING
    roctxRangePush("making mesh");
    print_amd_gpu_memory_used("making mesh");

#endif

    std::shared_ptr<mesh::Mesh<T>> mesh;

    if (filename.size() > 0)
    {
      dolfinx::io::XDMFFile xdmf(MPI_COMM_WORLD, filename, "r");
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
    roctxRangePop();
#endif

#ifdef ROCM_TRACING
    roctxRangePush("making V");
    print_amd_gpu_memory_used("making V");
#endif
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

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
      std::cout << "-----------------------------------\n";
      std::cout << std::flush;
    }

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
#ifdef ROCM_TRACING
    roctxRangePush("making forms");
    print_amd_gpu_memory_used("making forms");
#endif
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
#ifdef ROCM_TRACING
    roctxRangePop();
#endif
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> out;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
          }

          return {out, {out.size()}};
        });

#ifdef ROCM_TRACING
    roctxRangePush("doing topology");
    print_amd_gpu_memory_used("doing topology");
#endif
    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

#ifdef ROCM_TRACING
    roctxRangePush("doing boundary conditions");
    print_amd_gpu_memory_used("doing boundary conditions");
#endif
    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
    acc::MatrixOperator<T> op(a, {bc});
    auto map = op.index_map();

    fem::Function<T> u(V);
    la::Vector<T> b(map, 1);

#ifdef ROCM_TRACING
    roctxRangePush("assembling and scattering");
    print_amd_gpu_memory_used("assembling and scattering");
#endif
    b.set(T(0.0));
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, T>(b.mutable_array(), {bc});
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

#ifdef ROCM_TRACING
    roctxRangePush("setup device x");
    print_amd_gpu_memory_used("setup device x");
#endif
    DeviceVector x(map, 1);
    x.set(T{0.0});
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

#ifdef ROCM_TRACING
    roctxRangePush("setup device y");
    print_amd_gpu_memory_used("setup device y");
#endif
    DeviceVector y(map, 1);
    y.copy_from_host(b); // Copy data from host vector to device vector
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

#ifdef ROCM_TRACING
    roctxRangePush("matrix operator");
    print_amd_gpu_memory_used("matrix operator");
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
    roctxRangePop();
#endif

    // Create distributed CG solver
#ifdef ROCM_TRACING
    roctxRangePush("creating cg solver");
    print_amd_gpu_memory_used("creating cg solver");
#endif
    dolfinx::acc::CGSolver<DeviceVector> cg(map, 1);
    cg.set_max_iterations(30);
    cg.set_tolerance(1e-5);
    cg.store_coefficients(true);
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

    // Solve
#ifdef ROCM_TRACING
    roctxRangePush("cg solve");
    print_amd_gpu_memory_used("before cg solve");
#endif

    dolfinx::common::Timer tcg("ZZZ CG");
    int its = cg.solve(op, x, y, false);
    tcg.stop();

#ifdef ROCM_TRACING
    roctxRangePop();
#endif
    if (rank == 0)
    {
      std::cout << "Number of iterations " << its << std::endl;
      std::cout << std::flush;
    }

#ifdef ROCM_TRACING
    roctxRangePush("get eigenvalues");
    print_amd_gpu_memory_used("get eigenvalues");
#endif
    std::vector<T> eign = cg.compute_eigenvalues();
    std::sort(eign.begin(), eign.end());
    std::array<T, 2> eig_range = {0.3 * eign.back(), 1.2 * eign.back()};
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

    if (rank == 0)
      std::cout << "Eigenvalues:" << eig_range[0] << " - " << eig_range[1] << std::endl;

#ifdef ROCM_TRACING
    roctxRangePush("chebyshev solve");
    print_amd_gpu_memory_used("chebyshev solve");
#endif

    dolfinx::common::Timer tcheb("ZZZ Chebyshev");
    dolfinx::acc::Chebyshev<DeviceVector> cheb(map, 1, eig_range, 3);
    cheb.set_max_iterations(10);
#ifdef ROCM_TRACING
    roctxRangePop();
#endif

    T rs = cheb.residual(op, x, y);
    if (rank == 0)
      std::cout << "Cheb resid = " << rs << std::endl;

#ifdef ROCM_TRACING
    roctxRangePush("chebyshev solve");
    print_amd_gpu_memory_usage("chebyshev solve");
#endif
    cheb.solve(op, x, y, true);

#ifdef ROCM_TRACING
    roctxRangePop();
#endif
    tcheb.stop();

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

#ifdef ROCM_SMI
  err = shutdown_rocm_smi();
#endif

  MPI_Finalize();

  return 0;
}
