#include "poisson.h"
#include "src/cg.hpp"
#include "src/chebyshev.hpp"
#include "src/csr.hpp"
#include "src/operators.hpp"
#include "src/vector.hpp"
#include "src/matrix-free.hpp"

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
      dolfinx::fem::CoordinateElement<T> element(mesh::CellType::tetrahedron, 1);
      dolfinx::io::XDMFFile xdmf(MPI_COMM_WORLD, filename, "r");
      mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(
          xdmf.read_mesh(element, mesh::GhostMode::none, "mesh"));
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
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
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

    const int num_cells_local = mesh->topology()->index_map(tdim)->size_local();

    // Input vector
    auto map = V->dofmap()->index_map;
    DeviceVector u(map, 1);
    u.set(T{1.0});

    // Output vector
    DeviceVector y(map, 1);
    y.set(T{0.0});

    // Constants
    // TODO Pack these properly
    std::vector<T> c{2.0};
    thrust::device_vector<T> c_d(c.size());
    thrust::copy(c.data(), c.data() + c.size(), c_d.begin());

    // Coordinate DOFs
    std::span<const T> x = mesh->geometry().x();
    thrust::device_vector<T> x_d(x.size());
    thrust::copy(x.data(), x.data() + x.size(), x_d.begin());

    // Geomerty dofmap
    thrust::device_vector<std::int32_t> x_dofmap_d(mesh->geometry().dofmap().size());
    thrust::copy(mesh->geometry().dofmap().data_handle(),
                 mesh->geometry().dofmap().data_handle() + mesh->geometry().dofmap().size(),
                 x_dofmap_d.begin());

    // V dofmap
    thrust::device_vector<std::int32_t> V_dofmap_d(V->dofmap()->map().size());
    thrust::copy(V->dofmap()->map().data_handle(),
                 V->dofmap()->map().data_handle() + V->dofmap()->map().size(), V_dofmap_d.begin());

    // Make spans
    std::span<T> c_span(thrust::raw_pointer_cast(c_d.data()), c_d.size());
    std::span<T> x_span(thrust::raw_pointer_cast(x_d.data()), x_d.size());
    std::span<std::int32_t> x_dofmap_d_span(thrust::raw_pointer_cast(x_dofmap_d.data()),
                                            x_dofmap_d.size());
    std::span<std::int32_t> V_dofmap_d_span(thrust::raw_pointer_cast(V_dofmap_d.data()),
                                            V_dofmap_d.size());

    MatFreeOp<T> op(c_span, x_span, x_dofmap_d_span, V_dofmap_d_span);

    // acc::MatrixOperator<T> op(a, {bc});
    // auto map = op.column_index_map();

    // fem::Function<T> u(V);
    // la::Vector<T> b(map, 1);

    // b.set(T(0.0));
    // fem::assemble_vector(b.mutable_array(), *L);
    // fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    // b.scatter_rev(std::plus<T>());
    // fem::set_bc<T, T>(b.mutable_array(), {bc});

    // DeviceVector x(map, 1);
    // x.set(T{0.0});

    // DeviceVector y(map, 1);
    // y.copy_from_host(b); // Copy data from host vector to device vector

    // T norm = acc::norm(x);
    // if (rank == 0)
    // {
    //   std::cout << "Norm x vector initial " << norm << std::endl;
    //   std::cout << std::flush;
    // }

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
