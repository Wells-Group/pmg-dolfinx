#include "poisson.h"
#include "src/cg.hpp"
#include "src/chebyshev.hpp"
#include "src/csr.hpp"
#include "src/laplacian.hpp"
#include "src/mesh.hpp"
#include "src/operators.hpp"
#include "src/vector.hpp"

#include <array>
#include <basix/e-lagrange.h>
#include <basix/quadrature.h>
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
  float peak_mem = 0.0;
  float global_peak_mem = 0.0;

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
  spdlog::set_level(spdlog::level::info);

  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Create a hexahedral mesh
    const int order = 3;
    double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / order;
    std::int64_t n0 = static_cast<std::int64_t>(nx_approx);
    std::array<std::int64_t, 3> nx = {n0, n0, n0};

    // Try to improve fit to ndofs +/- 5 in each direction
    if (n0 > 5)
    {
      std::int64_t best_misfit
          = (n0 * order + 1) * (n0 * order + 1) * (n0 * order + 1) - ndofs * size;
      best_misfit = std::abs(best_misfit);
      for (std::int64_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
        for (std::int64_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
          for (std::int64_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
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
    spdlog::warn("Mesh size {}x{}x{}", nx[0], nx[1], nx[2]);

    // Tensor product element
    auto element = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false));

    // First order coordinate element
    auto element_1 = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, 1,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false));
    dolfinx::fem::CoordinateElement<T> coord_element(element_1);

    // Create mesh with overlap region
    std::shared_ptr<mesh::Mesh<T>> mesh;
    {
      mesh::Mesh<T> base_mesh = mesh::create_box<T>(
          comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]}, mesh::CellType::hexahedron);
      mesh = std::make_shared<mesh::Mesh<T>>(ghost_layer_mesh(base_mesh, coord_element));
    }

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, *element, {}));
    auto [lcells, bcells] = compute_boundary_cells(V);

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
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
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

    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);

    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;

    // Copy data to GPU
    // Constants
    // TODO Pack these properly
    const int num_cells_all = mesh->topology()->index_map(tdim)->size_local()
                              + mesh->topology()->index_map(tdim)->num_ghosts();
    thrust::device_vector<T> constants_d(num_cells_all, kappa->value[0]);
    std::span<const T> constants_d_span(thrust::raw_pointer_cast(constants_d.data()),
                                        constants_d.size());

    // V dofmap
    thrust::device_vector<std::int32_t> dofmap_d(
        dofmap->map().data_handle(), dofmap->map().data_handle() + dofmap->map().size());
    std::span<const std::int32_t> dofmap_d_span(thrust::raw_pointer_cast(dofmap_d.data()),
                                                dofmap_d.size());
    spdlog::info("Send dofmap to GPU (size = {})", dofmap_d.size());

    auto map = V->dofmap()->index_map;

    // Put geometry information onto device
    const fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();
    auto xdofmap = mesh->geometry().dofmap();

    // Geometry dofmap
    thrust::device_vector<std::int32_t> xdofmap_d(xdofmap.data_handle(),
                                                  xdofmap.data_handle() + xdofmap.size());
    std::span<const std::int32_t> xdofmap_d_span(thrust::raw_pointer_cast(xdofmap_d.data()),
                                                 xdofmap_d.size());
    // Geometry points
    thrust::device_vector<T> xgeom_d(mesh->geometry().x().begin(), mesh->geometry().x().end());
    std::span<const T> xgeom_d_span(thrust::raw_pointer_cast(xgeom_d.data()), xgeom_d.size());

    // dphi tables at quadrature points
    // Quadrature points and weights on hex (3D)
    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::gll, basix::cell::type::hexahedron, basix::polyset::type::standard,
        order + 1);

    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, Gweights.size());
    std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

    // Copy dphi to device (skipping phi in table)
    thrust::device_vector<T> dphi_d(phi_b.begin() + phi_b.size() / 4, phi_b.end());
    std::span<const T> dphi_d_span(thrust::raw_pointer_cast(dphi_d.data()), dphi_d.size());

    // Copy quadrature weights to device
    thrust::device_vector<T> Gweights_d(Gweights.begin(), Gweights.end());
    std::span<const T> Gweights_d_span(thrust::raw_pointer_cast(Gweights_d.data()),
                                       Gweights_d.size());

    const int num_dofs = map->size_local() + map->num_ghosts();
    std::vector<std::int8_t> bc_marker(num_dofs, 0);
    bc->mark_dofs(bc_marker);
    thrust::device_vector<std::int8_t> bc_marker_d(bc_marker.begin(), bc_marker.end());
    std::span<const std::int8_t> bc_marker_d_span(thrust::raw_pointer_cast(bc_marker_d.data()),
                                                  bc_marker_d.size());

    // Create operators
    acc::MatFreeLaplacian<T> op(3, constants_d_span, dofmap_d_span, xgeom_d_span, xdofmap_d_span,
                                dphi_d_span, Gweights_d_span, lcells, bcells, bc_marker_d_span);

    acc::MatrixOperator<T> op_mat(a, {bc});

    // Set diagonal for Jacobi preconditioing
    DeviceVector diag_inv(map, 1);
    op_mat.get_diag_inverse(diag_inv);
    op.set_diag_inverse(diag_inv);

    // Assemble RHS
    la::Vector<T> b(map, 1);
    // b.set(1.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, T>(b.mutable_array(), {bc});

    DeviceVector u(map, 1);
    u.set(T{1.0});
    u.scatter_fwd();

    DeviceVector y(map, 1);
    y.set(T{0.0});

    // Create distributed CG solver
    dolfinx::acc::CGSolver<DeviceVector> cg(map, 1);
    cg.set_max_iterations(20);
    cg.set_tolerance(1e-6);
    cg.store_coefficients(true);

    // Solve
    dolfinx::common::Timer tcg("ZZZ CG");
    int its = cg.solve(op, y, u, true);
    tcg.stop();

    if (rank == 0)
    {
      std::cout << "Number of iterations " << its << std::endl;
      std::cout << std::flush;
    }

    std::vector<T> eign = cg.compute_eigenvalues();
    std::sort(eign.begin(), eign.end());
    std::cout << "Computed eigs = (" << eign.front() << ", " << eign.back() << ")\n";
    std::array<T, 2> eig_range = {0.1 * eign.back(), 1.1 * eign.back()};

    if (rank == 0)
      std::cout << "Using eig range:" << eig_range[0] << " - " << eig_range[1] << std::endl;

    dolfinx::common::Timer tcheb("ZZZ Chebyshev");
    dolfinx::acc::Chebyshev<DeviceVector> cheb(map, 1, eig_range);
    cheb.set_max_iterations(30);

    // Try non-zero initial guess to make sure that works OK
    y.set(1.0);
    u.copy_from_host(b); // Copy data from host vector to device vector
                         //     err_check(hipDeviceSynchronize());
    fem::set_bc<T, T>(y.mutable_array(), {bc});
    err_check(hipDeviceSynchronize());

    cheb.solve(op, y, u, true);

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

  MPI_Finalize();

  return 0;
}
