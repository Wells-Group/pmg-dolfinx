#include "../../src/amg.hpp"
#include "../../src/cg.hpp"
#include "../../src/chebyshev.hpp"
#include "../../src/csr.hpp"
#include "../../src/laplacian.hpp"
#include "../../src/mesh.hpp"
#include "../../src/operators.hpp"
#include "../../src/pmg.hpp"
#include "../../src/precompute.hpp"
#include "../../src/vector.hpp"
#include "poisson.h"

#include <thrust/device_vector.h>

#include <array>
#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace dolfinx;
using T = double;
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
namespace po = boost::program_options;

template <typename FineOperator>
void solve(std::shared_ptr<mesh::Mesh<double>> mesh, bool use_amg)
{
  if constexpr (std::is_same_v<FineOperator, acc::MatFreeLaplacian<T>>)
  {
    spdlog::info("------- MatFree -------");
  }
  else
  {
    spdlog::info("------- CSR -------");
  }

  int rank = dolfinx::MPI::rank(mesh->comm());
  int size = dolfinx::MPI::size(mesh->comm());
  std::vector<int> order = {1, 3};
  std::vector form_a = {form_poisson_a1, form_poisson_a3};
  std::vector form_L = {form_poisson_L1, form_poisson_L3};

  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  spdlog::debug("Create facets");
  topology->create_connectivity(fdim, tdim);

  std::vector<std::shared_ptr<fem::FunctionSpace<T>>> V(form_a.size());
  std::vector<std::shared_ptr<fem::Form<T, T>>> a(V.size());
  std::vector<std::shared_ptr<fem::Form<T, T>>> L(V.size());
  std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bcs(V.size());

  // List of LHS operators (CSR or MatrixFree), one for each level.
  std::vector<std::shared_ptr<FineOperator>> operators(V.size());

  std::vector<std::shared_ptr<const common::IndexMap>> maps(V.size());

  auto facets = dolfinx::mesh::exterior_facet_indices(*topology);

  std::vector<std::size_t> ndofs(V.size());

  // Prepare and set Constants for the bilinear form
  auto kappa = std::make_shared<fem::Constant<T>>(2.0);
  for (std::size_t i = 0; i < form_a.size(); i++)
  {
    spdlog::info("Creating FunctionSpace at order {}", order[i]);
    auto element = basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order[i],
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false);

    V[i] = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element, {}));

    ndofs[i] = V[i]->dofmap()->index_map->size_global();
    a[i] = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_a[i], {V[i], V[i]}, {}, {{"c0", kappa}}, {}));
  }

  spdlog::info("Compute boundary cells");
  // Compute local and boundary cells (needed for MatFreeLaplacian)
  auto [lcells, bcells] = compute_boundary_cells(V.back());

  // assemble RHS for each level
  for (std::size_t i = 0; i < V.size(); i++)
  {
    spdlog::info("Build RHS for order {}", order[i]);

    // auto f = std::make_shared<fem::Function<T>>(V[i]);
    // f->interpolate(
    //     [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    //     {
    //       std::vector<T> out;
    //       for (std::size_t p = 0; p < x.extent(1); ++p)
    //       {
    //         auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
    //         auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
    //         out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
    //       }

    //       return {out, {out.size()}};
    //     });

    L[i] = std::make_shared<fem::Form<T, T>>(
        fem::create_form<T>(*form_L[i], {V[i]}, {}, {{"c0", kappa}}, {}));

    auto dofmap = V[i]->dofmap();
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    bcs[i] = std::make_shared<const fem::DirichletBC<T, T>>(0.0, bdofs, V[i]);
  }

  // If not using AMG coarse_solver will be a nullptr, and operators[0] will be applied
  std::shared_ptr<CoarseSolverType<T>> coarse_solver;
  if (use_amg)
    coarse_solver = std::make_shared<CoarseSolverType<T>>(a[0], bcs[0]);

  // RHS
  std::size_t ncells = mesh->topology()->index_map(3)->size_global();
  if (rank == 0)
  {
    std::cout << "-----------------------------------\n";
    std::cout << "Number of ranks : " << size << "\n";
    std::cout << "Number of cells-global : " << ncells << "\n";
    std::cout << "Number of dofs-global : " << ndofs.back() << "\n";
    std::cout << "Number of cells-rank : " << ncells / size << "\n";
    std::cout << "Number of dofs-rank : " << ndofs.back() / size << "\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Hierarchy: " << std::endl;
    for (std::size_t i = 0; i < ndofs.size(); i++)
    {
      std::cout << "Level " << i << ": " << ndofs[i] << "\n";
    }
    std::cout << "-----------------------------------\n";
    std::cout << std::flush;
  }

  // Data for required quantities for MatFreeLaplacian:

  // Dofmaps for each level
  std::vector<thrust::device_vector<std::int32_t>> dofmapV(V.size());
  std::vector<std::span<std::int32_t>> device_dofmaps;

  // Geometry
  thrust::device_vector<T> geomx_device;
  std::span<T> geom_x;
  thrust::device_vector<std::int32_t> geomx_dofmap_device;
  std::span<std::int32_t> geom_x_dofmap;
  std::vector<thrust::device_vector<T>> geometry_dphi_d(V.size());
  std::vector<std::span<const T>> geometry_dphi_d_span;
  std::vector<thrust::device_vector<T>> Gweights_d(V.size());
  std::vector<std::span<const T>> Gweights_d_span;

  // BCs
  std::vector<thrust::device_vector<std::int8_t>> bc_marker_d(V.size());
  std::vector<std::span<const std::int8_t>> bc_marker_d_span;

  // Copy bc_dofs to device (list of all dofs, with BCs marked with 0)
  for (std::size_t i = 0; i < V.size(); ++i)
  {
    spdlog::debug("Copy BCs[{}] to device", i);
    auto [dofs, pos] = bcs[i]->dof_indices();
    std::vector<std::int8_t> active_bc_dofs(
        V[i]->dofmap()->index_map->size_local() + V[i]->dofmap()->index_map->num_ghosts(), 0);
    for (std::int32_t index : dofs)
      active_bc_dofs[index] = 1;
    bc_marker_d[i]
        = thrust::device_vector<std::int8_t>(active_bc_dofs.begin(), active_bc_dofs.end());
    bc_marker_d_span.push_back(
        std::span(thrust::raw_pointer_cast(bc_marker_d[i].data()), bc_marker_d[i].size()));
  }

  // Copy constants to device (all same, one per cell, scalar)
  thrust::device_vector<T> constants(mesh->topology()->index_map(tdim)->size_local()
                                         + mesh->topology()->index_map(tdim)->num_ghosts(),
                                     kappa->value[0]);
  std::span<T> device_constants(thrust::raw_pointer_cast(constants.data()), constants.size());

  if constexpr (std::is_same_v<FineOperator, acc::MatFreeLaplacian<T>>)
  {
    // Copy dofmaps to device (only for MatFreeLaplacian)
    std::vector<thrust::device_vector<std::int32_t>> dofmapV(order.size());

    for (std::size_t i = 0; i < V.size(); ++i)
    {
      dofmapV[i].resize(V[i]->dofmap()->map().size());
      spdlog::debug("Copy dofmap (V{}) : {}", i, dofmapV[i].size());
      thrust::copy(V[i]->dofmap()->map().data_handle(),
                   V[i]->dofmap()->map().data_handle() + V[i]->dofmap()->map().size(),
                   dofmapV[i].begin());
      device_dofmaps.push_back(
          std::span<std::int32_t>(thrust::raw_pointer_cast(dofmapV[i].data()), dofmapV[i].size()));
    }

    for (std::size_t i = 0; i < V.size(); ++i)
    {
      spdlog::debug("Copy geometry quadrature tables to device [{}]", i);
      // Quadrature points and weights on hex (3D)
      auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
          basix::quadrature::type::gll, basix::cell::type::hexahedron,
          basix::polyset::type::standard, order[i]);
      // Tables at quadrature points [phi, dphix, dphiy, dphiz]
      const fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();
      std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, Gweights.size());
      std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
      cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

      // Copy dphi to device (skipping phi in table)
      geometry_dphi_d[i].resize(phi_b.size() * 3 / 4);
      thrust::copy(phi_b.begin() + phi_b.size() / 4, phi_b.end(), geometry_dphi_d[i].begin());
      geometry_dphi_d_span.push_back(std::span(thrust::raw_pointer_cast(geometry_dphi_d[i].data()),
                                               geometry_dphi_d[i].size()));

      // Copy quadrature weights to device
      Gweights_d[i].resize(Gweights.size());
      thrust::copy(Gweights.begin(), Gweights.end(), Gweights_d[i].begin());
      Gweights_d_span.push_back(
          std::span(thrust::raw_pointer_cast(Gweights_d[i].data()), Gweights_d[i].size()));
    }

    spdlog::debug("Copy geometry data to device");
    geomx_device.resize(mesh->geometry().x().size());
    spdlog::info("Copy geometry to device :{}", geomx_device.size());
    thrust::copy(mesh->geometry().x().begin(), mesh->geometry().x().end(), geomx_device.begin());
    geom_x = std::span<T>(thrust::raw_pointer_cast(geomx_device.data()), geomx_device.size());

    geomx_dofmap_device.resize(mesh->geometry().dofmap().size());
    thrust::copy(mesh->geometry().dofmap().data_handle(),
                 mesh->geometry().dofmap().data_handle() + mesh->geometry().dofmap().size(),
                 geomx_dofmap_device.begin());
    geom_x_dofmap = std::span<std::int32_t>(thrust::raw_pointer_cast(geomx_dofmap_device.data()),
                                            geomx_dofmap_device.size());
  }

  std::vector<std::shared_ptr<DeviceVector>> bs(V.size());
  for (std::size_t i = 0; i < V.size(); i++)
  {
    std::shared_ptr<fem::Form<T, T>> a_i = a[i];
    std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bc_i = {bcs[i]};

    if constexpr (std::is_same_v<FineOperator, acc::MatFreeLaplacian<T>>)
    {
      maps[i] = V[i]->dofmap()->index_map;

      spdlog::info("Create operator on V[{}]", i);
      operators[i] = std::make_shared<acc::MatFreeLaplacian<T>>(
          order[i], device_constants, device_dofmaps[i], geom_x, geom_x_dofmap,
          geometry_dphi_d_span[i], Gweights_d_span[i], lcells, bcells, bc_marker_d_span[i]);

      // FIXME: do this better
      // Compute CSR matrix, to get diagonal for MatFree
      acc::MatrixOperator<T> A(a_i, bc_i);
      DeviceVector diag_inv(maps[i], 1);
      A.get_diag_inverse(diag_inv);
      operators[i]->set_diag_inverse(diag_inv);
    }
    else
    {
      operators[i] = std::make_shared<acc::MatrixOperator<T>>(a_i, bc_i);
      maps[i] = operators[i]->column_index_map();
    }

    la::Vector<T> b(maps[i], 1);
    b.set(T(0.0));
    fem::assemble_vector(b.mutable_array(), *L[i]);

    fem::apply_lifting<T, T>(b.mutable_array(), {a[i]}, {{bcs[i]}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, T>(b.mutable_array(), {bcs[i]});

    spdlog::info("b[{}].norm = {}", i, dolfinx::la::norm(b));

    bs[i] = std::make_shared<DeviceVector>(maps[i], 1);
    bs[i]->copy_from_host(b);
  }

  spdlog::info("Create Chebyshev smoothers");

  // Create chebyshev smoother for each level
  std::vector<std::shared_ptr<acc::Chebyshev<DeviceVector>>> smoothers(V.size());
  for (std::size_t i = 0; i < V.size(); i++)
  {
    dolfinx::acc::CGSolver<DeviceVector> cg(maps[i], 1);
    cg.set_max_iterations(20);
    cg.set_tolerance(1e-6);
    cg.store_coefficients(true);

    DeviceVector x(maps[i], 1);
    x.set(T{0.0});
    DeviceVector y(maps[i], 1);
    y.set(T{1.0});

    [[maybe_unused]] int its = cg.solve(*operators[i], x, y, false);
    std::vector<T> eign = cg.compute_eigenvalues();
    std::sort(eign.begin(), eign.end());
    spdlog::info("Eigenvalues level {}: {} - {}", i, eign.front(), eign.back());
    std::array<T, 2> eig_range = {0.1 * eign.back(), 1.1 * eign.back()};
    smoothers[i] = std::make_shared<acc::Chebyshev<DeviceVector>>(maps[i], 1, eig_range);
    smoothers[i]->set_max_iterations(2);
  }

  // Create Prolongation operator
  std::vector<std::shared_ptr<acc::MatrixOperator<T>>> prolongation(V.size() - 1);

  // From V1 to V0
  spdlog::warn("Creating Prolongation Operators");
  for (int i = 0; i < V.size() - 1; ++i)
    prolongation[i] = std::make_shared<acc::MatrixOperator<T>>(*V[i], *V[i + 1]);

  using CSRType = acc::MatrixOperator<T>;
  using SolverType = acc::Chebyshev<DeviceVector>;

  using PMG = acc::MultigridPreconditioner<DeviceVector, FineOperator, CSRType, CSRType, SolverType,
                                           CoarseSolverType<T>>;

  spdlog::info("Create PMG");
  PMG pmg(maps, 1, bc_marker_d_span[0]);
  pmg.set_solvers(smoothers);
  pmg.set_operators(operators);
  spdlog::info("Set Coarse Solver");
  pmg.set_coarse_solver(coarse_solver);

  // Sets CSR matrices or matrix-free kernels to do interpolation
  pmg.set_interpolators(prolongation);

  // Create solution vector
  spdlog::info("Create x");
  DeviceVector x(maps.back(), 1);
  x.set(T{0.0});

  for (int i = 0; i < 10; i++)
  {
    pmg.apply(*bs.back(), x, true);
    // spdlog::info("------ end of iteration ------");
  }
}

int main(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(50000),
      "number of dofs per rank")("amg", po::bool_switch()->default_value(false));

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  bool use_amg = vm["amg"].as<bool>();

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int size = 0;
    MPI_Comm_size(comm, &size);

    int max_order = 3; // FIXME

    double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / max_order;
    std::int64_t n0 = static_cast<int>(nx_approx);
    std::array<std::int64_t, 3> nx = {n0, n0, n0};

    // Try to improve fit to ndofs +/- 5 in each direction
    if (n0 > 5)
    {
      std::int64_t best_misfit
          = (n0 * max_order + 1) * (n0 * max_order + 1) * (n0 * max_order + 1) - ndofs * size;
      best_misfit = std::abs(best_misfit);
      for (std::int64_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
        for (std::int64_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
          for (std::int64_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
          {
            std::int64_t misfit
                = (nx0 * max_order + 1) * (ny0 * max_order + 1) * (nz0 * max_order + 1)
                  - ndofs * size;
            if (std::abs(misfit) < best_misfit)
            {
              best_misfit = std::abs(misfit);
              nx = {nx0, ny0, nz0};
            }
          }
    }

    spdlog::info("Creating mesh of size: {}x{}x{}", nx[0], nx[1], nx[2]);

    // Create mesh
    std::shared_ptr<mesh::Mesh<T>> mesh;
    {
      mesh::Mesh<T> base_mesh = mesh::create_box<T>(
          comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]}, mesh::CellType::hexahedron);

      // First order coordinate element
      auto element_1 = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
          basix::element::family::P, basix::cell::type::hexahedron, 1,
          basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false));
      dolfinx::fem::CoordinateElement<T> coord_element(element_1);

      mesh = std::make_shared<mesh::Mesh<T>>(ghost_layer_mesh(base_mesh, coord_element));
    }

    // Solve using Matrix-free operators
    // solve<acc::MatFreeLaplacian<T>>(mesh, use_amg);

    // Solve using CSR matrices
    solve<acc::MatrixOperator<T>>(mesh, use_amg);

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

  PetscFinalize();
  return 0;
}
