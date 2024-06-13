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

// #define MATRIX_FREE

using namespace dolfinx;
using T = double;
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(50000), "number of dofs per rank")(
      "amg", po::bool_switch()->default_value(false),
      "Use AMG solver on coarse level")("csr-interpolation", po::bool_switch()->default_value(true),
                                        "Use CSR matrices to interpolate between levels");

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
  bool use_csr_interpolation = vm["csr-interpolation"].as<bool>();

  std::vector<int> order = {1, 3};
  std::vector form_a = {form_poisson_a1, form_poisson_a3};
  std::vector form_L = {form_poisson_L1, form_poisson_L3};

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::string thread_name = "RANK: " + std::to_string(rank);
    if (rank == 0)
      spdlog::set_level(spdlog::level::info);

    int max_order = order.back();
    double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / max_order;
    std::size_t n0 = static_cast<int>(nx_approx);
    std::array<std::size_t, 3> nx = {n0, n0, n0};

    // Try to improve fit to ndofs +/- 5 in each direction
    if (n0 > 5)
    {
      std::int64_t best_misfit
          = (n0 * max_order + 1) * (n0 * max_order + 1) * (n0 * max_order + 1) - ndofs * size;
      best_misfit = std::abs(best_misfit);
      for (std::size_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
        for (std::size_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
          for (std::size_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
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

      mesh = std::make_shared<mesh::Mesh<T>>(ghost_layer_mesh(base_mesh));
    }

    auto topology = mesh->topology_mutable();
    int tdim = topology->dim();
    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);

    int ncells_local = topology->index_map(tdim)->size_local();
    auto f_to_c = topology->connectivity(fdim, tdim);
    // Create list of cells needed for matrix-free updates
    std::vector<std::int32_t> ip_cells;
    for (std::int32_t f = 0; f < f_to_c->num_nodes(); ++f)
    {
      const auto& cells_f = f_to_c->links(f);
      for (std::int32_t c : cells_f)
      {
        // If facet attached to a ghost cell, add all cells to list
        // FIXME: should this really be via vertex, not facet?
        if (c >= ncells_local)
          ip_cells.insert(ip_cells.end(), cells_f.begin(), cells_f.end());
      }
    }
    std::sort(ip_cells.begin(), ip_cells.end());
    ip_cells.erase(std::unique(ip_cells.begin(), ip_cells.end()), ip_cells.end());
    spdlog::info("Got {} boundary cells.", ip_cells.size());

    // Compute local cells
    std::vector<std::int32_t> local_cells(topology->index_map(tdim)->size_local()
                                          + topology->index_map(tdim)->num_ghosts());
    std::iota(local_cells.begin(), local_cells.end(), 0);
    for (std::int32_t c : ip_cells)
      local_cells[c] = -1;
    std::erase(local_cells, -1);
    spdlog::info("Got {} local cells", local_cells.size());

    // Copy lists of local and boundary cells to device
    thrust::device_vector<std::int32_t> ipcells_device(ip_cells.size());
    spdlog::info("Copy IP_cells : {}", ip_cells.size());
    thrust::copy(ip_cells.begin(), ip_cells.end(), ipcells_device.begin());
    thrust::device_vector<std::int32_t> lcells_device(local_cells.size());
    spdlog::info("Copy local_cells :{}", local_cells.size());
    thrust::copy(local_cells.begin(), local_cells.end(), lcells_device.begin());
    std::span<std::int32_t> ipcells_span(thrust::raw_pointer_cast(ipcells_device.data()),
                                         ipcells_device.size());
    std::span<std::int32_t> lcells_span(thrust::raw_pointer_cast(lcells_device.data()),
                                        lcells_device.size());

    std::vector<std::shared_ptr<fem::FunctionSpace<T>>> V(form_a.size());
    std::vector<std::shared_ptr<fem::Form<T, T>>> a(V.size());
    std::vector<std::shared_ptr<fem::Form<T, T>>> L(V.size());
    std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bcs(V.size());
#ifdef MATRIX_FREE
    std::vector<std::shared_ptr<acc::MatFreeLaplacian<T>>> operators(V.size());
#else
    std::vector<std::shared_ptr<acc::MatrixOperator<T>>> operators(V.size());
#endif
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

    // assemble RHS for each level
    for (std::size_t i = 0; i < V.size(); i++)
    {
      spdlog::info("Build RHS for order {}", order[i]);

      auto f = std::make_shared<fem::Function<T>>(V[i]);
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

      auto Li = std::make_shared<fem::Form<T, T>>(
          fem::create_form<T>(*form_L[i], {V[i]}, {{"w0", f}}, {}, {}));
      L[i] = Li;

      auto dofmap = V[i]->dofmap();
      auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
      bcs[i] = std::make_shared<const fem::DirichletBC<T, T>>(0.0, bdofs, V[i]);
    }

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

    // Copy dofmaps to device
    std::vector<thrust::device_vector<std::int32_t>> dofmapV(order.size());
    std::vector<std::span<std::int32_t>> device_dofmaps;
    for (std::size_t i = 0; i < V.size(); ++i)
    {
      dofmapV[i].resize(V[i]->dofmap()->map().size());
      spdlog::info("Copy dofmap (V{}) : {}", i, dofmapV[i].size());
      thrust::copy(V[i]->dofmap()->map().data_handle(),
                   V[i]->dofmap()->map().data_handle() + V[i]->dofmap()->map().size(),
                   dofmapV[i].begin());
      device_dofmaps.push_back(
          std::span<std::int32_t>(thrust::raw_pointer_cast(dofmapV[i].data()), dofmapV[i].size()));
    }

    // Copy geometry to device
    thrust::device_vector<T> geomx_device(mesh->geometry().x().size());
    spdlog::info("Copy geometry to device :{}", geomx_device.size());
    thrust::copy(mesh->geometry().x().begin(), mesh->geometry().x().end(), geomx_device.begin());
    std::span<T> geom_x(thrust::raw_pointer_cast(geomx_device.data()), geomx_device.size());

    thrust::device_vector<std::int32_t> geomx_dofmap_device(mesh->geometry().dofmap().size());
    spdlog::info("Copy geometry to device :{}", geomx_dofmap_device.size());
    thrust::copy(mesh->geometry().dofmap().data_handle(),
                 mesh->geometry().dofmap().data_handle() + mesh->geometry().dofmap().size(),
                 geomx_dofmap_device.begin());
    std::span<std::int32_t> geom_x_dofmap(thrust::raw_pointer_cast(geomx_dofmap_device.data()),
                                          geomx_dofmap_device.size());

    // Copy constants to device
    thrust::device_vector<T> constants(kappa->value.begin(), kappa->value.end());
    std::span<T> device_constants(thrust::raw_pointer_cast(constants.data()), constants.size());

    // Copy bc_dofs to device (list of all dofs, with BCs marked with 0)
    std::vector<thrust::device_vector<std::int8_t>> device_bc_dofs(V.size());
    for (std::size_t i = 0; i < V.size(); ++i)
    {
      auto [dofs, pos] = bcs[i]->dof_indices();
      std::vector<std::int8_t> active_bc_dofs(
          V[i]->dofmap()->index_map->size_local() + V[i]->dofmap()->index_map->num_ghosts(), 1);
      for (std::int32_t index : dofs)
        active_bc_dofs[index] = 0;
      device_bc_dofs[i]
          = thrust::device_vector<std::int8_t>(active_bc_dofs.begin(), active_bc_dofs.end());
    }

    std::vector<std::shared_ptr<DeviceVector>> bs(V.size());
    for (std::size_t i = 0; i < V.size(); i++)
    {
      std::shared_ptr<fem::Form<T, T>> a_i = a[i];
      std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bc_i = {bcs[i]};

#ifdef MATRIX_FREE
      int degree = i + 1;
      maps[i] = V[i]->dofmap()->index_map;
      int num_cells = mesh->topology()->index_map(tdim)->size_local()
                      + mesh->topology()->index_map(tdim)->num_ghosts();

      spdlog::info("Num cells = {}", num_cells);

      spdlog::info("Create operator on V[{}]", i);
      std::span<const std::int8_t> bc_span(thrust::raw_pointer_cast(device_bc_dofs[i].data()),
                                           device_bc_dofs[i].size());
      operators[i] = std::make_shared<acc::MatFreeLaplacian<T>>(
          degree, lcells_span, ipcells_span, device_constants, geom_x, geom_x_dofmap,
          device_dofmaps[i], bc_span);
#else
      operators[i] = std::make_shared<acc::MatrixOperator<T>>(a_i, bc_i);
      maps[i] = operators[i]->column_index_map();
#endif

      la::Vector<T> b(maps[i], 1);
      b.set(T(0.0));
      fem::assemble_vector(b.mutable_array(), *L[i]);

      fem::apply_lifting<T, T>(b.mutable_array(), {a[i]}, {{bcs[i]}}, {}, T(1));
      b.scatter_rev(std::plus<T>());
      fem::set_bc<T, T>(b.mutable_array(), {bcs[i]});

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
      cg.set_tolerance(1e-5);
      cg.store_coefficients(true);

      DeviceVector x(maps[i], 1);
      x.set(T{0.0});

      //      (*operators[i])(*bs[i], x);

      [[maybe_unused]] int its = cg.solve(*operators[i], x, *bs[i], false);
      std::vector<T> eign = cg.compute_eigenvalues();
      std::sort(eign.begin(), eign.end());
      std::array<T, 2> eig_range = {0.1 * eign.back(), 1.1 * eign.back()};
      smoothers[i] = std::make_shared<acc::Chebyshev<DeviceVector>>(maps[i], 1, eig_range);

      spdlog::info("Eigenvalues level {}: {} - {}", i, eign.front(), eign.back());
    }

    smoothers[0]->set_max_iterations(10);
    smoothers[1]->set_max_iterations(5);

    // Create Prolongation operator
    std::vector<std::shared_ptr<acc::MatrixOperator<T>>> prolongation(V.size() - 1);

    // Interpolation and prolongation kernels
    std::vector<std::shared_ptr<Interpolator<T>>> int_kerns(V.size() - 1);
    std::vector<std::shared_ptr<Interpolator<T>>> prolong_kerns(V.size() - 1);

    // From V1 to V0
    if (use_csr_interpolation)
    {
      spdlog::warn("Creating Prolongation Operators");
      for (int i = 0; i < V.size() - 1; ++i)
        prolongation[i] = std::make_shared<acc::MatrixOperator<T>>(*V[i], *V[i + 1]);
    }
    else
    {
      // These are alternative restriction/prolongation kernels, which should replace the CSR
      // matrices when fully working

      // auto interpolator_V1_V0 = std::make_shared<Interpolator<T>>(
      //     V[1]->element()->basix_element(), V[0]->element()->basix_element(),
      //     device_dofmaps[1], device_dofmaps[0], ipcells_span, lcells_span, false);
      // auto interpolator_V2_V1 = std::make_shared<Interpolator<T>>(
      //     V[2]->element()->basix_element(), V[1]->element()->basix_element(),
      //     device_dofmaps[2], device_dofmaps[1], ipcells_span, lcells_span, false);
      // auto interpolator_V0_V1 = std::make_shared<Interpolator<T>>(
      //     V[0]->element()->basix_element(), V[1]->element()->basix_element(),
      //     device_dofmaps[0], device_dofmaps[1], ipcells_span, lcells_span, false);
      // auto interpolator_V1_V2 = std::make_shared<Interpolator<T>>(
      //     V[1]->element()->basix_element(), V[2]->element()->basix_element(),
      //     device_dofmaps[1], device_dofmaps[2], ipcells_span, lcells_span, false);

      // int_kerns = {interpolator_V1_V0, interpolator_V2_V1};
      // prolong_kerns = {interpolator_V0_V1, interpolator_V1_V2};
    }

#ifdef MATRIX_FREE
    using OpType = acc::MatFreeLaplacian<T>;
#else
    using OpType = acc::MatrixOperator<T>;
#endif
    using CSRType = acc::MatrixOperator<T>;
    using SolverType = acc::Chebyshev<DeviceVector>;

    using PMG = acc::MultigridPreconditioner<DeviceVector, OpType, CSRType, CSRType, SolverType,
                                             CoarseSolverType<T>>;

    spdlog::info("Create PMG");
    PMG pmg(maps, 1);
    pmg.set_solvers(smoothers);
    pmg.set_operators(operators);
    spdlog::info("Set Coarse Solver");
    pmg.set_coarse_solver(coarse_solver);

    // Sets CSR matrices or matrix-free kernels to do interpolation
    pmg.set_interpolators(prolongation);

    pmg.set_interpolation_kernels(int_kerns);
    pmg.set_prolongation_kernels(prolong_kerns);

    // Create solution vector
    spdlog::info("Create x");
    DeviceVector x(maps.back(), 1);
    x.set(T{0.0});

    for (int i = 0; i < 10; i++)
    {
      pmg.apply(*bs.back(), x, true);
      // spdlog::info("------ end of iteration ------");
    }

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

  PetscFinalize();
  return 0;
}
