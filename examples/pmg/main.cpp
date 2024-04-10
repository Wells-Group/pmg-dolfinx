#include "../../src/cg.hpp"
#include "../../src/chebyshev.hpp"
#include "../../src/csr.hpp"
#include "../../src/pmg.hpp"
#include "../../src/vector.hpp"
#include "poisson.h"

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

int main(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(50000), "number of dofs per rank");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();

  std::vector fs_poisson_a = {functionspace_form_poisson_a1, functionspace_form_poisson_a2,
                              functionspace_form_poisson_a3};
  std::vector form_a = {form_poisson_a1, form_poisson_a2, form_poisson_a3};
  std::vector form_L = {form_poisson_L1, form_poisson_L2, form_poisson_L2};

  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //    if (rank == 0)
    //      loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

    const int order = 3;
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

    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
        comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]}, mesh::CellType::hexahedron));
    auto topology = mesh->topology_mutable();
    int tdim = topology->dim();
    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);

    std::vector<std::shared_ptr<fem::FunctionSpace<T>>> V(form_a.size());
    std::vector<std::shared_ptr<fem::Form<T, T>>> a(V.size());
    std::vector<std::shared_ptr<fem::Form<T, T>>> L(V.size());
    std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bcs(V.size());
    std::vector<std::shared_ptr<acc::MatrixOperator<T>>> operators(V.size());
    std::vector<std::shared_ptr<const common::IndexMap>> maps(V.size());

    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    std::vector<std::size_t> ndofs(V.size());

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    for (std::size_t i = 0; i < form_a.size(); i++)
    {
      V[i] = std::make_shared<fem::FunctionSpace<T>>(
          fem::create_functionspace(fs_poisson_a[i], "v_0", mesh));
      ndofs[i] = V[i]->dofmap()->index_map->size_global();
      a[i] = std::make_shared<fem::Form<T>>(
          fem::create_form<T>(*form_a[i], {V[i], V[i]}, {}, {{"c0", kappa}}, {}));
    }

    // assemble RHS for each level
    for (std::size_t i = 0; i < V.size(); i++)
    {
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

    std::vector<std::shared_ptr<DeviceVector>> bs(V.size());

    for (std::size_t i = 0; i < V.size(); i++)
    {
      std::shared_ptr<fem::Form<T, T>> a_i = a[i];
      std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bc_i = {bcs[i]};
      operators[i] = std::make_shared<acc::MatrixOperator<T>>(a_i, bc_i);
      maps[i] = operators[i]->column_index_map();

      la::Vector<T> b(maps[i], 1);
      b.set(T(0.0));
      fem::assemble_vector(b.mutable_array(), *L[i]);
      fem::apply_lifting<T, T>(b.mutable_array(), {a[i]}, {{bcs[i]}}, {}, T(1));
      b.scatter_rev(std::plus<T>());
      fem::set_bc<T, T>(b.mutable_array(), {bcs[i]});

      bs[i] = std::make_shared<DeviceVector>(maps[i], 1);
      bs[i]->copy_from_host(b);
    }

    for (std::size_t i = 0; i < V.size(); i++)
    {
      auto size = operators[i]->nnz();
      if (rank == 0)
      {
        std::cout << "Number of nonzeros level " << i << ": ";
        std::cout << size << std::endl;
      }
    }

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

      (*operators[i])(*bs[i], x);

      [[maybe_unused]] int its = cg.solve(*operators[i], x, *bs[i], false);
      int na = cg.alphas().size();
      for (int i = 0; i < na; ++i)
        LOG(INFO) << i << "] a = " << cg.alphas()[i] << ", b = " << cg.betas()[i];
      
      std::vector<T> eign = cg.compute_eigenvalues();
      std::sort(eign.begin(), eign.end());
      std::array<T, 2> eig_range = {0.3 * eign.back(), 1.2 * eign.back()};
      smoothers[i] = std::make_shared<acc::Chebyshev<DeviceVector>>(maps[i], 1, eig_range, 2);

      if (rank == 0)
      {
        std::cout << "Eigenvalues level " << i << ": ";
        std::cout << eig_range[0] << " " << eig_range[1] << std::endl;
      }
    }

    smoothers[0]->set_max_iterations(20);
    smoothers[1]->set_max_iterations(5);
    smoothers[2]->set_max_iterations(5);

    // Create Restriction operator
    std::vector<std::shared_ptr<acc::MatrixOperator<T>>> restriction(V.size() - 1);

    // Create Prolongation operator
    std::vector<std::shared_ptr<acc::MatrixOperator<T>>> prolongation(V.size() - 1);

    // From V1 to V0
    LOG(WARNING) << "Creating Prolongation Operators";
    prolongation[0] = std::make_shared<acc::MatrixOperator<T>>(*V[0], *V[1]);
    // From V2 to V1
    prolongation[1] = std::make_shared<acc::MatrixOperator<T>>(*V[1], *V[2]);

    using OpType = acc::MatrixOperator<T>;
    using SolverType = acc::Chebyshev<DeviceVector>;

    using PMG = acc::MultigridPreconditioner<DeviceVector, OpType, OpType, OpType, SolverType>;

    PMG pmg(maps, 1);
    pmg.set_solvers(smoothers);
    pmg.set_operators(operators);
    pmg.set_interpolators(prolongation);

    // Create solution vector
    DeviceVector x(maps.back(), 1);
    x.set(T{0.0});

    for (int i = 0; i < 10; i++)
      pmg.apply(*bs.back(), x);

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

  MPI_Finalize();
  return 0;
}
