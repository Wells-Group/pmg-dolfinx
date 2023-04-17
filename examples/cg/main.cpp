#include "poisson.h"
#include "src/cg.hpp"
#include "src/operators.hpp"
#include "src/vector.hpp"

#include <basix/e-lagrange.h>
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <petscdevice.h>

using namespace dolfinx;
using T = double;

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Create a hexahedral mesh
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        mesh::create_box<T>(comm, {{{0, 0, 0}, {1, 1, 1}}}, {15, 15, 15},
                            mesh::CellType::hexahedron));

    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
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

    fem::Function<T> u(V);
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());

    b.set(T(0.0));
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, T>(b.mutable_array(), {bc});

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
    using HostVector = dolfinx::acc::Vector<T, acc::Device::CPP>;

    DeviceVector x(V->dofmap()->index_map, 1);
    x.set(T{0.0});

    DeviceVector y(V->dofmap()->index_map, 1);
    y.copy_from_host(b); // Copy data from host vector to device vector

    // Create petsc operator
    PETScOperator op(a, {bc});
    // x = Ay
    op(y, x);

    // Create distributed CG solver
    dolfinx::acc::CGSolver<DeviceVector> cg(V->dofmap()->index_map, 1);
    cg.set_max_iterations(100);
    cg.set_tolerance(1e-5);
    cg.store_coefficients(true);

    // Solve
    int its = cg.solve(op, x, y, true);

    if (rank == 0)
      std::cout << "Number of iterations" << its << std::endl;
  }

  PetscFinalize();
  return 0;
}
