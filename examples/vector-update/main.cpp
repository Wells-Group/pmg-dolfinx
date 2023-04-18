#include "poisson.h"
#include "src/cg.hpp"
#include "src/operators.hpp"
#include "src/vector.hpp"
#include "src/ghost_layer.hpp"

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
    auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
        comm, {{{0, 0, 0}, {1, 1, 1}}}, {10, 10, 10}, mesh::CellType::hexahedron));

    mesh = create_ghost_layer(mesh);

    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
    using HostVector = dolfinx::acc::Vector<T, acc::Device::CPP>;
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;

    DeviceVector x(V->dofmap()->index_map, 1);
    x.set(T(rank));

    DeviceVector y(V->dofmap()->index_map, 1);
    y.set(T{1});

    bool verbose = true;
    for (int i = 0; i < 100; i++)
    {
      x.scatter_fwd_begin();
      auto value = acc::norm(x, dolfinx::la::Norm::l2);
      acc::axpy(x, 1.0, x, y);
      x.scatter_fwd_end();

      if (rank == 0 and verbose)
        std::cout << "Dot value: " << value << std::endl;
    }
  }

  PetscFinalize();
  return 0;
}
