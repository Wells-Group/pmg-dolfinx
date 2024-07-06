#include "poisson.h"
#include "src/cg.hpp"
#include "src/ghost_layer.hpp"
#include "src/operators.hpp"
#include "src/vector.hpp"
#include "src/mesh.hpp"

#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
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
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  const std::size_t ndofs = 50000;

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};

    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int order = 2;
    double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / order;
    std::int64_t n0 = static_cast<int>(nx_approx);
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

    // Create a hexahedral mesh

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

    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, *element, {}));


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
    }

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
