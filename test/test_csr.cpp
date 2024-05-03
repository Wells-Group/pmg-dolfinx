#include "../../src/csr.hpp"
#include "../../src/matrix-free.hpp"
#include "../../src/operators.hpp"
#include "../../src/vector.hpp"
#include "../../src/mesh.hpp"
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

int main(int argc, char* argv[])
{

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ncells", po::value<std::size_t>()->default_value(100), "number of cells in each direction per rank")(
      "degree", po::value<int>()->default_value(1), "Finite element degree");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }
  const std::size_t nc = vm["ncells"].as<std::size_t>();
  int degree = vm["degree"].as<int>();

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Create mesh
    std::shared_ptr<mesh::Mesh<T>> mesh;
    {
      mesh::Mesh<T> base_mesh = mesh::create_box<T>(
          comm, {{{0, 0, 0}, {1, 1, 1}}}, {nc, nc, nc}, mesh::CellType::hexahedron);
      mesh = std::make_shared<mesh::Mesh<T>>(ghost_layer_mesh(base_mesh));
    }

    int tdim = mesh->topology()->dim();
    std::int64_t ncells = mesh->topology()->index_map(tdim)->size_global();

    if (rank == 0)
        std::cout << "Number of cells: " << ncells << std::endl;

    // Create function space
    auto e0 = basix::create_tp_element<T>(
          basix::element::family::P, basix::cell::type::hexahedron, degree,
          basix::element::lagrange_variant::gll_warped, 
          basix::element::dpc_variant::unset, false);
    auto e1 = basix::create_tp_element<T>(
          basix::element::family::P, basix::cell::type::hexahedron, degree,
          basix::element::lagrange_variant::gll_warped, 
          basix::element::dpc_variant::unset, false);

    auto V0 = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, e0, {}));
    auto V1 = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, e1, {}));

    acc::MatrixOperator<T> M(*V0, *V1);

    auto map0 = V0->dofmap()->index_map;
    auto map1 = V1->dofmap()->index_map;


    DeviceVector x(map0, 1);
    x.set(T(1.0));
    DeviceVector y(map1, 1);

    
    M(x, y);

    // check
    auto normx = acc::norm(x);
    auto normy = acc::norm(y);
    std::cout << " Norm x:  " << normx << std::endl;
    std::cout << " Norm y:  " << normy << std::endl;


    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }
  return 0;
}