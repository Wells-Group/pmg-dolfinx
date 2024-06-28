#include "poisson.h"
#include "src/operators.hpp"
#include "src/vector.hpp"

#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
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
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(500), "number of dofs per rank");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};

    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int order = 1;
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
    auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
        comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]}, mesh::CellType::hexahedron));

    auto element = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, 1,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false);
    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element, {}));

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

    spdlog::info("Interpolate");

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

    spdlog::info("Topology");

    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);

    spdlog::info("DirichletBC");

    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    fem::Function<T> u(V);
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());

    spdlog::info("Assemble Vector");
    b.set(T(0.0));
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, T>(b.mutable_array(), {bc});

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
    using HostVector = dolfinx::acc::Vector<T, acc::Device::CPP>;

    spdlog::info("Create Petsc Operator");

    // Create petsc operator
    PETScOperator op(a, {bc});

    auto im_op = op.index_map();
    spdlog::info("OP:{} {} {}", im_op->size_global(), im_op->size_local(), im_op->num_ghosts());
    auto im_V = V->dofmap()->index_map;
    spdlog::info("V:{} {} {}", im_V->size_global(), im_V->size_local(), im_V->num_ghosts());

    DeviceVector x(V->dofmap()->index_map, 1);
    x.set(T{0.0});

    DeviceVector y(op.index_map(), 1);
    y.copy_from_host(b); // Copy data from host vector to device vector

    spdlog::info("Apply operator");

    // x = A.y
    op(y, x);

    spdlog::info("get device matrix");
    Mat A = op.device_matrix();

    spdlog::info("Create Petsc KSP");
    // Create PETSc KSP object
    KSP solver;
    PC prec;
    KSPCreate(comm, &solver);
    spdlog::info("Set KSP Type");
    KSPSetType(solver, KSPCG);
    spdlog::info("Set Operators");
    KSPSetOperators(solver, A, A);
    spdlog::info("Set PC Type");
    KSPGetPC(solver, &prec);
    PCSetType(prec, PCHYPRE);
    //    spdlog::info( "Set AMG Type";
    //    PCGAMGSetType(prec, PCGAMGAGG);
    KSPSetFromOptions(solver);
    spdlog::info("KSP Setup");
    KSPSetUp(solver);

    spdlog::info("Create Petsc HIP arrays");
    // SET OPTIONS????
    const PetscInt local_size = V->dofmap()->index_map->size_local();
    const PetscInt global_size = V->dofmap()->index_map->size_global();
    Vec _b, _x;
    VecCreateMPIHIPWithArray(comm, PetscInt(1), local_size, global_size, NULL, &_x);
    VecCreateMPIHIPWithArray(comm, PetscInt(1), local_size, global_size, NULL, &_b);

    VecHIPPlaceArray(_b, y.array().data());
    VecHIPPlaceArray(_x, x.array().data());

    dolfinx::common::Timer tsolve("ZZZ Solve");
    KSPSolve(solver, _b, _x);
    tsolve.stop();
    KSPView(solver, PETSC_VIEWER_STDOUT_WORLD);

    KSPConvergedReason reason;
    KSPGetConvergedReason(solver, &reason);

    PetscInt num_iterations = 0;
    int ierr = KSPGetIterationNumber(solver, &num_iterations);
    if (ierr != 0)
      spdlog::error("KSPGetIterationNumber Error: {}", ierr);

    if (rank == 0)
    {
      std::cout << "Converged reason: " << reason << "\n";
      std::cout << "Num iterations: " << num_iterations << "\n";
    }

    VecHIPResetArray(_b);
    VecHIPResetArray(_x);

    spdlog::info("x.norm = {}", acc::norm(x));
    spdlog::info("[0] y.norm = {}", acc::norm(y));

    // y = A.x
    op(x, y);

    spdlog::info("[1] y.norm = {}", acc::norm(y));

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

  PetscFinalize();
  return 0;
}
