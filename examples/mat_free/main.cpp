#include "poisson.h"
#include "src/cg.hpp"
#include "src/chebyshev.hpp"
#include "src/csr.hpp"
#include "src/laplacian.hpp"
#include "src/operators.hpp"
#include "src/precompute.hpp"
#include "src/vector.hpp"

#include <array>
#include <basix/finite-element.h>
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
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(343), "number of dofs per rank");

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
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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
    auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
        comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]}, mesh::CellType::hexahedron));

    auto element = basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false);

    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::gll, basix::cell::type::hexahedron, basix::polyset::type::standard,
        4);

    // Compute the geometrical factor and copy to device
    auto G_ = compute_scaled_geometrical_factor<T>(mesh, Gpoints, Gweights);
    thrust::device_vector<T> G_d(G_.begin(), G_.end());
    std::span<const T> geometry_d_span(thrust::raw_pointer_cast(G_d.data()), G_d.size());

    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element));

    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    std::size_t ncells = mesh->topology()->index_map(tdim)->size_global();
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

    int fdim = tdim - 1;
    topology->create_connectivity(fdim, tdim);

    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    // Copy data to GPU
    // Constants
    // TODO Pack these properly
    const int num_cells_local = mesh->topology()->index_map(tdim)->size_local();
    thrust::device_vector<T> constants_d(num_cells_local, kappa->value[0]);
    std::span<const T> constants_d_span(thrust::raw_pointer_cast(constants_d.data()),
                                        constants_d.size());

    // V dofmap
    thrust::device_vector<std::int32_t> dofmap_d(
        dofmap->map().data_handle(), dofmap->map().data_handle() + dofmap->map().size());
    std::span<const std::int32_t> dofmap_d_span(thrust::raw_pointer_cast(dofmap_d.data()),
                                                dofmap_d.size());

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;

    // Input vector
    auto map = V->dofmap()->index_map;
    DeviceVector u(map, 1);
    u.set(T{0.0});

    // Output vector
    DeviceVector y(map, 1);
    y.set(T{0.0});

    // Create matrix free operator
    spdlog::debug("Create MatFreLaplacian");
    acc::MatFreeLaplacian<3, T> op(num_cells_local, constants_d_span, dofmap_d_span,
                                   geometry_d_span);

    la::Vector<T> b(map, 1);
    b.set(T(0.0));

    fem::assemble_vector(b.mutable_array(), *L);
    // TODO BCs
    // fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    // b.scatter_rev(std::plus<T>());
    // fem::set_bc<T, T>(b.mutable_array(), {bc});
    u.copy_from_host(b); // Copy data from host vector to device vector

    op(u, y);
    std::cout << "Norm of u = " << acc::norm(u) << "\n";
    std::cout << "Norm of y = " << acc::norm(y) << "\n";

    // Compare to assembling on CPU and copying matrix to GPU
    DeviceVector z(map, 1);
    z.set(T{0.0});

    acc::MatrixOperator<T> mat_op(a, {});
    mat_op(u, z);
    std::cout << "Norm of u = " << acc::norm(u) << "\n";
    std::cout << "Norm of z = " << acc::norm(z) << "\n";

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

  MPI_Finalize();

  return 0;
}
