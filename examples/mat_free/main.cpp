#include "poisson.h"
#include "src/cg.hpp"
#include "src/chebyshev.hpp"
#include "src/csr.hpp"
#include "src/laplacian.hpp"
#include "src/mesh.hpp"
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
#include <thrust/sequence.h>

#include "src/amd_gpu.hpp"

using namespace dolfinx;
using T = double;
namespace po = boost::program_options;

template <std::floating_point T>
std::vector<T> create_geom(MPI_Comm comm, std::array<std::array<double, 3>, 2> p,
                           std::array<std::int64_t, 3> n)
{
  // Extract data
  const std::array<double, 3> p0 = p[0];
  const std::array<double, 3> p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array range_p
      = dolfinx::MPI::local_range(dolfinx::MPI::rank(comm), n_points, dolfinx::MPI::size(comm));

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);
  const double z0 = std::min(p0[2], p1[2]);
  const double z1 = std::max(p0[2], p1[2]);

  const T a = x0;
  const T b = x1;
  const T ab = (b - a) / static_cast<T>(nx);
  const T c = y0;
  const T d = y1;
  const T cd = (d - c) / static_cast<T>(ny);
  const T e = z0;
  const T f = z1;
  const T ef = (f - e) / static_cast<T>(nz);

  if (std::abs(x0 - x1) < 2.0 * std::numeric_limits<double>::epsilon()
      or std::abs(y0 - y1) < 2.0 * std::numeric_limits<double>::epsilon()
      or std::abs(z0 - z1) < 2.0 * std::numeric_limits<double>::epsilon())
  {
    throw std::runtime_error("Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 or ny < 1 or nz < 1)
  {
    throw std::runtime_error("BoxMesh has non-positive number of vertices in some dimension");
  }

  std::vector<T> geom;
  geom.reserve((range_p[1] - range_p[0]) * 3);
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v)
  {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const T z = e + ef * static_cast<T>(iz);
    const T y = c + cd * static_cast<T>(iy);
    const T x = a + ab * static_cast<T>(ix);
    geom.insert(geom.end(), {x, y, z});
  }

  return geom;
}

template <std::floating_point T>
dolfinx::mesh::Mesh<T>
build_hex(MPI_Comm comm, MPI_Comm subcomm, std::array<std::array<double, 3>, 2> p,
          std::array<std::int64_t, 3> n, const dolfinx::fem::CoordinateElement<T>& element)
{
  common::Timer timer("Build BoxMesh (hexahedra)");
  std::vector<T> x;
  std::vector<std::int64_t> cells;
  if (subcomm != MPI_COMM_NULL)
  {
    x = create_geom<T>(subcomm, p, n);

    // Create cuboids
    const std::int64_t nx = n[0];
    const std::int64_t ny = n[1];
    const std::int64_t nz = n[2];
    const std::int64_t n_cells = nx * ny * nz;
    std::array range_c = dolfinx::MPI::local_range(dolfinx::MPI::rank(subcomm), n_cells,
                                                   dolfinx::MPI::size(subcomm));
    cells.reserve((range_c[1] - range_c[0]) * 8);
    for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
    {
      const std::int64_t iz = i / (nx * ny);
      const std::int64_t j = i % (nx * ny);
      const std::int64_t iy = j / nx;
      const std::int64_t ix = j % nx;

      const std::int64_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
      const std::int64_t v1 = v0 + 1;
      const std::int64_t v2 = v0 + (nx + 1);
      const std::int64_t v3 = v1 + (nx + 1);
      const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
      const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
      const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
      const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);
      cells.insert(cells.end(), {v0, v1, v2, v3, v4, v5, v6, v7});
    }
  }

  auto partitioner = dolfinx::mesh::create_cell_partitioner();

  return create_mesh(comm, subcomm, cells, element, subcomm, x, {x.size() / 3, 3}, partitioner);
}

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

    spdlog::info("Mesh shape: {}x{}x{}", nx[0], nx[1], nx[2]);

    auto element = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false));

    auto element_1 = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, 1,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false));

    dolfinx::fem::CoordinateElement<T> coord_element(element_1);

    // Create mesh
    std::shared_ptr<mesh::Mesh<T>> mesh;
    {
      mesh::Mesh<T> base_mesh = mesh::create_box<T>(
          comm, {{{0, 0, 0}, {1, 1, 1}}},
          {(std::size_t)nx[0], (std::size_t)nx[1], (std::size_t)nx[2]}, mesh::CellType::hexahedron);

      //      mesh::Mesh<T> base_mesh = build_hex<T>(comm, comm, {{{0, 0, 0}, {1, 1, 1}}},
      //                                             {nx[0], nx[1], nx[2]}, coord_element);

      //      mesh = std::make_shared<mesh::Mesh<T>>(base_mesh);
      mesh = std::make_shared<mesh::Mesh<T>>(ghost_layer_mesh(base_mesh, coord_element));
    }

    std::stringstream sg;

    for (auto q : mesh->geometry().x())
      sg << q << " ";
    spdlog::debug("x = {}", sg.str());

    auto [lcells, bcells] = compute_boundary_cells(mesh);

    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::gll, basix::cell::type::hexahedron, basix::polyset::type::standard,
        4);

    // Compute the geometrical factor and copy to device
    auto G_ = compute_scaled_geometrical_factor<T>(mesh, Gpoints, Gweights);

    std::stringstream ssG;
    for (auto q : G_)
      ssG << q << " ";
    spdlog::debug("G = [{}]", ssG.str());

    thrust::device_vector<T> G_d(G_.begin(), G_.end());
    std::span<const T> geometry_d_span(thrust::raw_pointer_cast(G_d.data()), G_d.size());

    spdlog::info("Send G to GPU (size = {})", G_d.size());

    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, *element));

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
    const int num_cells_local = mesh->topology()->index_map(tdim)->size_local()
                                + mesh->topology()->index_map(tdim)->num_ghosts();
    thrust::device_vector<T> constants_d(num_cells_local, kappa->value[0]);
    std::span<const T> constants_d_span(thrust::raw_pointer_cast(constants_d.data()),
                                        constants_d.size());

    spdlog::info("Send constants to GPU (size = {})", constants_d.size());

    // V dofmap
    thrust::device_vector<std::int32_t> dofmap_d(
        dofmap->map().data_handle(), dofmap->map().data_handle() + dofmap->map().size());
    std::span<const std::int32_t> dofmap_d_span(thrust::raw_pointer_cast(dofmap_d.data()),
                                                dofmap_d.size());

    spdlog::info("Send dofmap to GPU (size = {})", dofmap_d.size());
    std::stringstream sd;
    for (int i = 0; i < dofmap->map().size(); ++i)
      sd << dofmap->map().data_handle()[i] << " ";
    spdlog::debug("domfpa = {}", sd.str());

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;

    // Input vector
    auto map = V->dofmap()->index_map;
    DeviceVector u(map, 1);
    u.set(T{0.0});

    // Output vector
    DeviceVector y(map, 1);
    y.set(T{0.0});

    thrust::device_vector<int> cell_list_d(num_cells_local);
    thrust::sequence(cell_list_d.begin(), cell_list_d.end());
    std::span<const int> cells_local(thrust::raw_pointer_cast(cell_list_d.data()),
                                     cell_list_d.size());

    spdlog::info("Send cell list to GPU (size = {})", cell_list_d.size());

    // Create matrix free operator
    spdlog::debug("Create MatFreLaplacian");
    acc::MatFreeLaplacian<T> op(3, cells_local, constants_d_span, dofmap_d_span, geometry_d_span);

    la::Vector<T> b(map, 1);
    auto barr = b.mutable_array();

    std::copy(f->x()->array().begin(), f->x()->array().end(), barr.begin());

    // fem::assemble_vector(b.mutable_array(), *L);
    // TODO BCs
    // fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    // b.scatter_rev(std::plus<T>());
    // fem::set_bc<T, T>(b.mutable_array(), {bc});
    u.copy_from_host(b); // Copy data from host vector to device vector
    u.scatter_fwd();

    // Matrix free
    op(u, y);
    //    y.scatter_rev();

    std::cout << "Norm of u = " << acc::norm(u) << "\n";
    std::cout << "Norm of y = " << acc::norm(y) << "\n";

    std::vector<T> yhost = y.data_copy();

    // Compare to assembling on CPU and copying matrix to GPU
    DeviceVector z(map, 1);
    z.set(T{0.0});

    acc::MatrixOperator<T> mat_op(a, {});
    mat_op(u, z);
    std::cout << "Norm of u = " << acc::norm(u) << "\n";
    std::cout << "Norm of z = " << acc::norm(z) << "\n";

    std::vector<T> zhost = z.data_copy();

    // for (std::size_t i = 0; i < zhost.size(); ++i)
    // {
    //   if (std::abs(zhost[i] - yhost[i]) > 1e-8)
    //     std::cout << i << ": " << zhost[i] << ", " << yhost[i] << "\n";
    // }

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  }

  MPI_Finalize();

  return 0;
}
