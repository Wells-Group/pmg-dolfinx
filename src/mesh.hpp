
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <span>

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

/// Create hex mesh with a coordinate element
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

/// @brief Create a new mesh with an extra boundary layer, such that all cells on other processes
/// which share a vertex with this process are ghosted.
/// @param mesh Input mesh
/// @param coord_element A coordinate element for the new mesh. This may be tensor product ordering.

template <std::floating_point T>
dolfinx::mesh::Mesh<T> ghost_layer_mesh(dolfinx::mesh::Mesh<T>& mesh,
                                        dolfinx::fem::CoordinateElement<T> coord_element)
{
  constexpr int tdim = 3;
  constexpr int gdim = 3;
  std::size_t ncells = mesh.topology()->index_map(tdim)->size_local();
  std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations = mesh.topology()->index_map(0)->index_to_dest_ranks();

  // Map from any local cells to processes where they should be ghosted
  std::map<int, std::vector<int>> cell_to_dests;
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  std::vector<int> cdests;
  for (std::size_t c = 0; c < ncells; ++c)
  {
    cdests.clear();
    for (auto v : c_to_v->links(c))
    {
      auto vdest = vertex_destinations.links(v);
      for (int dest : vdest)
        cdests.push_back(dest);
    }
    std::sort(cdests.begin(), cdests.end());
    cdests.erase(std::unique(cdests.begin(), cdests.end()), cdests.end());
    if (!cdests.empty())
      cell_to_dests[c] = cdests;
  }

  spdlog::info("cell_to_dests= {}, ncells = {}", cell_to_dests.size(), ncells);

  auto partitioner
      = [cell_to_dests, ncells](MPI_Comm comm, int nparts,
                                const std::vector<dolfinx::mesh::CellType>& cell_types,
                                const std::vector<std::span<const std::int64_t>>& cells)
  {
    int rank = dolfinx::MPI::rank(comm);
    std::vector<std::int32_t> dests;
    std::vector<int> offsets = {0};
    for (int c = 0; c < ncells; ++c)
    {
      dests.push_back(rank);
      if (auto it = cell_to_dests.find(c); it != cell_to_dests.end())
        dests.insert(dests.end(), it->second.begin(), it->second.end());

      // Ghost to other processes
      offsets.push_back(dests.size());
    }
    return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests), std::move(offsets));
  };

  std::array<std::size_t, 2> xshape = {num_vertices, gdim};
  std::span<const T> x(mesh.geometry().x().data(), xshape[0] * xshape[1]);

  auto dofmap = mesh.geometry().dofmap();
  auto imap = mesh.geometry().index_map();
  std::vector<std::int32_t> permuted_dofmap;
  std::vector<int> perm = basix::tp_dof_ordering(
      basix::element::family::P, mesh::cell_type_to_basix_type(coord_element.cell_shape()),
      coord_element.degree(), coord_element.variant(), basix::element::dpc_variant::unset, false);
  for (std::size_t c = 0; c < dofmap.extent(0); ++c)
  {
    auto cell_dofs = std::submdspan(dofmap, c, std::full_extent);
    for (int i = 0; i < dofmap.extent(1); ++i)
      permuted_dofmap.push_back(cell_dofs(perm[i]));
  }
  std::vector<std::int64_t> permuted_dofmap_global(permuted_dofmap.size());
  imap->local_to_global(permuted_dofmap, permuted_dofmap_global);

  auto new_mesh
      = dolfinx::mesh::create_mesh(mesh.comm(), mesh.comm(), std::span(permuted_dofmap_global),
                                   coord_element, mesh.comm(), x, xshape, partitioner);

  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());
  spdlog::info("** NEW MESH num_local_cells = {}",
               new_mesh.topology()->index_map(tdim)->size_local());

  return new_mesh;
}

/// @brief Compute two lists of cell indices:
/// 1. cells which are "local", i.e. the dofs on
/// these cells are not shared with any other process.
/// 2. cells which share dofs with other processes.
///
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
compute_boundary_cells(std::shared_ptr<dolfinx::fem::FunctionSpace<T>> V)
{
  auto mesh = V->mesh();
  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  topology->create_connectivity(fdim, tdim);

  int ncells_local = topology->index_map(tdim)->size_local();
  int ncells_ghost = topology->index_map(tdim)->num_ghosts();
  int ndofs_local = V->dofmap()->index_map->size_local();

  std::vector<std::uint8_t> cell_mark(ncells_local + ncells_ghost, 0);
  for (int i = 0; i < ncells_local; ++i)
  {
    auto cell_dofs = V->dofmap()->cell_dofs(i);
    for (auto dof : cell_dofs)
      if (dof >= ndofs_local)
        cell_mark[i] = 1;
  }
  for (int i = ncells_local; i < ncells_local + ncells_ghost; ++i)
    cell_mark[i] = 1;

  std::vector<int> local_cells;
  std::vector<int> boundary_cells;
  for (int i = 0; i < cell_mark.size(); ++i)
  {
    if (cell_mark[i])
      boundary_cells.push_back(i);
    else
      local_cells.push_back(i);
  }

  spdlog::debug("lcells:{}, bcells:{}", local_cells.size(), boundary_cells.size());

  return {std::move(local_cells), std::move(boundary_cells)};
}
