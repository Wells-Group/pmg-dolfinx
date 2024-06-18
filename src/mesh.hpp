
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <span>

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
  std::size_t num_cell_vertices = dolfinx::mesh::num_cell_vertices(mesh.topology()->cell_type());

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

  // Convert topology to global indexing, and restrict to non-ghost cells
  std::vector<std::int32_t> topo = mesh.topology()->connectivity(tdim, 0)->array();
  topo.resize(ncells * num_cell_vertices);
  spdlog::info("topo.size = {}", topo.size());

  std::vector<std::int64_t> topo_global(topo.size());
  mesh.topology()->index_map(0)->local_to_global(topo, topo_global);

  auto new_mesh = dolfinx::mesh::create_mesh(mesh.comm(), mesh.comm(), std::span(topo_global),
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
compute_boundary_cells(std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh)
{
  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  topology->create_connectivity(fdim, tdim);

  int ncells_local = topology->index_map(tdim)->size_local();
  auto f_to_c = topology->connectivity(fdim, tdim);
  // Create list of cells needed for matrix-free updates
  std::vector<std::int32_t> boundary_cells;
  for (std::int32_t f = 0; f < f_to_c->num_nodes(); ++f)
  {
    const auto& cells_f = f_to_c->links(f);
    for (std::int32_t c : cells_f)
    {
      // If facet attached to a ghost cell, add all cells to list
      // FIXME: should this really be via vertex, not facet?
      if (c >= ncells_local)
        boundary_cells.insert(boundary_cells.end(), cells_f.begin(), cells_f.end());
    }
  }
  std::sort(boundary_cells.begin(), boundary_cells.end());
  boundary_cells.erase(std::unique(boundary_cells.begin(), boundary_cells.end()),
                       boundary_cells.end());
  spdlog::info("Got {} boundary cells.", boundary_cells.size());

  // Compute local cells
  std::vector<std::int32_t> local_cells(topology->index_map(tdim)->size_local()
                                        + topology->index_map(tdim)->num_ghosts());
  std::iota(local_cells.begin(), local_cells.end(), 0);
  for (std::int32_t c : boundary_cells)
    local_cells[c] = -1;
  std::erase(local_cells, -1);
  spdlog::info("Got {} local cells", local_cells.size());

  return {std::move(local_cells), std::move(boundary_cells)};
}
