
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <span>

template <std::floating_point T>
dolfinx::mesh::Mesh<T> ghost_layer_mesh(dolfinx::mesh::Mesh<T>& mesh)
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

  auto partitioner
      = [cell_to_dests, ncells](MPI_Comm comm, int nparts, dolfinx::mesh::CellType cell_type,
                                const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
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
  std::vector<std::int64_t> topo_global(topo.size());
  mesh.topology()->index_map(0)->local_to_global(topo, topo_global);

  auto new_mesh
      = dolfinx::mesh::create_mesh(mesh.comm(), mesh.comm(), std::span(topo_global),
                                   mesh.geometry().cmap(), mesh.comm(), x, xshape, partitioner);

  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());

  return new_mesh;
}
