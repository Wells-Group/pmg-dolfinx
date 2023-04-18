
#pragma once

#include <memory>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>

template <typename T>
std::shared_ptr<mesh::Mesh<T>> create_ghost_layer(std::shared_ptr<mesh::Mesh<T>> mesh)
{
  MPI_Comm comm = mesh->comm();
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  // Compute sharing of owned vertices
  std::shared_ptr<const common::IndexMap> v_imap = mesh->topology()->index_map(0);
  std::shared_ptr<const common::IndexMap> c_imap = mesh->topology()->index_map(3);
  auto shared_v = v_imap->index_to_dest_ranks();

  auto cv = mesh->topology()->connectivity(3, 0);
  std::vector<int> offset = {0};
  std::set<int> p;
  std::vector<std::int32_t> dests;
  for (int c = 0; c < c_imap->size_local(); ++c)
  {
    p.clear();

    for (auto v : cv->links(c))
      p.insert(shared_v.links(v).begin(), shared_v.links(v).end());

    if (p.find(rank) != p.end())
      throw std::runtime_error("Found self in sharing processes");
    dests.push_back(rank);
    dests.insert(dests.end(), p.begin(), p.end());
    offset.push_back(dests.size());
  }

  graph::AdjacencyList<std::int32_t> new_partition(dests, offset);

  auto partitioner = [&new_partition](MPI_Comm, int, int, const graph::AdjacencyList<std::int64_t>&)
  { return new_partition; };

  fem::CoordinateElement element(mesh::CellType::hexahedron, 1);

  std::vector<std::int64_t> global_v(cv->array().size());
  v_imap->local_to_global(cv->array(), global_v);
  graph::AdjacencyList<std::int64_t> cv_global(global_v, cv->offsets());
  std::size_t num_pts = mesh->geometry().index_map()->size_local();
  std::span<T> geom(mesh->geometry().x().data(), num_pts * 3);
  auto new_mesh = std::make_shared<mesh::Mesh<T>>(
      mesh::create_mesh(comm, cv_global, {element}, geom, {num_pts, 3}, partitioner));

  std::cout << "rank = " << rank << ":" << new_mesh->topology()->index_map(3)->num_ghosts()
            << std::endl;

  return new_mesh;
}
