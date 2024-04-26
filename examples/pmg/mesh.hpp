
#include <dolfinx/mesh/Mesh.h>

template <std::floating_point T>
dolfinx::mesh::Mesh<T> ghost_layer_mesh(dolfinx::mesh::Mesh<T>& mesh);
