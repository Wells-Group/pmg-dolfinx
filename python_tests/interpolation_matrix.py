from mpi4py import MPI
import basix
import dolfinx
from dolfinx.mesh import create_unit_cube
from dolfinx.cpp.fem.petsc import interpolation_matrix
from scipy.sparse import csr_matrix
import numpy as np


comm = MPI.COMM_WORLD
mesh = create_unit_cube(comm, 5, 5, 5, cell_type=dolfinx.mesh.CellType.hexahedron)
tdim = mesh.topology.dim

order1 = 1
order2 = 3

V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", order1))
V2 = dolfinx.fem.functionspace(mesh, ("Lagrange", order2))

I = interpolation_matrix(V1._cpp_object, V2._cpp_object)
I.assemble()

sizes = I.getSizes()
m = sizes[0][0]
n = sizes[1][1]

row_ptr, col_indices, values =  I.getValuesCSR()
I = csr_matrix((values, col_indices, row_ptr))

b = np.arange(V1.dofmap.index_map.size_local)

print('Original in V1: ', b)
w = I @ b
print('Interpolating into V2: ', w)

v = I.T @ w
print('Interpolating back to V1 via transpose: ', v)

# Matrix free version...

Q1 = basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron,
                          order1, basix.LagrangeVariant.gll_warped)
Q2 = basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron,
                          order2, basix.LagrangeVariant.gll_warped)

np.set_printoptions(suppress=True)
mat = basix.compute_interpolation_operator(Q1, Q2)

# Matrix-free interpolate from V1->V2
w = np.empty(V2.dofmap.index_map.size_local)
for i in range(mesh.topology.index_map(tdim).size_local):
    in_cell_dofs = V1.dofmap.list[i]
    out_cell_dofs = V2.dofmap.list[i]
    in_vals = b[in_cell_dofs]
    out_vals = mat @ in_vals
    w[out_cell_dofs] = out_vals

# Matrix-free interpolate from V2->V1
uq, count = np.unique(np.sort(np.array(V2.dofmap.list).flatten()), return_counts=True)
# Get 'multiplicity' of each DoF
assert np.allclose(uq, np.arange(len(uq)))
mu = 1/count

v_mf = np.zeros(V1.dofmap.index_map.size_local)
for i in range(mesh.topology.index_map(tdim).size_local):
    in_cell_dofs = V2.dofmap.list[i]
    out_cell_dofs = V1.dofmap.list[i]
    in_vals = w[in_cell_dofs] * mu[in_cell_dofs]
    out_vals = mat.T @ in_vals
    v_mf[out_cell_dofs] += out_vals

assert np.allclose(v, v_mf)
