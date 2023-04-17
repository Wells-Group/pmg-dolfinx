from mpi4py import MPI
import dolfinx
from dolfinx.mesh import create_unit_cube
from dolfinx.cpp.fem.petsc import interpolation_matrix
from scipy.sparse import csr_matrix


comm = MPI.COMM_WORLD
mesh = create_unit_cube(comm, 10, 10, 10, cell_type=dolfinx.mesh.CellType.hexahedron)

V1 = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 4))
V2 = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 5))


I = interpolation_matrix(V1._cpp_object, V2._cpp_object)
I.assemble()

sizes = I.getSizes()
m = sizes[0][0]
n = sizes[1][1]

row_ptr, col_indices, values =  I.getValuesCSR()
I = csr_matrix((values, col_indices, row_ptr))
I.eliminate_zeros()
I.prune()
nnz_per_row = I.nnz/I.shape[0]

print(nnz_per_row)