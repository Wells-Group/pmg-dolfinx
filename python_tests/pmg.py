from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc
import ufl
from ufl import TestFunction, TrialFunction, dx, inner, grad, div
import numpy as np
from petsc4py import PETSc


def boundary_condition(V):
    msh = V.mesh
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    return fem.dirichletbc(0.0, dofs, V)


def create_a(V, kappa):
    u, v = TrialFunction(V), TestFunction(V)
    a = kappa * inner(grad(u), grad(v)) * dx
    return fem.form(a)


def create_L(V, kappa, u_e):
    v = TestFunction(V)
    f = -kappa * div(grad(u_e))
    L = inner(f, v) * dx
    return fem.form(L)


def residual(b, A, u):
    r = A.createVecRight()
    A.mult(u.vector, r)
    r.axpby(1, -1, b.vector)
    return r


def norm_L2(comm, v, measure=ufl.dx):
    return np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM
        )
    )


n = 10
ks = [1, 3]
num_iters = 10
kappa = 1.0
comm = MPI.COMM_WORLD
msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=mesh.CellType.hexahedron)

# Exact solution
x = ufl.SpatialCoordinate(msh)
u_e = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])

Vs = [fem.functionspace(msh, ("Lagrange", k)) for k in ks]

As = []
bcs = []
b = fem.Function(Vs[-1])
for V in Vs:
    a = create_a(V, kappa)
    bc = boundary_condition(V)
    bcs.append(bc)
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    As.append(A)

    # Assemble RHS
    if V == Vs[-1]:
        L = create_L(Vs[-1], kappa, u_e)
        petsc.assemble_vector(b.vector, L)
        petsc.apply_lifting(b.vector, [a], bcs=[[bc]])
        petsc.set_bc(b.vector, bcs=[bc])

interp_ops = [petsc.interpolation_matrix(Vs[i], Vs[i + 1]) for i in range(len(Vs) - 1)]
for interp_op in interp_ops:
    interp_op.assemble()

# Create solvers
solvers = []

# Coarse
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver_prefix = "solver_0_"
solver.setOptionsPrefix(solver_prefix)
solver.setOperators(As[0])
solver.setType(PETSc.KSP.Type.PREONLY)
solver.pc.setType(PETSc.PC.Type.LU)
solver.setFromOptions()
solvers.append(solver)

# Fine
for i in range(1, len(ks)):
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver_prefix = f"solver_{i}_"
    solver.setOptionsPrefix(solver_prefix)
    opts = PETSc.Options()
    smoother_options = {
        "ksp_type": "chebyshev",
        "esteig_ksp_type": "cg",
        "ksp_chebyshev_esteig_steps": 10,
        "ksp_max_it": 2,
        "ksp_initial_guess_nonzero": True,
        "pc_type": "jacobi",
    }
    for key, val in smoother_options.items():
        opts[f"{solver_prefix}{key}"] = val
    solver.setOperators(As[i])
    solver.setFromOptions()
    solvers.append(solver)

u_1 = fem.Function(Vs[1])
# Set initial guess to be sinusoidal
# omega = 2 * np.pi * 1
# u_1.interpolate(lambda x: np.sin(omega * x[0]) * np.sin(omega * x[1]))

rs = [fem.Function(V) for V in Vs]
es = [fem.Function(V) for V in Vs]

u_1_file = io.VTXWriter(msh.comm, "u_1.bp", u_1, "bp4")
u_1_file.write(-1)
r_files = [io.VTXWriter(msh.comm, f"r_{i}.bp", r, "bp4") for (i, r) in enumerate(rs)]
e_files = [io.VTXWriter(msh.comm, f"e_{i}.bp", e, "bp4") for (i, e) in enumerate(es)]

r_norm_0 = residual(b, As[1], u_1).norm()
for i in range(num_iters):
    # Start of iteration
    print(f"Iteration {i + 1}:")
    print(
        f"    Initial:              residual norm = {(residual(b, As[1], u_1)).norm()}"
    )

    # Smooth A_1 u_1 = b_1 on fine level
    solvers[1].solve(b.vector, u_1.vector)

    # Compute residual r_1 = b_1 - A_1 u_1
    rs[1].vector.array[:] = residual(b, As[1], u_1)
    print(f"    After initial smooth: residual norm = {rs[1].vector.norm()}")
    r_files[1].write(i)

    # Interpolate residual to coarse level
    interp_op.multTranspose(rs[1].vector, rs[0].vector)
    r_files[0].write(i)

    # Solve A_0 e_0 = r_0 for error on coarse level
    petsc.set_bc(rs[0].vector, bcs=[bcs[0]])
    solvers[0].solve(rs[0].vector, es[0].vector)
    e_files[0].write(i)

    # Interpolate error to fine level
    es[1].interpolate(es[0])
    e_files[1].write(i)

    # Add error to solution u_1 += e_1
    u_1.vector.array[:] += es[1].vector.array

    print(
        f"    After correction:     residual norm = {(residual(b, As[1], u_1)).norm()}"
    )

    # Smooth on fine level A_1 u_1 = b_1
    solvers[1].solve(b.vector, u_1.vector)

    print(
        f"    After final smooth:   residual norm = {(residual(b, As[1], u_1)).norm()}"
    )

    u_1_file.write(i)

    r_norm = residual(b, As[1], u_1).norm()
    print(f"\n    Relative residual norm = {r_norm / r_norm_0}")

    # Compute error in solution
    e_u_1 = norm_L2(comm, u_e - u_1)
    print(f"\n    L2-norm of error in u_1 = {e_u_1}\n\n")
