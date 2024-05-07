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


def u_i(x):
    "Initial guess of solution"
    # omega = 2 * np.pi * 1
    # return np.sin(omega * x[0]) * np.sin(omega * x[1])
    return np.zeros_like(x[0])


def level_print(string, level):
    print(f"{(len(ks) - level) * "    "}{string}")


n = 10
ks = [1, 3]
num_iters = 10
kappa = 1.0
comm = MPI.COMM_WORLD
msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=mesh.CellType.hexahedron)

# Exact solution
x = ufl.SpatialCoordinate(msh)
u_e = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])

# TODO Remove s
Vs = [fem.functionspace(msh, ("Lagrange", k)) for k in ks]
# Solutions
us = [fem.Function(V) for V in Vs]
# Residuals
rs = [fem.Function(V) for V in Vs]
# Corrections
dus = [fem.Function(V) for V in Vs]
# Right-hand sides
bs = [fem.Function(V) for V in Vs]

# Operators
As = []
# Boundary conditions
bcs = []
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
        petsc.assemble_vector(bs[-1].vector, L)
        petsc.apply_lifting(bs[-1].vector, [a], bcs=[[bc]])
        petsc.set_bc(bs[-1].vector, bcs=[bc])

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

u_files = [io.VTXWriter(msh.comm, f"u_{i}.bp", u, "bp4") for (i, u) in enumerate(us)]
r_files = [io.VTXWriter(msh.comm, f"r_{i}.bp", r, "bp4") for (i, r) in enumerate(rs)]
du_files = [io.VTXWriter(msh.comm, f"e_{i}.bp", e, "bp4") for (i, e) in enumerate(dus)]

r_norm_0 = residual(bs[-1], As[-1], us[-1]).norm()
us[-1].interpolate(u_i)
for iter in range(num_iters):
    # Start of iteration
    print(f"Iteration {iter + 1}:")

    # FIXME Zero initial guesses of error every iter?
    for u in us[:-1]:
        u.vector.set(0.0)

    for i in range(len(ks) - 1, 0, -1):
        level_print(f"Level {i}:", i)
        level_print(
            f"    Initial:              residual norm = {(residual(bs[i], As[i], us[i])).norm()}",
            i,
        )
        # Smooth A_1 u_1 = b_1 on fine level
        solvers[i].solve(bs[i].vector, us[i].vector)

        # Compute residual r_1 = b_1 - A_1 u_1
        rs[i].vector.array[:] = residual(bs[i], As[i], us[i])
        level_print(
            f"    After initial smooth: residual norm = {rs[i].vector.norm()}", i
        )
        r_files[i].write(iter)

        # Interpolate residual to coarse level
        interp_ops[i - 1].multTranspose(rs[i].vector, bs[i - 1].vector)

    # Solve A_0 e_0 = r_0 for error on coarse level
    petsc.set_bc(bs[0].vector, bcs=[bcs[0]])
    solvers[0].solve(bs[0].vector, us[0].vector)
    u_files[0].write(iter)
    level_print("Level 0:", 0)
    level_print(f"    residual norm = {(residual(bs[0], As[0], us[0])).norm()}", 0)

    for i in range(len(ks) - 1):
        # Interpolate error to fine level
        dus[i + 1].interpolate(us[i])
        du_files[i + 1].write(iter)

        # Add error to solution u_1 += e_1
        us[i + 1].vector.array[:] += dus[i + 1].vector.array

        level_print(f"Level {i + 1}:", i + 1)
        level_print(
            f"    After correction:     residual norm = {(residual(bs[i + 1], As[i + 1], us[i + 1])).norm()}",
            i + 1,
        )

        # Smooth on fine level A_1 u_1 = b_1
        solvers[i + 1].solve(bs[i + 1].vector, us[i + 1].vector)

        level_print(
            f"    After final smooth:   residual norm = {(residual(bs[i + 1], As[i + 1], us[i + 1])).norm()}",
            i + 1,
        )

        u_files[i + 1].write(iter)

    r_norm = residual(bs[-1], As[-1], us[-1]).norm()
    print(f"\n    Relative residual norm = {r_norm / r_norm_0}")

    # Compute error in solution
    e_u = norm_L2(comm, u_e - us[-1])
    print(f"\n    L2-norm of error in u_1 = {e_u}\n\n")
