from mpi4py import MPI
from dolfinx.mesh import exterior_facet_indices, create_unit_cube
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
)
from dolfinx import fem, mesh
from ufl import TestFunction, TrialFunction, dx, inner, grad
import numpy as np
from cg import CGSolver
from petsc4py import PETSc


class Chebyshev:
    def __init__(self, A, max_iter, eig_range, kind, verbose=False) -> None:
        self.A = A
        self.max_iter = max_iter
        self.eig_range = eig_range
        self.kind = kind
        self.verbose = verbose
        self.coeffs = []

        self.theta = (eig_range[1] + eig_range[0]) / 2.0
        self.delta = (eig_range[1] - eig_range[0]) / 2.0

        if not(kind == 1 or kind == 4):
            raise ValueError(f"Invalid kind: {kind}")

        # if degree - 1 == 0:
        #     self.coeffs.append(1.0 / theta)
        # elif degree - 1 == 1:
        #     self.coeffs.append(2 / (delta * delta - 2 * theta * theta))
        #     self.coeffs.append(-4 * theta / (delta * delta - 2 * theta * theta))
        # elif degree - 1 == 2:
        #     tmp_0 = 3 * delta * delta
        #     tmp_1 = theta * theta
        #     tmp_2 = 1.0 / (-4 * theta**3 + theta * tmp_0)
        #     self.coeffs.append(-4 * tmp_2)
        #     self.coeffs.append(12 / (tmp_0 - 4 * tmp_1))
        #     self.coeffs.append(tmp_2 * (tmp_0 - 12 * tmp_1))
        # else:
        #     raise RuntimeError(f"Degree {degree} Chebyshev smoother not supported")

    def solve(self, b, x):
        if self.kind == 1:
            self.cheb1(b, x)
        elif self.kind == 4:
            self.cheb4(b, x)

    def cheb1(self, b, x):
        sigma = self.theta / self.delta
        rho = 1 / sigma

        r = b - self.A @ x
        d = r.copy() * float(1.0 / self.theta)

        for i in range(self.max_iter):
            x = x + d
            r = r - self.A @ d
            rho_new = 1/(2 * sigma - rho)
            d *= float(rho * rho_new)
            d += float(2 * rho/self.delta) * r.copy()
            rho = rho_new

            if self.verbose:
                print(f"Iteration {i + 1}, residual norm = {np.linalg.norm(r)}")

    def cheb4(self, b, x):
        x *= 0.25
        r = b - self.A @ x
        d = r.copy() * float(4 / (3 * self.eigrange[1]))

        for i in range(self.max_iter):
            x += beta * d
            r = r - self.A @ d
            d *= (2*i - 1)/(2*i + 3)

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    # msh = create_unit_cube(MPI.COMM_WORLD, 16, 18, 20, cell_type=mesh.CellType.hexahedron)
    msh = create_unit_cube(
        MPI.COMM_WORLD, 10, 10, 10, cell_type=mesh.CellType.hexahedron
    )
    print(f"Num cells = {msh.topology.index_map(msh.topology.dim).size_global}")

    V = fem.functionspace(msh, ("CG", 1))
    print(f"NDOFS = {V.dofmap.index_map.size_global}")
    u, v = TestFunction(V), TrialFunction(V)
    k = 2.0

    a = k * inner(grad(u), grad(v)) * dx
    a = fem.form(a)

    def f_expr(x):
        dx = (x[0] - 0.5) ** 2
        dy = (x[1] - 0.5) ** 2
        return 1000 * np.exp(-(dx + dy) / 0.02)

    f = fem.Function(V)
    f.interpolate(f_expr)
    L = fem.form(inner(f, v) * dx)

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = exterior_facet_indices(msh.topology)
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = fem.dirichletbc(0.0, dofs, V)

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()

    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    set_bc(b, [bc])

    cg_solver = CGSolver(A, 5, 1e-6, False)
    x = A.createVecRight()
    cg_solver.solve(b, x)
    est_eigs = cg_solver.compute_eigs()
    print(f"Estimated min/max eigenvalues = {est_eigs}")

    eigs = [0.8 * est_eigs[0], 1.2 * est_eigs[1]]

    smoother = Chebyshev(A, 30, eigs, 1, verbose=True)
    x.set(0.0)
    smoother.solve(b, x)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver_prefix = "solver_"
    solver.setOptionsPrefix(solver_prefix)
    opts = PETSc.Options()
    smoother_options = {
        "ksp_type": "chebyshev",
        "ksp_max_it": 30,
        "pc_type": "none",
        "ksp_chebyshev_eigenvalues": f"{eigs[0]}, {eigs[1]}",
        "ksp_chebyshev_kind": "first"
    }
    for key, val in smoother_options.items():
        opts[f"{solver_prefix}{key}"] = val
    solver.setOperators(A)
    def monitor(ksp, its, rnorm):
        print("Iteration: {}, rel. residual: {}".format(its, rnorm))
    solver.setMonitor(monitor)
    solver.setNormType(solver.NormType.NORM_UNPRECONDITIONED)
    solver.setFromOptions()
    solver.view()
    x.set(0.0)
    solver.solve(b, x)
