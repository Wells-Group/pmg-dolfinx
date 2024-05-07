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


class Chebyshev:
    def __init__(self, A, max_iter, eig_range, degree, verbose=False) -> None:
        self.A = A
        self.max_iter = max_iter
        self.eig_range = eig_range
        self.degree = degree
        self.verbose = verbose
        self.coeffs = []

        theta = (eig_range[1] + eig_range[0]) / 2.0
        delta = (eig_range[1] - eig_range[0]) / 2.0

        if degree - 1 == 0:
            self.coeffs.append(1.0 / theta)
        elif degree - 1 == 1:
            self.coeffs.append(2 / (delta * delta - 2 * theta * theta))
            self.coeffs.append(-4 * theta / (delta * delta - 2 * theta * theta))
        elif degree - 1 == 2:
            tmp_0 = 3 * delta * delta
            tmp_1 = theta * theta
            tmp_2 = 1.0 / (-4 * theta**3 + theta * tmp_0)
            self.coeffs.append(-4 * tmp_2)
            self.coeffs.append(12 / (tmp_0 - 4 * tmp_1))
            self.coeffs.append(tmp_2 * (tmp_0 - 12 * tmp_1))
        else:
            raise RuntimeError(f"Degree {degree} Chebyshev smoother not supported")

    def solve(self, b, x):
        for i in range(self.max_iter):
            r = b - self.A @ x
            # Have to cast to float for some reason
            z = float(self.coeffs[0]) * r

            for k in range(1, len(self.coeffs)):
                z = float(self.coeffs[k]) * r + self.A @ z

            x += z

            if self.verbose:
                print(f"Iteration {i + 1}, residual norm = {np.linalg.norm(r)}")


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

    smoother = Chebyshev(A, 30, (0.8 * est_eigs[0], 1.2 * est_eigs[1]), 3, verbose=True)
    smoother.solve(b, x)
