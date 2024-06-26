from mpi4py import MPI
from dolfinx.mesh import exterior_facet_indices, create_unit_cube
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
)
from dolfinx import fem, mesh
from ufl import TestFunction, TrialFunction, inner, grad, Measure
import numpy as np
from cg import CGSolver
from petsc4py import PETSc
import basix


class Chebyshev:
    def __init__(
        self, A, max_iter, eig_range, kind, jacobi=False, verbose=False
    ) -> None:
        self.A = A
        self.max_iter = max_iter
        self.eig_range = eig_range
        self.kind = kind
        self.verbose = verbose
        self.coeffs = []

        self.theta = (eig_range[1] + eig_range[0]) / 2.0
        self.delta = (eig_range[1] - eig_range[0]) / 2.0

        if not (kind == 1 or kind == 4):
            raise ValueError(f"Invalid kind: {kind}")

        if jacobi:
            self.S = A.getDiagonal()
            self.S.reciprocal()
        else:
            self.S = A.createVecRight()
            self.S.set(1.0)

    def solve(self, b, x):
        if self.kind == 1:
            self.cheb1(b, x)
        elif self.kind == 4:
            self.cheb4(b, x)

    def cheb1(self, b, x):
        sigma = self.theta / self.delta
        rho = 1 / sigma

        r = self.S * (b - self.A @ x)
        d = r.copy() * float(1.0 / self.theta)

        for i in range(self.max_iter):
            x += d
            r = r - self.S * (self.A @ d)
            rho_new = 1 / (2 * sigma - rho)
            d *= float(rho * rho_new)
            d += r.copy() * float(2 * rho_new / self.delta)
            rho = rho_new

            if self.verbose:
                print(
                    f"Iteration {i + 1}, PRECONDITIONED residual norm = {np.linalg.norm(r)}"
                )

    def cheb4(self, b, x):
        r = b - self.A @ x
        if self.verbose:
            print(f"Iteration 0, b norm = {np.linalg.norm(b)}")
            print(f"Iteration 0, x norm = {np.linalg.norm(x)}")
            print(f"Iteration 0, UNPRECONDITIONED residual norm = {np.linalg.norm(r)}")
            print(f"S.norm = {np.linalg.norm(self.S)}")
        d = self.S * r.copy() * float(4 / (3 * self.eig_range[1]))
        print(
            f"Iteration 0, z norm = {np.linalg.norm(d)} fac = {float(4 / (3 * self.eig_range[1]))}"
        )

        for i in range(1, self.max_iter + 1):
            x += d
            r = r - self.A @ d
            d *= float((2 * i - 1) / (2 * i + 3))
            d += (
                self.S * r.copy() * float((8 * i + 4) / (2 * i + 3) / self.eig_range[1])
            )
            if self.verbose:
                print(f"Iteration {i}, x norm = {np.linalg.norm(x)}")
                print(f"Iteration {i}, z norm = {np.linalg.norm(d)}")
                print(
                    f"Iteration {i}, UNPRECONDITIONED residual norm = {np.linalg.norm(r)}"
                )


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    n = 5
    k = 3
    tensor_prod = True
    msh = create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=mesh.CellType.hexahedron)
    print(f"Num cells = {msh.topology.index_map(msh.topology.dim).size_global}")

    family = basix.ElementFamily.P
    variant = basix.LagrangeVariant.gll_warped
    cell_type = msh.basix_cell()

    if tensor_prod:
        # Tensor product element
        basix_element = basix.create_tp_element(family, cell_type, k, variant)
        element = basix.ufl._BasixElement(basix_element)  # basix ufl element
        dx = Measure("dx", metadata={"quadrature_rule": "GLL", "quadrature_degree": 4})
    else:
        element = basix.ufl.element(family, cell_type, k, variant)
        dx = Measure("dx")

    V = fem.functionspace(msh, ("CG", 3))

    print(f"NDOFS = {V.dofmap.index_map.size_global}")
    u, v = TestFunction(V), TrialFunction(V)
    kappa = 2.0

    a = kappa * inner(grad(u), grad(v)) * dx
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
    apply_lifting(b, [a], [[bc]])
    set_bc(b, [bc])

    cg_solver = CGSolver(A, 20, 1e-6, jacobi=True, verbose=False)
    y = A.createVecRight()
    y.set(0.0)
    u = A.createVecRight()
    u.set(1.0)
    cg_solver.solve(b, y)
    est_eigs = cg_solver.compute_eigs()
    print(f"Estimated min/max eigenvalues = {est_eigs}")

    eigs = [0.1 * est_eigs[-1], 1.1 * est_eigs[-1]]

    max_cheb_iters = 30
    smoother = Chebyshev(A, max_cheb_iters, eigs, 4, jacobi=True, verbose=True)
    # Try with non-zero initial guess to check that works OK
    y.set(0.0)
    set_bc(y, [bc])
    smoother.solve(b, y)

    # Compare to PETSc
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver_prefix = "solver_"
    solver.setOptionsPrefix(solver_prefix)
    opts = PETSc.Options()
    smoother_options = {
        "ksp_type": "chebyshev",
        "ksp_max_it": max_cheb_iters,
        "pc_type": "jacobi",
        "ksp_chebyshev_eigenvalues": f"{eigs[0]}, {eigs[1]}",
        "ksp_chebyshev_kind": "fourth",
        "ksp_initial_guess_nonzero": True,
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
    y.set(0.0)
    set_bc(y, [bc])
    solver.solve(b, y)
