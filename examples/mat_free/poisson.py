from ufl import (Coefficient, Constant, FunctionSpace, Mesh, Measure,
                 TestFunction, TrialFunction, dx, grad, inner)
import basix
from basix.ufl import blocked_element, wrap_element

family = basix.ElementFamily.P
cell_type = basix.CellType.hexahedron
degree = 3
variant = basix.LagrangeVariant.gll_warped
e = wrap_element(basix.create_tp_element(family, cell_type, degree, variant))

coord_ele = basix.create_tp_element(family, cell_type, 1, variant)
coord_element = blocked_element(wrap_element(coord_ele), (3,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
kappa = Constant(mesh)

deg_to_quad = {2: 3, 3: 4, 4: 6, 5: 8}
dx = Measure("dx", metadata={"quadrature_rule": "GLL",
                             "quadrature_degree": deg_to_quad[degree]})
a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx
