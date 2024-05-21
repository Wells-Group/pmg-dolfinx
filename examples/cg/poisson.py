from ufl import (Coefficient, Constant, FunctionSpace, Mesh,
                 TestFunction, TrialFunction, dx, grad, inner)
from basix.ufl import element

coord_element = element("Lagrange", "hexahedron", 1, shape=(3,))
mesh = Mesh(coord_element)

e = element("Lagrange", "hexahedron", 3)
V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
kappa = Constant(mesh)

a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx
