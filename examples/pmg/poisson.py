from ufl import (Coefficient, Constant, FunctionSpace, Mesh,
                 TestFunction, TrialFunction, dx, grad, inner)
from basix.ufl import element


# Load namespace
ns = vars()
forms = []
for degree in range(1, 4):
    e = element("Lagrange", "hexahedron", degree)
    coord_element = element("Lagrange", "hexahedron", 1, rank=1)
    mesh = Mesh(coord_element)

    V = FunctionSpace(mesh, e)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Coefficient(V)
    kappa = Constant(mesh)

    aname = 'a' + str(degree)
    Lname = 'L' + str(degree)


    # Insert into namespace so that the forms will be named a1, a2, a3 etc.
    ns[aname] = kappa * inner(grad(u), grad(v)) * dx
    ns[Lname] = inner(f, v) * dx

    
    # Delete, so that the forms will get unnamed args and coefficients
    # and default to v_0, v_1, w0, w1 etc.
    del u, v, f, kappa

    forms += [ns[aname], ns[Lname]]