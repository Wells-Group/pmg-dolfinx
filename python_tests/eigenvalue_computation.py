from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import FunctionSpace, assemble_matrix, form, Function
from ufl import TestFunction, TrialFunction, dx, inner, grad

mesh = create_unit_square(MPI.COMM_WORLD, 1, 40)
Q = FunctionSpace(mesh, ("CG", 1))
u, v  = TestFunction(Q), TrialFunction(Q)
k = Function(Q)

def f(x):
    return (0.1*x[0] < 0.5) + 1e3

k.interpolate(f)

a = k * inner(grad(u), grad(v))*dx
a = form(a)

A = assemble_matrix(a)
A = A.to_dense()

from scipy import linalg
import numpy as np

vals = np.real(linalg.eigvals(A))
vals = sorted(vals)
print("# Eign Min/max = ", vals[0], vals[-1])

x = np.zeros(A.shape[0])
b = np.ones(A.shape[0])
r = b - A@x
p = r.copy()
rnorm = r.dot(r)

ne = 30
alpha = []
beta = []
for i in range(ne):
    y = A@p
    alpha.append(rnorm/(p.dot(y)))
    x += alpha[-1] * p
    r -= alpha[-1] * y
    rnorm_new = r.dot(r)
    beta.append(rnorm_new/rnorm)
    rnorm = rnorm_new
    p = beta[-1]*p + r

# Compute tridiagonal matrix (see Yousef Saad Iterative Methods ch:6.7.3)
trmat = np.zeros((ne, ne))
for i in range(ne):
    trmat[i, i] = 1/alpha[i]
for i in range(1, ne):
    trmat[i, i] += beta[i-1] / alpha[i-1]
    trmat[i, i - 1] = np.sqrt(beta[i - 1])/alpha[i - 1]
    trmat[i - 1, i] = np.sqrt(beta[i - 1])/alpha[i - 1]

np.set_printoptions(linewidth=200)

for j in range(2, ne):
    eig_est = sorted(np.real(linalg.eigvals(trmat[:j, :j])))
    print(j, eig_est[0], eig_est[-1])
