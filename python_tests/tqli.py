# Implementation of TQLI from "Numerical Recipes in C" by Press et al.

import numpy as np
from scipy import linalg


def tqli_ml(d, e, m, l):
    g = (d[l + 1] - d[l]) / (2.0 * e[l])
    r = np.sqrt(g**2 + 1.0)
    if g >= 0:
        g = d[m] - d[l] + e[l] / (g + r)
    else:
        g = d[m] - d[l] + e[l] / (g - r)
    s = c = 1.0
    p = 0.0

    for i in range(m - 1, l - 1, -1):
        f = s * e[i]
        b = c * e[i]
        r = np.sqrt(f**2 + g**2)
        e[i + 1] = r
        if r == 0.0:
            d[i + 1] -= p
            e[m] = 0.0
            break

        s = f / r
        c = g / r
        g = d[i + 1] - p

        r = (d[i] - g) * s + 2.0 * c * b
        p = s * r
        d[i + 1] = g + p
        g = c * r - b

    d[l] -= p
    e[l] = g
    e[m] = 0.0


def tqli_m(d, e, n, l):
    for m in range(l, n - 1):
        dd = abs(d[m]) + abs(d[m + 1])
        if abs(e[m]) + dd == dd:
            return m
    return n - 1


def tqli(d, e, max_iters=30):
    n = len(d)
    for l in range(n):
        iter = 0

        m = tqli_m(d, e, n, l)
        while m != l:
            if iter == max_iters:
                raise RuntimeError("Max TQLI iters reached")
            iter += 1
            tqli_ml(d, e, m, l)
            m = tqli_m(d, e, n, l)


d = np.array(
    [
        0.41346478,
        0.60294793,
        0.67818808,
        0.98075562,
        1.05615479,
        0.91088679,
        1.03500905,
        1.0903314,
        0.86035037,
        1.17691283,
    ]
)
e = np.array(
    [
        0.44621888,
        0.13732653,
        0.48100418,
        0.49721771,
        0.40803956,
        0.60132455,
        0.37474224,
        0.56690515,
        0.37387996,
        0,
    ]
)


if __name__ == "__main__":
    eigs_scipy = sorted(linalg.eigh_tridiagonal(d, e[:-1], eigvals_only=True))

    eigs_tqli = d.copy()
    tqli(eigs_tqli, e)

    assert np.allclose(eigs_scipy, eigs_tqli)
