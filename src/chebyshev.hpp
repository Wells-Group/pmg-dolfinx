// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "vector.hpp"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <iostream>

using namespace dolfinx;

namespace dolfinx::acc
{

/// Conjugate gradient method
template <typename Vector>
class Chebyshev
{
  /// The value type
  using T = typename Vector::value_type;

public:
  Chebyshev(std::shared_ptr<const common::IndexMap> map, int bs, std::array<T, 2> eig_range) : _eig_range(eig_range)
  {
    _p = std::make_unique<Vector>(map, bs);
    _z = std::make_unique<Vector>(map, bs);
    _q = std::make_unique<Vector>(map, bs);
    _r = std::make_unique<Vector>(map, bs);
  }

  void set_max_iterations(int max_iter) { _max_iter = max_iter; }

  template <typename Operator>
  T residual(Operator& A, Vector& x, const Vector& b)
  {
    A(x, *_q);
    axpy(*_r, T(-1), *_q, b);
    T rnorm = squared_norm(*_r);
    return rnorm;
  }
  
  // Solve Ax = b
  template <typename Operator>
  void solve(Operator& A, Vector& x, const Vector& b, bool verbose)
  {
    T alpha, beta;
    T c = (_eig_range[1] - _eig_range[0]) / 2.0;
    T d = (_eig_range[1] + _eig_range[0]) / 2.0;

    // Compute initial residual r = b - Ax
    A(x, *_q);

    axpy(*_r, T(-1), *_q, b);

    for (int i = 0; i < _max_iter; ++i)
    {
      // Preconditioner z = M.solve(r);
      copy(*_z, *_r);

      if (i == 0)
      {
        copy(*_p, *_z);
        alpha = 2.0 / d;
      }
      else
      {
        beta = c * alpha / 2.0; // calculate new beta
        beta = beta * beta;
        alpha = 1.0 / (d - beta);       // calculate new alpha
        acc::axpy(*_p, beta, *_p, *_z); // update search direction
      }

      // q = A.p;
      A(*_p, *_q);

      // Update x and r
      // Update x (x <- x + alpha*p)
      acc::axpy(x, alpha, *_p, x);

      // Update r (r <- r - alpha*q)
      acc::axpy(*_r, -alpha, *_q, *_r);
    }
  }

private:
  /// Limit for the number of iterations the solver is allowed to do
  int _max_iter;

  /// Eigenvalues
  std::array<T, 2> _eig_range;

  /// Working vectors
  std::unique_ptr<Vector> _p;
  std::unique_ptr<Vector> _z;
  std::unique_ptr<Vector> _q;
  std::unique_ptr<Vector> _r;
};
} // namespace dolfinx::acc
