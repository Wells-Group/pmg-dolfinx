// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "vector.hpp"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>

using namespace dolfinx;

namespace dolfinx::acc
{

/// Conjugate gradient method
template <typename Vector>
class Chebyshev
{
  /// The value type
  using T = Vector::value_type;

public:
  CGSolver(std::shared_ptr<const common::IndexMap> map, int bs)
      : _map{map}, _bs{bs}
  {
    _r = std::make_unique<Vector>(_map, _bs);
    _y = std::make_unique<Vector>(_map, _bs);
    _p = std::make_unique<Vector>(_map, _bs);
  }

  void set_max_iterations(int max_iter) { _max_iter = max_iter; }

  // Solve Ax = b
  template <typename Operator>
  int solve(Operator& A, Vector& x, const Vector& b)
  {
    T alpha, beta, rho, rho1;
    T normb = squared_norm(*_r);

    // y = A.p;
    A(x, *_y);
    

  }

private:
  /// Limit for the number of iterations the solver is allowed to do
  int _max_iter;

  /// Relative tolerance.
  double _rtol;

  /// compute eigenvalues
  bool _compute_eigenvalues = false;

  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  /// Working vectors
  std::unique_ptr<Vector> _p;
  std::unique_ptr<Vector> _z;
  std::unique_ptr<Vector> _q;

  std::vector<T> _alphas;
  std::vector<T> _betas;
};
} // namespace dolfinx::acc
