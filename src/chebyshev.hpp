// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "vector.hpp"
#include <algorithm>
#include <cmath>
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
  Chebyshev(std::shared_ptr<const common::IndexMap> map, int bs, std::array<T, 2> eig_range,
            int order)
      : _eig_range(eig_range)
  {
    _p = std::make_unique<Vector>(map, bs);
    _z = std::make_unique<Vector>(map, bs);
    _q = std::make_unique<Vector>(map, bs);
    _r = std::make_unique<Vector>(map, bs);
    _coeffs = container<T, acc::Device::CPP>(order, 0);

    T theta = (_eig_range[1] + _eig_range[0]) / 2.0;
    T delta = (_eig_range[1] - _eig_range[0]) / 2.0;

    switch (order - 1)
    {
    case 0:
    {
      _coeffs[0] = 1.0 / theta;
      break;
    }
    case 1:
    {
      _coeffs[0] = 2 / (delta * delta - 2 * theta * theta);
      _coeffs[1] = -4 * theta / (delta * delta - 2 * theta * theta);
      break;
    }
    case 2:
    {

      T tmp_0 = 3 * delta * delta;
      T tmp_1 = theta * theta;
      T tmp_2 = 1.0 / (-4 * std::pow(theta, 3) + theta * tmp_0);
      _coeffs[0] = -4 * tmp_2;
      _coeffs[1] = 12 / (tmp_0 - 4 * tmp_1);
      _coeffs[2] = tmp_2 * (tmp_0 - 12 * tmp_1);
      break;
    }
    case 3:
    {
      T tmp_0 = delta * delta;
      T tmp_1 = theta * theta;
      T tmp_2 = 8 * tmp_0;
      T tmp_3 = 1.0 / (std::pow(delta, 4) + 8 * std::pow(theta, 4) - tmp_1 * tmp_2);
      _coeffs[3] = tmp_3 * (32 * std::pow(theta, 3) - 16 * theta * tmp_0);
      _coeffs[2] = tmp_3 * (-48 * tmp_1 + tmp_2);
      _coeffs[1] = 32 * theta * tmp_3;
      _coeffs[0] = -8 * tmp_3;
      break;
    }
    case 4:
    {
      T tmp_0 = 5 * std::pow(delta, 4);
      T tmp_1 = std::pow(theta, 4);
      T tmp_2 = theta * theta;
      T tmp_3 = delta * delta;
      T tmp_4 = 60 * tmp_3;
      T tmp_5 = 20 * tmp_3;
      T tmp_6 = 1.0 / (16 * std::pow(theta, 5) - std::pow(theta, 3) * tmp_5 + theta * tmp_0);
      T tmp_7 = 160 * tmp_2;
      T tmp_8 = 1.0 / (tmp_0 + 16 * tmp_1 - tmp_2 * tmp_5);
      _coeffs[4] = tmp_6 * (tmp_0 + 80 * tmp_1 - tmp_2 * tmp_4);
      _coeffs[3] = tmp_8 * (tmp_4 - tmp_7);
      _coeffs[2] = tmp_6 * (-tmp_5 + tmp_7);
      _coeffs[1] = -80 * tmp_8;
      _coeffs[0] = 16 * tmp_6;
      break;
    }
    default:
      throw std::runtime_error("Chebyshev smoother not implemented for order = "
                               + std::to_string(order));
    }
  }

  void set_max_iterations(int max_iter) { _max_iter = max_iter; }

  template <typename Operator>
  T residual(Operator& A, Vector& x, const Vector& b)
  {
    A(x, *_q);
    acc::axpy(*_r, T(-1), *_q, b);
    return acc::norm(*_r, dolfinx::la::Norm::l2);
  }

  void set_diagonal(std::shared_ptr<Vector> diag)
  {
    _diag = diag;
    acc::transform(*_diag, [](auto e) { return T{1.0} / e; });
  }

  // Solve Ax = b
  template <typename Operator>
  void solve(Operator& A, Vector& x, const Vector& b, bool verbose)
  {
    // Horner's Rule is applied to avoid computing A^k directly
    for (int it = 0; it < _max_iter; it++)
    {
      A(x, *_q);
      acc::axpy(*_r, T(-1), *_q, b);

      _z->set(T{0.0});
      acc::axpy(*_z, _coeffs[0], *_r, *_z);

      for (std::size_t k = 1; k < _coeffs.size(); k++)
      {
        A(*_z, *_q);
        acc::axpy(*_z, _coeffs[k], *_r, *_q);
      }

      acc::axpy(x, T{1.0}, x, *_z);
    }
  }

private:
  /// Limit for the number of iterations the solver is allowed to do
  int _max_iter;

  /// Eigenvalues
  std::array<T, 2> _eig_range;

  acc::container<T, acc::Device::CPP> _coeffs;

  /// Working vectors
  std::unique_ptr<Vector> _p;
  std::unique_ptr<Vector> _z;
  std::unique_ptr<Vector> _q;
  std::unique_ptr<Vector> _r;
  std::shared_ptr<Vector> _diag;
};
} // namespace dolfinx::acc
