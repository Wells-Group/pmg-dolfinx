// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "vector.hpp"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>

#include "amd_gpu.hpp"

using namespace dolfinx;

namespace
{
template <typename T>
inline void tqli_ml(T* d, T* e, int m, int l)
{
  T g = (d[l + 1] - d[l]) / (2.0 * e[l]);
  T r = sqrt(g * g + 1.0);
  if (g >= 0)
    g = d[m] - d[l] + e[l] / (g + r);
  else
    g = d[m] - d[l] + e[l] / (g - r);

  T p = 0.0, s = 1.0, c = 1.0;
  for (int i = m - 1; i >= l; i--)
  {
    T f = s * e[i];
    T b = c * e[i];

    T r = sqrt(f * f + g * g);
    e[i + 1] = r;

    if (r == 0.0)
    {
      d[i + 1] -= p;
      e[m] = 0.0;
      return;
    }

    s = f / r;
    c = g / r;
    g = d[i + 1] - p;

    r = (d[i] - g) * s + 2.0 * c * b;
    p = s * r;
    d[i + 1] = g + p;
    g = c * r - b;
  }
  d[l] -= p;
  e[l] = g;
  e[m] = 0.0;
}

template <typename T>
int tqli(T* d, T* e, int n)
{

  auto tqli_m = [&d, &e, &n](int l)
  {
    int m;
    for (m = l; m < n - 1; m++)
    {
      T dd = std::abs(d[m]) + std::abs(d[m + 1]);
      if (std::abs(e[m]) + dd == dd)
        return m;
    }
    return n - 1;
  };

  for (int l = 0; l < n; l++)
  {
    int iter = 0;
    int m;
    while ((m = tqli_m(l)) != l)
    {
      if (iter++ == 300)
        return -1;
      tqli_ml(d, e, m, l);
    }
    e[l] = 0.0;
  }
  return 0;
}

} // namespace

namespace dolfinx::acc
{

/// Conjugate gradient method
template <typename Vector>
class CGSolver
{
  /// The value type
  using T = typename Vector::value_type;

public:
  CGSolver(std::shared_ptr<const common::IndexMap> map, int bs) : _map{map}, _bs{bs}
  {
    _r = std::make_unique<Vector>(_map, _bs);
    _y = std::make_unique<Vector>(_map, _bs);
    _p = std::make_unique<Vector>(_map, _bs);
  }

  void set_max_iterations(int max_iter)
  {
    _max_iter = max_iter;
    _alphas.reserve(_max_iter);
    _betas.reserve(_max_iter);
    _residuals.reserve(_max_iter);
  }
  void set_tolerance(double tolerance) { _rtol = tolerance; }

  void store_coefficients(bool val) { _store_coeffs = val; }

  std::vector<T> alphas() { return _alphas; }
  std::vector<T> betas() { return _betas; }

  std::vector<T> compute_eigenvalues()
  {
    int ne = _alphas.size();
    if (ne < 2)
      throw std::runtime_error("Insufficient data to compute eigenvalues");

    std::vector<T> d(ne, 0);
    std::vector<T> e(ne, 0);
    for (int i = 0; i < ne; ++i)
      d[i] = 1.0 / _alphas[i];
    for (int i = 0; i < ne - 1; ++i)
    {
      d[i + 1] += _betas[i] / _alphas[i];
      e[i] = std::sqrt(_betas[i]) / _alphas[i];
    }

    if (tqli(d.data(), e.data(), ne - 1) == -1)
      throw std::runtime_error("Eigenvalue estimate failed");

    return d;
  }

  T residual() const { return _residuals.back(); }

  // Solve Ax = b
  template <typename Operator>
  int solve(Operator& A, Vector& x, const Vector& b, bool verbose = false)
  {
    MPI_Comm comm = _map->comm();
    int rank;
    MPI_Comm_rank(comm, &rank);

    // TODO: check sizes

    // Compute initial residual r0 = b - Ax0
    A(x, *_y);
    axpy(*_r, T(-1), *_y, b);
    copy(*_p, *_r);

    T rnorm0 = squared_norm(*_r);
    T rnorm = rnorm0;

    spdlog::info("CG: rnorm0 = {}", rnorm0);

    // Iterations of CG
    const T rtol2 = _rtol * _rtol;

    int k = 0;
    while (k < _max_iter)
    {
      add_profiling_annotation("cg solver iteration");
      ++k;

      // MatVec
      // y = A.p;
      A(*_p, *_y);

      // Calculate alpha = r.r/p.y
      const T alpha = rnorm / inner_product(*_p, *_y);

      // Update x and r
      // Update x (x <- x + alpha*p)
      acc::axpy(x, alpha, *_p, x);

      // Update r (r <- r - alpha*y)
      acc::axpy(*_r, -alpha, *_y, *_r);

      // Update residual norm
      const T rnorm_new = squared_norm(*_r);
      const T beta = rnorm_new / rnorm;
      rnorm = rnorm_new;

      if (rank == 0 and verbose)
      {
        std::cout << "Iteration " << k << " residual " << std::sqrt(rnorm) << std::endl;
      }

      if (rnorm / rnorm0 < rtol2)
        break;

      // Update p.
      // Update p (p <- beta*p + r)
      axpy(*_p, beta, *_p, *_r);

      if (_store_coeffs)
      {
        _alphas.push_back(alpha);
        _betas.push_back(beta);
        _residuals.push_back(rnorm);
      }
      remove_profiling_annotation("cg solver iteration");
    }
    return k;
  }

private:
  /// Limit for the number of iterations the solver is allowed to do
  int _max_iter;

  /// Relative tolerance.
  T _rtol;

  /// Store coefficients of CG iterations
  bool _store_coeffs = false;

  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  /// Working vectors
  std::unique_ptr<Vector> _r;
  std::unique_ptr<Vector> _y;
  std::unique_ptr<Vector> _p;

  // Storage for coefficients of CG iterations (if required)
  std::vector<T> _alphas;
  std::vector<T> _betas;
  std::vector<T> _residuals;
};
} // namespace dolfinx::acc
