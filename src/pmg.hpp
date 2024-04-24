// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "interpolate.hpp"
#include "vector.hpp"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>

using namespace dolfinx;

namespace dolfinx::acc
{
/// Conjugate gradient method
template <typename Vector, typename Operator, typename Prolongation, typename Restriction,
          typename Solver>
class MultigridPreconditioner
{
  /// The value type
  using T = typename Vector::value_type;

public:
  MultigridPreconditioner(std::vector<std::shared_ptr<const common::IndexMap>> maps, int bs)
      : _maps{maps}, _bs{bs}
  {
    int num_levels = _maps.size();

    // Update vector sizes
    _du.resize(num_levels);
    _u.resize(num_levels);
    _r.resize(num_levels);
    _b.resize(num_levels);

    // Allocate data on device for temporary vectors
    for (int i = 0; i < num_levels; i++)
    {
      _du[i] = std::make_unique<Vector>(_maps[i], _bs);
      _u[i] = std::make_unique<Vector>(_maps[i], _bs);
      _r[i] = std::make_unique<Vector>(_maps[i], _bs);
      _b[i] = std::make_unique<Vector>(_maps[i], _bs);
    }
  }

  void set_solvers(std::vector<std::shared_ptr<Solver>>& solvers) { _solvers = solvers; }

  void set_operators(std::vector<std::shared_ptr<Operator>>& operators) { _operators = operators; }

  void set_interpolation_kernels(std::vector<std::shared_ptr<Interpolator<T>>>& interpolators)
  {
    _interpolation_kernels = interpolators;
  }

  void set_prolongation_kernels(std::vector<std::shared_ptr<Interpolator<T>>>& prolong)
  {
    _prolongation_kernels = prolong;
  }

  void set_interpolators(std::vector<std::shared_ptr<Prolongation>>& interpolators)
  {
    _interpolation = interpolators;
  }

  void set_restriction_interpolators(std::vector<std::shared_ptr<Restriction>>& interpolators)
  {
    _res_interpolation = interpolators;
  }

  // Apply M^{-1}x = y
  void apply(const Vector& x, Vector& y, bool verbose = false)
  {

    dolfinx::common::Timer t0("~apply MultigridPreconditioner preconditioner");

    [[maybe_unused]] int num_levels = _maps.size();

    // Set to zeros
    for (int i = 0; i < num_levels - 1; i++)
      _u[i]->set(T{0});
    acc::copy(*_u.back(), y);

    acc::copy(*_b.back(), x);

    for (int i = num_levels - 1; i > 0; i--)
    {
      // r = b[i] - A[i] * u[i]
      (*_operators[i])(*_u[i], *_r[i]);
      axpy(*_r[i], T(-1), *_r[i], *_b[i]);

      double rn = acc::norm(*_r[i]);
      LOG(INFO) << "Residual norm before (" << i << ") = " << rn;

      // u[i] = M^-1 b[i]
      _solvers[i]->solve(*_operators[i], *_u[i], *_b[i], false);

      // r = b[i] - A[i] * u[i]
      (*_operators[i])(*_u[i], *_r[i]);
      axpy(*_r[i], T(-1), *_r[i], *_b[i]);

      rn = acc::norm(*_r[i]);
      LOG(INFO) << "Residual norm after (" << i << ") = " << rn;

      // Restrict residual from level i to level (i - 1)
      if (_interpolation_kernels[i - 1])
      {
        LOG(INFO) << "***** Using interpolation kernel " << i - 1;

        // Use "interpolation kernel" if available. Interpolate r[i] into b[i-1].
        _interpolation_kernels[i - 1]->interpolate(_r[i]->mutable_array().data(),
                                                   _b[i - 1]->mutable_array().data());
      }
      else
        (*_res_interpolation[i - 1])(*_r[i], *_b[i - 1], false);
    }

    // r = b[i] - A[i] * u[i]
    (*_operators[0])(*_u[0], *_r[0]);
    axpy(*_r[0], T(-1), *_r[0], *_b[0]);

    double rn = acc::norm(*_r[0]);
    LOG(INFO) << "Residual norm before (0) = " << rn;

    // Solve coarse problem
    _solvers[0]->solve(*_operators[0], *_u[0], *_b[0], false);

    // r = b[i] - A[i] * u[i]
    (*_operators[0])(*_u[0], *_r[0]);
    axpy(*_r[0], T(-1), *_r[0], *_b[0]);

    rn = acc::norm(*_r[0]);
    LOG(INFO) << "Residual norm after (0) = " << rn;

    for (int i = 0; i < num_levels - 1; i++)
    {
      // [coarse->fine] Prolong correction
      if (_prolongation_kernels[i])
      {
        LOG(INFO) << "***** Using prolongation kernel " << i;
        // Use "prolongation kernel" if available. Interpolate u[i] into du[i+1].
        _prolongation_kernels[i]->interpolate(_u[i]->mutable_array().data(),
                                              _du[i + 1]->mutable_array().data());
      }
      else
      {
        (*_interpolation[i])(*_u[i], *_du[i + 1], false);
      }

      // update U
      axpy(*_u[i + 1], T(1), *_u[i + 1], *_du[i + 1]);

      // r = b[i] - A[i] * u[i]
      (*_operators[i + 1])(*_u[i + 1], *_r[i + 1]);
      axpy(*_r[i + 1], T(-1), *_r[i + 1], *_b[i + 1]);
      double rn = acc::norm(*_r[i + 1]);
      LOG(INFO) << "Residual norm after u+du (" << i + 1 << ") = " << rn;

      // [fine] Post-smooth
      _solvers[i + 1]->solve(*_operators[i + 1], *_u[i + 1], *_b[i + 1], false);

      // r = b[i] - A[i] * u[i]
      (*_operators[i + 1])(*_u[i + 1], *_r[i + 1]);
      axpy(*_r[i + 1], T(-1), *_r[i + 1], *_b[i + 1]);
      rn = acc::norm(*_r[i + 1]);
      LOG(INFO) << "Residual norm after post-smoothing (" << i + 1 << ") = " << rn;
    }

    acc::copy(y, *_u.back());
  }

private:
  // Map describing the data layout
  std::vector<std::shared_ptr<const common::IndexMap>> _maps;

  // Block size
  int _bs;

  /// Working vectors
  std::vector<std::unique_ptr<Vector>> _du;
  std::vector<std::unique_ptr<Vector>> _u;
  std::vector<std::unique_ptr<Vector>> _r;
  std::vector<std::unique_ptr<Vector>> _b;

  // Prologation and restriction operatos
  // Size should be nlevels - 1
  std::vector<std::shared_ptr<Prolongation>> _interpolation;

  std::vector<std::shared_ptr<Interpolator<T>>> _interpolation_kernels;
  std::vector<std::shared_ptr<Interpolator<T>>> _prolongation_kernels;

  std::vector<std::shared_ptr<Restriction>> _res_interpolation;

  // Operators used to compute the residual
  std::vector<std::shared_ptr<Operator>> _operators;

  // Solvers for each level
  std::vector<std::shared_ptr<Solver>> _solvers;
};
} // namespace dolfinx::acc
