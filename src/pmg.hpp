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
          typename Solver, typename CoarseSolver>
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

  void set_coarse_solver(std::shared_ptr<CoarseSolver> solver) { _coarse_solver = solver; }

  void set_operators(std::vector<std::shared_ptr<Operator>>& operators) { _operators = operators; }

  void set_interpolators(std::vector<std::shared_ptr<Prolongation>>& interpolators)
  {
    _interpolation = interpolators;
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

    spdlog::info("Copy x to b");
    acc::copy(*_b.back(), x);

    for (int i = num_levels - 1; i > 0; i--)
    {
      spdlog::info("Level {}", i);

      // r = b[i] - A[i] * u[i]
      spdlog::debug("Operator {} on u -> r", i);
      (*_operators[i])(*_u[i], *_r[i]);

      spdlog::debug("axpy");
      axpy(*_r[i], T(-1), *_r[i], *_b[i]);
      spdlog::info("Inital: rnorm = {}", acc::norm(*_r[i]));

      // u[i] = M^-1 b[i]
      _solvers[i]->solve(*_operators[i], *_u[i], *_b[i], false);

      // r = b[i] - A[i] * u[i]
      (*_operators[i])(*_u[i], *_r[i]);
      axpy(*_r[i], T(-1), *_r[i], *_b[i]);

      spdlog::info("After initial smooth: rnorm = {}", acc::norm(*_r[i]));

      // Restrict residual from level i to level (i - 1)
      (*_interpolation[i - 1])(*_r[i], *_b[i - 1], true);
    }

    spdlog::info("Level 0");
    // r = b[i] - A[i] * u[i]
    // (*_operators[0])(*_u[0], *_r[0]);
    // axpy(*_r[0], T(-1), *_r[0], *_b[0]);

    // Solve coarse problem
    if (_coarse_solver)
      _coarse_solver->solve(*_u[0], *_b[0]);
    else
      _solvers[0]->solve(*_operators[0], *_u[0], *_b[0], false);

    // r = b[i] - A[i] * u[i]
    (*_operators[0])(*_u[0], *_r[0]);
    axpy(*_r[0], T(-1), *_r[0], *_b[0]);

    spdlog::info("After coarse solve: rnorm = {}", acc::norm(*_r[0]));

    for (int i = 0; i < num_levels - 1; i++)
    {
      spdlog::info("Level {}", i + 1);

      // [coarse->fine] Prolong correction
      (*_interpolation[i])(*_u[i], *_du[i + 1], false);

      spdlog::info("norm(_u[{}]) = {}", i, acc::norm(*_u[i]));
      spdlog::info("norm(_du[{}]) = {}", i + 1, acc::norm(*_du[i + 1]));

      // update U
      axpy(*_u[i + 1], T(1), *_u[i + 1], *_du[i + 1]);

      // r = b[i] - A[i] * u[i]
      (*_operators[i + 1])(*_u[i + 1], *_r[i + 1]);
      axpy(*_r[i + 1], T(-1), *_r[i + 1], *_b[i + 1]);

      spdlog::info("After correction: rnorm = {}", acc::norm(*_r[i + 1]));

      // [fine] Post-smooth
      _solvers[i + 1]->solve(*_operators[i + 1], *_u[i + 1], *_b[i + 1], false);

      // r = b[i] - A[i] * u[i]
      (*_operators[i + 1])(*_u[i + 1], *_r[i + 1]);
      axpy(*_r[i + 1], T(-1), *_r[i + 1], *_b[i + 1]);
      double rn = acc::norm(*_r[i + 1]);
      spdlog::info("Residual norm after post-smoothing ({}) = {}", i + 1, rn);
    }

    if (verbose == true)
    {
      std::cout << "rnorm after PMG = " << acc::norm(*_r[num_levels - 1]) << "\n";
    }

    spdlog::info("----------- end of iteration ---------");

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

  // Operators used to compute the residual
  std::vector<std::shared_ptr<Operator>> _operators;

  std::shared_ptr<CoarseSolver> _coarse_solver;

  // Solvers for each level
  std::vector<std::shared_ptr<Solver>> _solvers;
};
} // namespace dolfinx::acc
