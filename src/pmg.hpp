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
template <typename Vector, typename Operator, typename Interpolator, typename Solver>
class MGPreconditioner
{
  /// The value type
  using T = typename Vector::value_type;

public:
  MGPreconditioner(std::vector<std::shared_ptr<const common::IndexMap>> maps, int bs)
      : _maps{maps}, _bs{bs}
  {
    int num_levels = maps.size();

    // Update vector sizes
    _du.resize(num_levels);
    _u.resize(num_levels);
    _r.resize(num_levels);
    _b.resize(num_levels);

    // Allocate data on device for temporary vectors
    for (int i = 0; i < num_levels; i++)
    {
      _du[i] = std::make_unique<Vector>(_map[i], _bs);
      _u[i] = std::make_unique<Vector>(_map[i], _bs);
      _r[i] = std::make_unique<Vector>(_map[i], _bs);
      _b[i] = std::make_unique<Vector>(_map[i], _bs);
    }
  }

  void set_solvers(std::vector<std::shared_ptr<Solver>>& solvers) { _solvers = solvers; }

  void set_operators(std::vector<std::shared_ptr<Operator>>&) { _operators = operators; }

  void set_interpolators(std::vector<std::shared_ptr<Interpolator>>& interpolators)
  {
    _interpolators = interpolators;
  }

  // Apply M^{-1}x = y
  int apply(Vector& x, const Vector& y, bool verbose = false)
  {
    int num_levels = _maps.size();

    // Compute residual in the finest level
    // Compute initial residual r0 = b - Ax0
    auto& A_fine = *_operators.back(); // Get reference to the finest operator
    auto& b_fine = *_b.back();         // get reference to the finest b

    A_fine(x, *_b.back());
    axpy(b_fine, T(-1), b_fine, y);

    for (int i = 0; i < num_levels; i++)
      _u[i]->set(T{0});

    for (int i = num_levels - 1; i > 0; i--)
    {
      _solvers[i]->apply(*_operators[i], *_b[i], *_u[i]);
      residual();// compute residual
      
    }

    A(x, *_y);
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
  std::vector<std::shared_ptr<Interpolator>> _interpolators;

  // Operators used to compute the residual
  std::vector<std::shared_ptr<Operator>> _operators;

  // Solvers for each level
  std::vector<std::shared_ptr<Solver>> _solvers;
};
} // namespace dolfinx::acc
