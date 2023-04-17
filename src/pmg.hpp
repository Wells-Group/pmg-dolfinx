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

  // Apply M^{-1}x = b
  int apply(Vector& x, const Vector& b, bool verbose = false) {}

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
};
} // namespace dolfinx::acc
