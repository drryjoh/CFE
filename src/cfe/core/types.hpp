// Configurable floating-point scalar type and index types (AGENTS.md #11).
//
// Precision and indexing are independent concepts. `cfe::scalar` is the
// project-wide default precision, selected at configure time via the CMake
// cache variable `CFE_SCALAR_TYPE` (float|double). Math/field/backend
// components themselves remain templated on Scalar so a single build can
// still exercise both float and double explicitly (required by Phase 0),
// independent of whatever `cfe::scalar` currently resolves to.
#pragma once

#include <cstdint>

namespace cfe {

#if defined(CFE_SCALAR_IS_FLOAT)
using scalar = float;
#else
using scalar = double;
#endif

// Local (within-partition/rank) indices are expected to fit comfortably in
// 32 bits for the problem sizes targeted by Phase 0. Global indices may need
// to address a distributed problem larger than 2^31 elements, so a wider
// type is used. Neither is ever `long`/`long long` per AGENTS.md #11.
using local_index = std::int32_t;
using global_index = std::int64_t;

// Which spatial axis. Lives here (not in grid/) because numerics code
// (numerics/numerical_flux/upwind.hpp) needs it to index a per-axis
// velocity/flux without depending on the grid module -- it is a
// fundamental dimensional concept, not a grid-storage detail. Values are
// stable (X=0, Y=1, Z=2) so `static_cast<std::size_t>(axis)` can index a
// per-axis Vector/FixedArray directly.
enum class Axis
{
  X = 0,
  Y = 1,
  Z = 2
};

enum class Side
{
  Low,
  High
};

}  // namespace cfe
