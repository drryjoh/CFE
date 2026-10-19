// Configurable floating-point scalar type and index types (AGENTS.md #11).
//
// Precision and indexing are independent concepts. `cfe::scalar` is the
// project-wide default precision, selected at configure time via the CMake
// cache variable `CFE_SCALAR_TYPE` (float|double). Math/field/backend
// components themselves remain templated on Scalar so a single build can
// still exercise both float and double explicitly (required by Phase 0),
// independent of whatever `cfe::scalar` currently resolves to.
#pragma once

#include <cstddef>
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

// General-purpose runtime index/count/offset type -- the canonical name
// for what raw `std::size_t` is used for almost everywhere in this
// codebase: buffer offsets, element counts, flat-index results,
// `Backend::run` iteration indices, loop variables walking a buffer.
// Distinct from `local_index`/`global_index` above, which are narrower-
// purpose partition-local/distributed indices, not a general index type
// -- `Index` is not a replacement for either and does not change their
// width/signedness choices. Same underlying type as `std::size_t` (same
// width, same signedness); this is a naming/readability alias only, the
// same precedent `cfe::scalar` already established for floating-point
// precision.
//
// Does NOT apply to a non-type template parameter that is a compile-time
// arity/dimension constant baked into a type itself -- e.g. `N` in
// `FixedArray<Scalar, std::size_t N>` (math/fixed_array.hpp),
// `NComponents` in `Field<Scalar, std::size_t NComponents, Layout>`
// (field/field.hpp). Those stay plain `std::size_t`, on purpose: they
// answer "how many, fixed forever for this type" (a different question
// than "which one, at this particular run-time moment" that `Index`
// answers), and a local `constexpr std::size_t` copy of one of those
// values (e.g. `constexpr std::size_t n_components = Field::n_components;`)
// stays `std::size_t` too, for the same reason. So seeing both spellings
// side by side in the same file -- `template <class Scalar, std::size_t N>`
// next to a loop `for (Index k = 0; k < N; ++k)` -- is expected, not a
// leftover inconsistency; see `math/fixed_array.hpp` for exactly that
// pairing, and `tasks/0015-index-type.md` for the full worked-out rule.
using Index = std::size_t;

// Which spatial axis. Lives here (not in grid/) because numerics code
// (numerics/numerical_flux/upwind.hpp) needs it to index a per-axis
// velocity/flux without depending on the grid module -- it is a
// fundamental dimensional concept, not a grid-storage detail. Values are
// stable (X=0, Y=1, Z=2) so `static_cast<Index>(axis)` can index a
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
