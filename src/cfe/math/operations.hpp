// Free-function math operations on FixedArray (AGENTS.md #13).
//
// Kept intentionally small for Phase 0: componentwise arithmetic plus the
// two semantic operations the task spec requires, `contract` and `weight`.
// Every function here must be host/device callable and small enough to
// inline into a kernel body.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/math/fixed_array.hpp"

namespace cfe {

// --- componentwise operations ---------------------------------------------

template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> operator+(FixedArray<Scalar, N> a,
                                                 const FixedArray<Scalar, N>& b) {
  a += b;
  return a;
}

template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> operator-(FixedArray<Scalar, N> a,
                                                 const FixedArray<Scalar, N>& b) {
  a -= b;
  return a;
}

template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> operator*(FixedArray<Scalar, N> a, const Scalar& s) {
  a *= s;
  return a;
}

template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> operator*(const Scalar& s, FixedArray<Scalar, N> a) {
  a *= s;
  return a;
}

template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> operator/(FixedArray<Scalar, N> a, const Scalar& s) {
  a /= s;
  return a;
}

// Componentwise (Hadamard) product. Named `componentwise_multiply` to avoid
// colliding with `operator*`, which is reserved for scalar scaling above.
template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> componentwise_multiply(const FixedArray<Scalar, N>& a,
                                                              const FixedArray<Scalar, N>& b) {
  FixedArray<Scalar, N> out;
  for (std::size_t k = 0; k < N; ++k) out[k] = a[k] * b[k];
  return out;
}

// --- contract ---------------------------------------------------------------
// Full contraction of two same-size containers into a single scalar:
//   contract(A, B) = sum_k A[k] * B[k]
// This is the generalized dot product used for both vectors and states.
template <class Scalar, std::size_t N>
CFE_HOST_DEVICE Scalar contract(const FixedArray<Scalar, N>& a, const FixedArray<Scalar, N>& b) {
  Scalar sum = Scalar(0);
  for (std::size_t k = 0; k < N; ++k) sum += a[k] * b[k];
  return sum;
}

// --- weight -------------------------------------------------------------
// Componentwise weighting: scales each component of `a` by the matching
// component of weight vector `w`. Returns a container of the same size
// (unlike `contract`, which reduces to a scalar).
template <class Scalar, std::size_t N>
CFE_HOST_DEVICE FixedArray<Scalar, N> weight(const FixedArray<Scalar, N>& a,
                                              const FixedArray<Scalar, N>& w) {
  return componentwise_multiply(a, w);
}

} // namespace cfe
