// Lightweight fixed-size mathematical container (AGENTS.md #6, #13).
//
// `FixedArray<Scalar, N>` is the single building block used for scalars
// (N=1), vectors, and fixed-size physics states in Phase 0. It never
// allocates: storage is a plain in-place `Scalar[N]` so instances are
// trivially copyable and safe to pass by value into device kernels.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"

namespace cfe {

template <class Scalar, std::size_t N>
class FixedArray
{
  static_assert(N > 0, "FixedArray requires at least one component");

 public:
  using value_type = Scalar;
  static constexpr std::size_t size() { return N; }

  CFE_HOST_DEVICE
  FixedArray() : data_{} {}

  CFE_HOST_DEVICE
  explicit FixedArray(const Scalar& fill)
  {
    for (std::size_t k = 0; k < N; ++k) data_[k] = fill;
  }

  CFE_HOST_DEVICE
  Scalar& operator[](std::size_t k) { return data_[k]; }
  CFE_HOST_DEVICE
  const Scalar& operator[](std::size_t k) const { return data_[k]; }

  CFE_HOST_DEVICE
  Scalar* data() { return data_; }
  CFE_HOST_DEVICE
  const Scalar* data() const { return data_; }

  CFE_HOST_DEVICE
  FixedArray& operator+=(const FixedArray& other)
  {
    for (std::size_t k = 0; k < N; ++k) data_[k] += other.data_[k];
    return *this;
  }

  CFE_HOST_DEVICE
  FixedArray& operator-=(const FixedArray& other)
  {
    for (std::size_t k = 0; k < N; ++k) data_[k] -= other.data_[k];
    return *this;
  }

  CFE_HOST_DEVICE
  FixedArray& operator*=(const Scalar& s)
  {
    for (std::size_t k = 0; k < N; ++k) data_[k] *= s;
    return *this;
  }

  CFE_HOST_DEVICE
  FixedArray& operator/=(const Scalar& s)
  {
    for (std::size_t k = 0; k < N; ++k) data_[k] /= s;
    return *this;
  }

 private:
  Scalar data_[N];
};

// Semantic aliases (ARCHITECTURE.md #4/#5). These are the same underlying
// container; the alias communicates intent at call sites only.
template <class Scalar>
using ScalarValue = FixedArray<Scalar, 1>;

template <class Scalar, std::size_t Dim>
using Vector = FixedArray<Scalar, Dim>;

template <class Scalar, std::size_t NComponents>
using State = FixedArray<Scalar, NComponents>;

}  // namespace cfe
