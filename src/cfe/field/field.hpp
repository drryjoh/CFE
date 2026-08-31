// Contiguous field-storage abstraction with compile-time component counts
// (AGENTS.md #8/#10/#12, ARCHITECTURE.md #4, task spec item 8).
//
// `Field` owns a single contiguous host-side allocation sized for
// `n_cells * NComponents` scalars, allocated once outside any parallel loop.
// `FieldView` is the trivially-copyable, pointer-based accessor that is
// actually captured by parallel_for lambdas -- it carries no ownership and
// is equally valid whether the pointer it wraps addresses host memory or
// device memory (see backend/cuda/device_field.hpp), which keeps the field
// interface backend-agnostic per AGENTS.md #9.
#pragma once

#include <cstddef>
#include <vector>

#include "cfe/core/macros.hpp"
#include "cfe/field/layout.hpp"

namespace cfe {

// Non-owning accessor. Safe to pass by value into a `CFE_DEVICE` lambda.
template <class Scalar, std::size_t NComponents, class Layout = AoSLayout>
class FieldView
{
 public:
  static constexpr std::size_t n_components() { return NComponents; }

  CFE_HOST_DEVICE
  FieldView() : data_(nullptr), n_cells_(0) {}
  CFE_HOST_DEVICE
  FieldView(Scalar* data, std::size_t n_cells) : data_(data), n_cells_(n_cells) {}

  CFE_HOST_DEVICE
  std::size_t n_cells() const { return n_cells_; }
  CFE_HOST_DEVICE
  std::size_t size() const { return n_cells_ * NComponents; }

  CFE_HOST_DEVICE
  Scalar& operator()(std::size_t cell, std::size_t component)
  {
    return data_[Layout::index(cell, component, n_cells_, NComponents)];
  }

  CFE_HOST_DEVICE
  const Scalar& operator()(std::size_t cell, std::size_t component) const
  {
    return data_[Layout::index(cell, component, n_cells_, NComponents)];
  }

  CFE_HOST_DEVICE
  Scalar* data() { return data_; }
  CFE_HOST_DEVICE
  const Scalar* data() const { return data_; }

 private:
  Scalar* data_;
  std::size_t n_cells_;
};

// Owning host-side storage. Allocation happens only in the constructor,
// never inside a parallel loop.
template <class Scalar, std::size_t NComponents, class Layout = AoSLayout>
class Field
{
 public:
  using View = FieldView<Scalar, NComponents, Layout>;

  explicit Field(std::size_t n_cells) : n_cells_(n_cells), storage_(n_cells * NComponents) {}

  std::size_t n_cells() const { return n_cells_; }
  static constexpr std::size_t n_components() { return NComponents; }
  std::size_t size() const { return storage_.size(); }

  Scalar& operator()(std::size_t cell, std::size_t component)
  {
    return storage_[Layout::index(cell, component, n_cells_, NComponents)];
  }

  const Scalar& operator()(std::size_t cell, std::size_t component) const
  {
    return storage_[Layout::index(cell, component, n_cells_, NComponents)];
  }

  Scalar* data() { return storage_.data(); }
  const Scalar* data() const { return storage_.data(); }

  View view() { return View(storage_.data(), n_cells_); }
  View view() const { return View(const_cast<Scalar*>(storage_.data()), n_cells_); }

 private:
  std::size_t n_cells_;
  std::vector<Scalar> storage_;
};

}  // namespace cfe
