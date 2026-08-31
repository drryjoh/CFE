// Device-side counterpart to cfe::Field (task spec items 5 and 8).
//
// UNVERIFIED: see cuda_backend.cuh -- no CUDA hardware/toolkit was available
// to build or run this file during Phase 0.
//
// Allocation happens once in the constructor/`resize`, never inside a
// parallel loop (AGENTS.md #10). `view()` returns the same `FieldView` used
// by the CPU backends, so kernels written against `FieldView` do not need
// to know whether the pointer they were given addresses host or device
// memory.
#pragma once

#include <cstddef>
#include <stdexcept>

#include "cfe/field/field.hpp"
#include "cfe/field/layout.hpp"

namespace cfe {
namespace backend {
namespace cuda {

template <class Scalar, std::size_t NComponents, class Layout = AoSLayout>
class DeviceField
{
 public:
  using View = FieldView<Scalar, NComponents, Layout>;

  explicit DeviceField(std::size_t n_cells) : n_cells_(n_cells), data_(nullptr)
  {
    const std::size_t bytes = n_cells_ * NComponents * sizeof(Scalar);
    if (cudaMalloc(reinterpret_cast<void**>(&data_), bytes) != cudaSuccess) {
      throw std::runtime_error("cfe::backend::cuda::DeviceField: cudaMalloc failed");
    }
  }

  ~DeviceField()
  {
    if (data_ != nullptr) cudaFree(data_);
  }

  DeviceField(const DeviceField&) = delete;
  DeviceField& operator=(const DeviceField&) = delete;

  void copy_from_host(const Scalar* host_data)
  {
    const std::size_t bytes = n_cells_ * NComponents * sizeof(Scalar);
    cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice);
  }

  void copy_to_host(Scalar* host_data) const
  {
    const std::size_t bytes = n_cells_ * NComponents * sizeof(Scalar);
    cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost);
  }

  std::size_t n_cells() const { return n_cells_; }
  View view() { return View(data_, n_cells_); }

 private:
  std::size_t n_cells_;
  Scalar* data_;
};

}  // namespace cuda
}  // namespace backend
}  // namespace cfe
