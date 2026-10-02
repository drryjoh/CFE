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

#include "cfe/backend/cuda/cuda_check.cuh"
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

  // Zero-initialized on construction, matching cfe::Field's host-side
  // std::vector (which always zero-inits) -- cudaMalloc alone does not.
  // This matters beyond tidiness: a solver's scratch buffers (e.g.
  // ssp_rk2_step's stage1) are only ever written at cells the solver
  // actually integrates, but grid/boundary/ghost_fill.hpp's per-axis
  // ghost-fill still needs to *read* every cell on the other two axes'
  // full padded extent (to correctly fill corners/edges) -- cells that
  // are real on one axis but ghost on another are never written by
  // anything, so without this they are a genuine uninitialized-memory
  // read, not just an untidy allocation. Caught directly by
  // `compute-sanitizer --tool initcheck`; see agent_history.md.
  explicit DeviceField(std::size_t n_cells) : n_cells_(n_cells), data_(nullptr)
  {
    const std::size_t bytes = n_cells_ * NComponents * sizeof(Scalar);
    CFE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    CFE_CUDA_CHECK(cudaMemset(data_, 0, bytes));
  }

  // Not CFE_CUDA_CHECK'd: throwing out of a destructor is undefined
  // behavior if it happens during stack unwinding, and there is nothing
  // meaningful to do with a cudaFree failure here beyond that anyway.
  ~DeviceField()
  {
    if (data_ != nullptr) cudaFree(data_);
  }

  DeviceField(const DeviceField&) = delete;
  DeviceField& operator=(const DeviceField&) = delete;

  void copy_from_host(const Scalar* host_data)
  {
    const std::size_t bytes = n_cells_ * NComponents * sizeof(Scalar);
    CFE_CUDA_CHECK(cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice));
  }

  void copy_to_host(Scalar* host_data) const
  {
    const std::size_t bytes = n_cells_ * NComponents * sizeof(Scalar);
    CFE_CUDA_CHECK(cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost));
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
