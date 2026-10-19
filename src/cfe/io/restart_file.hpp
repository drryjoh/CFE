// Binary checkpoint/restart format for resuming a deterministic explicit
// CFE run partway through, without re-simulating from t=0 (task
// tasks/0016-checkpoint-restart.md -- motivation: re-verifying a change
// against tutorials/euler_2d_forward_step/'s full resolution sweep
// requires re-running every case from scratch, including the most
// expensive one (res288_hllc, tens of minutes), even when the change
// being verified provably cannot affect physics).
//
// See ARCHITECTURE.md #19 ("do not design a custom binary file format
// prematurely -- place a CFE API over an established portable format")
// for why this is NOT a violation of that principle: io/vtk_writer.hpp
// is that established-format API, and its own header comment already
// scopes it AWAY from this use ("not for checkpointing a production-
// scale run"). A restart checkpoint has the opposite requirements from a
// visualization snapshot -- read back only by CFE itself, on the same
// machine/build that wrote it, as fast as possible, bit-exact, never
// opened in ParaView -- so a bespoke compact binary format is the
// correct tool here, not a gap in that principle. `kRestartFormatVersion`
// below is exactly the hook ARCHITECTURE.md #19 asks for ("checkpoints"
// is explicitly named as a long-term native-format data requirement) to
// let a future format change (e.g. species metadata for a reacting-flow
// restart) detect and reject an incompatible old file instead of
// silently misreading it.
//
// What is (and is not) serialized, and why:
//   - grid metadata (real-cell extents, ghost depth, spacing, origin --
//     enough to reconstruct/verify a CartesianGrid<Scalar>);
//   - the field's raw REAL-cell data only, every component -- ghost
//     cells are deliberately excluded. This relies on one assumption,
//     checked directly end-to-end (not just asserted) by
//     tests/unit/test_forward_step_restart.cpp: every ghost cell this
//     codebase has is fully reconstructible from real-cell data alone,
//     because `FvmSolver::residual()` (solver/explicit/fvm_solver.hpp)
//     calls `fill_ghost_cells()` at the START of every residual
//     evaluation, before reading any state -- so a ghost cell's value is
//     NEVER state that must survive a restart; it is always recomputed,
//     from real cells plus the boundary condition (and, for
//     `BlockInterfaceBoundary`, the neighbor block's own real cells --
//     also restored), before it is next read, whether this process has
//     been running continuously or just resumed. If that reconstruction
//     were ever NOT exact, the first post-restart `residual()` call
//     would already diverge from the continuous run -- which is exactly
//     what the bit-identical continuous-vs-restart-resume test would
//     catch, not a separate, weaker, direct ghost-value check;
//   - the current step count and simulation time, as free-standing
//     metadata for the resuming driver's own bookkeeping/reporting --
//     every field's solve computes its next `dt` from the CURRENT state
//     alone (adaptive CFL), never from the elapsed time value itself, so
//     these two numbers are not state the solve depends on; they round-
//     trip purely so a restarted run can report "resumed at step 11000,
//     t=3.67" instead of starting that bookkeeping over from zero.
//
// NOT portable across machines/builds: no byte-swapping, and `Scalar`
// size / component count / block shape must match exactly what the file
// was written with (checked explicitly below, not silently coerced) --
// this is a same-machine, same-build-configuration checkpoint, not an
// interchange format like the VTK writer's ASCII output. If a reacting-
// flow restart file ever needs to move between machines with different
// precision, THAT need is the trigger to revisit this, not a reason to
// generalize ahead of it now (AGENTS.md #6).
//
// Host-only: `write_restart_checkpoint`/`read_restart_checkpoint` read/
// write a `FieldView` via plain host-side loops and `std::fwrite`/
// `std::fread` -- unlike `FieldView` itself (field/field.hpp, equally
// valid over host or device memory), THIS header assumes the `views`
// it is handed are already host-accessible. There is no CUDA restart
// path today (this task is CPU-only by explicit scope), so this has no
// current consequence; a future device-resident `Field`/`DeviceField`
// would need an explicit device-to-host copy before calling
// `write_restart_checkpoint` (and host-to-device after
// `read_restart_checkpoint`), not a change to this header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe::io {

// Bump this whenever the on-disk layout below changes incompatibly.
inline constexpr std::uint32_t kRestartFormatVersion = 1;

namespace detail {

// Thin RAII wrapper so every throw path below still closes the file.
// RAII ("Resource Acquisition Is Initialization") is the general C++
// idiom this is an instance of: tie a resource's lifetime to an
// object's constructor/destructor, so it is released automatically
// when that object goes out of scope -- including when an exception
// unwinds the stack past it, which is exactly what happens here on
// every `throw` in `write_exact`/`read_exact` below. A Python analogy:
// the same guarantee a `with`-block/context manager gives you, except
// enforced by the language itself (the destructor runs unconditionally
// on scope exit) rather than relying on the caller remembering to use
// `with`. `std::FILE*` (not `std::fstream`) is used throughout this
// header for its simpler, more explicit fixed-size `fread`/`fwrite`
// interface, which is what bulk binary (de)serialization actually
// wants.
struct FileHandle
{
  std::FILE* file;

  explicit FileHandle(const std::string& path, const char* mode) : file(std::fopen(path.c_str(), mode))
  {
    if (!file) throw std::runtime_error("restart_file: could not open " + path);
  }
  ~FileHandle()
  {
    if (file) std::fclose(file);
  }
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
};

template <class T>
void write_exact(std::FILE* file, const T* data, Index count, const std::string& path)
{
  if (count == 0) return;
  if (std::fwrite(data, sizeof(T), count, file) != count) {
    throw std::runtime_error("restart_file: short write to " + path);
  }
}

template <class T>
void read_exact(std::FILE* file, T* data, Index count, const std::string& path)
{
  if (count == 0) return;
  if (std::fread(data, sizeof(T), count, file) != count) {
    throw std::runtime_error("restart_file: short read (truncated/corrupt file?) from " + path);
  }
}

}  // namespace detail

// Enough of a `CartesianGrid<Scalar>` to reconstruct it, or to verify
// that a grid already built by the caller (same case, same resolution)
// matches what the checkpoint was written for. Sizes are written as
// fixed-width `std::uint64_t` (not `std::size_t`, whose width is
// platform-dependent) so a file is at least self-consistent across
// builds that differ only in that respect.
template <class Scalar>
struct RestartBlockShape
{
  std::uint64_t nx = 0, ny = 0, nz = 0;
  std::uint64_t ngx = 0, ngy = 0, ngz = 0;
  Scalar dx{}, dy{}, dz{};
  Scalar origin_x{}, origin_y{}, origin_z{};

  static RestartBlockShape from_grid(const CartesianGrid<Scalar>& grid)
  {
    RestartBlockShape shape;
    shape.nx = grid.nx;
    shape.ny = grid.ny;
    shape.nz = grid.nz;
    shape.ngx = grid.ngx;
    shape.ngy = grid.ngy;
    shape.ngz = grid.ngz;
    shape.dx = grid.dx;
    shape.dy = grid.dy;
    shape.dz = grid.dz;
    shape.origin_x = grid.origin_x;
    shape.origin_y = grid.origin_y;
    shape.origin_z = grid.origin_z;
    return shape;
  }

  // Exact equality, not a tolerance-based "close enough": a restart is
  // only ever meant to resume the SAME case at the SAME resolution, re-
  // constructed by the SAME deterministic code path that wrote the
  // checkpoint (e.g. `ForwardStepBlocks`'s constructor, given the same
  // `cells_per_unit_height`) -- so every field here is expected to be
  // bit-identical, and a mismatch (even a tiny one) means the caller is
  // loading the wrong checkpoint for the grid it just built, which
  // should fail loudly, not be shrugged off as "close enough."
  bool matches(const CartesianGrid<Scalar>& grid) const
  {
    return nx == grid.nx && ny == grid.ny && nz == grid.nz && ngx == grid.ngx && ngy == grid.ngy &&
           ngz == grid.ngz && dx == grid.dx && dy == grid.dy && dz == grid.dz && origin_x == grid.origin_x &&
           origin_y == grid.origin_y && origin_z == grid.origin_z;
  }
};

// Writes one multi-block checkpoint: a shared (step, time) pair plus,
// per block, its grid shape and every REAL cell's raw state (see this
// file's header comment for why ghost cells are excluded). `grids` and
// `views` must be the same length and in the same block order the
// caller will later pass to `read_restart_checkpoint` (for
// tests/verification/forward_step_setup.hpp's two-block case: `{grid1,
// grid2}` / `{state_blocks.view(kUpstream), state_blocks.view(
// kDownstream)}`, in that order -- a single-block case just passes a
// length-1 list of each).
//
// Not performance-sensitive by the same reasoning as vtk_writer.hpp:
// this runs once per checkpoint (not per step), so host-only, ASCII-free
// binary I/O with one bulk read/write per block is already far cheaper
// than the thousands of timesteps it is meant to let a developer skip
// re-running -- see tutorials/euler_2d_forward_step/README.md's restart
// section for the measured comparison.
template <class Scalar, std::size_t NComponents, class Layout>
void write_restart_checkpoint(const std::string& path, const std::vector<CartesianGrid<Scalar>>& grids,
                               const std::vector<FieldView<const Scalar, NComponents, Layout>>& views,
                               Index step, Scalar time)
{
  if (grids.size() != views.size()) {
    throw std::invalid_argument("write_restart_checkpoint: grids.size() != views.size()");
  }
  const detail::FileHandle handle(path, "wb");
  std::FILE* file = handle.file;

  const std::uint32_t format_version = kRestartFormatVersion;
  const std::uint32_t scalar_size = static_cast<std::uint32_t>(sizeof(Scalar));
  const std::uint32_t n_components = static_cast<std::uint32_t>(NComponents);
  const std::uint64_t step64 = static_cast<std::uint64_t>(step);
  const std::uint32_t n_blocks = static_cast<std::uint32_t>(grids.size());

  detail::write_exact(file, &format_version, 1, path);
  detail::write_exact(file, &scalar_size, 1, path);
  detail::write_exact(file, &n_components, 1, path);
  detail::write_exact(file, &step64, 1, path);
  detail::write_exact(file, &time, 1, path);
  detail::write_exact(file, &n_blocks, 1, path);

  // Gathered into one contiguous buffer per block, in the same
  // k-slowest/j-next/i-fastest real-cell traversal order
  // io/vtk_writer.hpp already uses, components innermost -- then written
  // in ONE `fwrite` per block, not one call per scalar: a per-scalar
  // `fwrite` would mean one library call per cell*component, which is
  // needless overhead against a one-time checkpoint write this feature
  // exists specifically to make cheap relative to the steps it saves
  // (see this task's `agent_history.md` entry for the measured I/O-vs-
  // simulation-time comparison).
  std::vector<Scalar> buffer;
  for (Index b = 0; b < grids.size(); ++b) {
    const CartesianGrid<Scalar>& grid = grids[b];
    const RestartBlockShape<Scalar> shape = RestartBlockShape<Scalar>::from_grid(grid);
    detail::write_exact(file, &shape, 1, path);

    const Index n_real_cells = grid.nx * grid.ny * grid.nz;
    buffer.resize(n_real_cells * NComponents);
    Index out = 0;
    for (Index k = 0; k < grid.nz; ++k) {
      for (Index j = 0; j < grid.ny; ++j) {
        for (Index i = 0; i < grid.nx; ++i) {
          const Index cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          for (Index c = 0; c < NComponents; ++c) buffer[out++] = views[b](cell, c);
        }
      }
    }
    detail::write_exact(file, buffer.data(), buffer.size(), path);
  }
}

// Metadata returned by `read_restart_checkpoint` alongside the scattered
// field data -- the caller's own step-count/elapsed-time bookkeeping, not
// anything the solve itself consumes (see this file's header comment).
// Bundled as one struct, returned by value, rather than split across a
// return value and an out-parameter: `step` and `time` are two halves of
// the same "what did this checkpoint record" concept, and a caller
// (e.g. `ForwardStepBlocks::read_restart`) wants both together anyway.
template <class Scalar>
struct RestartCheckpointInfo
{
  Index step = 0;
  Scalar time{};
};

// Reads a checkpoint written by `write_restart_checkpoint` directly into
// caller-owned storage (`views`), which must already be sized/allocated
// for grids matching what the file was written for -- i.e. the caller
// has already reconstructed the SAME case (e.g. built a fresh
// `ForwardStepBlocks` at the same resolution) and is loading this
// checkpoint's real-cell state into it; ghost cells in `views` are left
// untouched (see this file's header comment for why that is safe: the
// first subsequent `FvmSolver::residual()` call refills them before
// reading them regardless).
//
// Verifies `format_version`, `sizeof(Scalar)`, `NComponents`, and every
// block's shape against what `grids`/`views` actually are -- throws
// `std::runtime_error` on ANY mismatch rather than reading a
// differently-shaped file into the wrong layout (a silent partial/
// misaligned read would be far worse than a loud failure here).
template <class Scalar, std::size_t NComponents, class Layout>
RestartCheckpointInfo<Scalar> read_restart_checkpoint(const std::string& path,
                                                       const std::vector<CartesianGrid<Scalar>>& grids,
                                                       const std::vector<FieldView<Scalar, NComponents, Layout>>& views)
{
  if (grids.size() != views.size()) {
    throw std::invalid_argument("read_restart_checkpoint: grids.size() != views.size()");
  }
  const detail::FileHandle handle(path, "rb");
  std::FILE* file = handle.file;

  std::uint32_t format_version = 0, scalar_size = 0, n_components = 0, n_blocks = 0;
  std::uint64_t step64 = 0;
  Scalar time{};

  detail::read_exact(file, &format_version, 1, path);
  if (format_version != kRestartFormatVersion) {
    throw std::runtime_error("read_restart_checkpoint: " + path + " has format version " +
                              std::to_string(format_version) + ", this build expects " +
                              std::to_string(kRestartFormatVersion));
  }
  detail::read_exact(file, &scalar_size, 1, path);
  if (scalar_size != sizeof(Scalar)) {
    throw std::runtime_error("read_restart_checkpoint: " + path + " was written with sizeof(Scalar)=" +
                              std::to_string(scalar_size) + ", this build uses " +
                              std::to_string(sizeof(Scalar)) +
                              " -- float/double mismatch between write and read");
  }
  detail::read_exact(file, &n_components, 1, path);
  if (n_components != NComponents) {
    throw std::runtime_error("read_restart_checkpoint: " + path + " has NComponents=" +
                              std::to_string(n_components) + ", caller expects " +
                              std::to_string(NComponents));
  }
  detail::read_exact(file, &step64, 1, path);
  detail::read_exact(file, &time, 1, path);
  detail::read_exact(file, &n_blocks, 1, path);
  if (n_blocks != grids.size()) {
    throw std::runtime_error("read_restart_checkpoint: " + path + " has " + std::to_string(n_blocks) +
                              " block(s), caller supplied " + std::to_string(grids.size()));
  }

  std::vector<Scalar> buffer;
  for (Index b = 0; b < grids.size(); ++b) {
    const CartesianGrid<Scalar>& grid = grids[b];
    RestartBlockShape<Scalar> shape{};
    detail::read_exact(file, &shape, 1, path);
    if (!shape.matches(grid)) {
      throw std::runtime_error("read_restart_checkpoint: " + path + " block " + std::to_string(b) +
                                "'s grid shape does not match the grid supplied for it -- wrong "
                                "checkpoint/resolution for this case?");
    }

    const Index n_real_cells = grid.nx * grid.ny * grid.nz;
    buffer.resize(n_real_cells * NComponents);
    detail::read_exact(file, buffer.data(), buffer.size(), path);

    // Copied out of the (const-ref) `views` vector into a local, rather
    // than indexed in place: `FieldView` is a plain, trivially-copyable
    // pointer+size (field/field.hpp) with BOTH a const and non-const
    // `operator()`, so indexing `views[b]` directly on a `const
    // vector<FieldView<...>>&` binds the CONST element overload (`const
    // Scalar&`, not writable) even though `FieldView` itself is mutable
    // -- a copy sidesteps that without needing `views` to be non-const
    // (which would force every caller to hand over a mutable vector just
    // to satisfy this internal detail).
    FieldView<Scalar, NComponents, Layout> view = views[b];
    Index in = 0;
    for (Index k = 0; k < grid.nz; ++k) {
      for (Index j = 0; j < grid.ny; ++j) {
        for (Index i = 0; i < grid.nx; ++i) {
          const Index cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          for (Index c = 0; c < NComponents; ++c) view(cell, c) = buffer[in++];
        }
      }
    }
  }

  return RestartCheckpointInfo<Scalar>{static_cast<Index>(step64), time};
}

}  // namespace cfe::io
