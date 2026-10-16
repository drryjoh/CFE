// Unit tests for cfe::io's binary checkpoint/restart format
// (src/cfe/io/restart_file.hpp, task tasks/0016-checkpoint-restart.md).
// These are format-level tests against a plain single-block
// CartesianGrid + Field -- they do not exercise a real solver at all.
// The actual "does restarting a real run reproduce the continuous run
// bit-for-bit" property (the crux of this feature) is tested separately
// in test_forward_step_restart.cpp, against the real two-block forward-
// step case; this file only checks that the file itself round-trips
// correctly and rejects the mismatches it claims to reject.
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/restart_file.hpp"
#include "test_framework.hpp"

namespace {

namespace fs = std::filesystem;
using Scalar = double;

// A fresh path per test (not a fixed shared name) so tests run in
// parallel/out of order never collide; removed again at the end of each
// test via a small RAII guard so a failed CFE_CHECK still cleans up
// (thrown exceptions still run destructors).
struct ScratchFile
{
  fs::path path;
  explicit ScratchFile(const char* tag)
      : path(fs::temp_directory_path() / (std::string("cfe_restart_test_") + tag + ".cferst"))
  {
  }
  ~ScratchFile() { std::remove(path.string().c_str()); }
};

cfe::CartesianGrid<Scalar> make_grid(std::size_t nx, std::size_t ny, std::size_t ng)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ny = ny;
  grid.nz = 1;
  grid.ngx = ng;
  grid.ngy = ng;
  grid.ngz = 0;
  grid.dx = Scalar(0.1);
  grid.dy = Scalar(0.2);
  grid.origin_x = Scalar(1.0);
  grid.origin_y = Scalar(-2.0);
  return grid;
}

}  // namespace

CFE_TEST(test_restart_file_round_trips_single_block_real_cells_exactly)
{
  constexpr std::size_t N = 3;
  const ScratchFile scratch("single_block");

  cfe::CartesianGrid<Scalar> grid = make_grid(5, 4, 2);
  cfe::Field<Scalar, N> field(grid.n_cells_total());
  for (std::size_t cell = 0; cell < grid.n_cells_total(); ++cell) {
    for (std::size_t c = 0; c < N; ++c) field(cell, c) = static_cast<Scalar>(cell) * Scalar(1.5) + Scalar(c);
  }
  // Poison the ghost cells with an obviously-wrong sentinel: the restart
  // file must never touch these on write, and read-back must leave
  // whatever the (freshly constructed) reading side already had there.
  for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
    for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
      const bool is_real = (i >= grid.ngx && i < grid.ngx + grid.nx && j >= grid.ngy && j < grid.ngy + grid.ny);
      if (!is_real) {
        const std::size_t cell = grid.flat_index(i, j, 0);
        for (std::size_t c = 0; c < N; ++c) field(cell, c) = Scalar(-999.0);
      }
    }
  }

  // `write_restart_checkpoint` takes a read-only `FieldView<const
  // Scalar,...>` per block -- `std::as_const` selects `Field::view()
  // const`'s `ConstView` overload rather than the mutable `View` one
  // `field.view()` would otherwise pick (Field has both, see
  // field/field.hpp; there is no implicit View->ConstView conversion,
  // by design -- the same reason `ForwardStepBlocks::write_restart`,
  // a `const` method, gets `ConstView` for free from `state_blocks.view
  // (...)` without writing this explicitly).
  cfe::io::write_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
      scratch.path.string(), {grid}, {std::as_const(field).view()}, std::size_t(1234), Scalar(0.5678));

  cfe::Field<Scalar, N> loaded(grid.n_cells_total());
  // Fill the fresh destination's ghosts with a DIFFERENT sentinel than
  // the source had, so "ghosts happen to match" can't hide a bug.
  for (std::size_t cell = 0; cell < grid.n_cells_total(); ++cell) {
    for (std::size_t c = 0; c < N; ++c) loaded(cell, c) = Scalar(-1.0);
  }

  const cfe::io::RestartCheckpointInfo<Scalar> info = cfe::io::read_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
      scratch.path.string(), {grid}, {loaded.view()});

  CFE_CHECK(info.step == 1234);
  CFE_CHECK(info.time == Scalar(0.5678));

  for (std::size_t i = 0; i < grid.nx; ++i) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, 0);
      for (std::size_t c = 0; c < N; ++c) {
        CFE_CHECK(loaded(cell, c) == field(cell, c));
      }
    }
  }
  // Ghosts were left exactly as the destination's own fresh-construction
  // sentinel, not overwritten by whatever garbage sat in the source's
  // ghosts -- confirms ghost cells are genuinely excluded from the file,
  // not merely "happened to round-trip."
  for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
    for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
      const bool is_real = (i >= grid.ngx && i < grid.ngx + grid.nx && j >= grid.ngy && j < grid.ngy + grid.ny);
      if (!is_real) {
        const std::size_t cell = grid.flat_index(i, j, 0);
        for (std::size_t c = 0; c < N; ++c) CFE_CHECK(loaded(cell, c) == Scalar(-1.0));
      }
    }
  }
}

CFE_TEST(test_restart_file_rejects_mismatched_grid_shape)
{
  constexpr std::size_t N = 1;
  const ScratchFile scratch("shape_mismatch");

  cfe::CartesianGrid<Scalar> written_grid = make_grid(5, 4, 2);
  cfe::Field<Scalar, N> field(written_grid.n_cells_total());
  cfe::io::write_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
      scratch.path.string(), {written_grid}, {std::as_const(field).view()}, std::size_t(0), Scalar(0));

  // Same ghost depth, different real-cell extent -- must be rejected,
  // not silently misread into the wrong shape.
  cfe::CartesianGrid<Scalar> different_grid = make_grid(6, 4, 2);
  cfe::Field<Scalar, N> loaded(different_grid.n_cells_total());

  bool threw = false;
  try {
    cfe::io::read_restart_checkpoint<Scalar, N, cfe::AoSLayout>(scratch.path.string(), {different_grid},
                                                                 {loaded.view()});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CFE_CHECK(threw);
}

CFE_TEST(test_restart_file_rejects_mismatched_component_count)
{
  const ScratchFile scratch("component_mismatch");

  cfe::CartesianGrid<Scalar> grid = make_grid(3, 3, 1);
  cfe::Field<Scalar, 2> field(grid.n_cells_total());
  cfe::io::write_restart_checkpoint<Scalar, 2, cfe::AoSLayout>(
      scratch.path.string(), {grid}, {std::as_const(field).view()}, std::size_t(0), Scalar(0));

  cfe::Field<Scalar, 3> loaded(grid.n_cells_total());
  bool threw = false;
  try {
    cfe::io::read_restart_checkpoint<Scalar, 3, cfe::AoSLayout>(scratch.path.string(), {grid}, {loaded.view()});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CFE_CHECK(threw);
}

// Three more reject paths the header's own doc comment claims are
// checked ("format_version, sizeof(Scalar), NComponents, and every
// block's shape") but which, until this test, had no referee of their
// own (Violet's review finding for this task) -- `NComponents` and
// block-shape mismatches were already covered above; these three round
// it out.
CFE_TEST(test_restart_file_rejects_mismatched_format_version)
{
  const ScratchFile scratch("format_version_mismatch");

  cfe::CartesianGrid<Scalar> grid = make_grid(3, 3, 1);
  cfe::Field<Scalar, 1> field(grid.n_cells_total());
  cfe::io::write_restart_checkpoint<Scalar, 1, cfe::AoSLayout>(
      scratch.path.string(), {grid}, {std::as_const(field).view()}, std::size_t(0), Scalar(0));

  // The format version is the first `std::uint32_t` written to the file
  // -- corrupt it in place (to a version number this build does not
  // recognize) and confirm `read_restart_checkpoint` rejects it rather
  // than reading the rest of the (now-presumably-incompatible) file as
  // if it matched this build's own layout.
  {
    std::FILE* file = std::fopen(scratch.path.string().c_str(), "r+b");
    CFE_CHECK(file != nullptr);
    const std::uint32_t bogus_version = cfe::io::kRestartFormatVersion + 1;
    CFE_CHECK(std::fwrite(&bogus_version, sizeof(bogus_version), 1, file) == 1);
    std::fclose(file);
  }

  cfe::Field<Scalar, 1> loaded(grid.n_cells_total());
  bool threw = false;
  try {
    cfe::io::read_restart_checkpoint<Scalar, 1, cfe::AoSLayout>(scratch.path.string(), {grid}, {loaded.view()});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CFE_CHECK(threw);
}

CFE_TEST(test_restart_file_rejects_mismatched_scalar_size)
{
  const ScratchFile scratch("scalar_size_mismatch");

  // Written as `double`, read back declaring `float` -- a genuine
  // precision mismatch between write and read, not a corrupted byte.
  cfe::CartesianGrid<double> grid = make_grid(3, 3, 1);
  cfe::Field<double, 1> field(grid.n_cells_total());
  cfe::io::write_restart_checkpoint<double, 1, cfe::AoSLayout>(
      scratch.path.string(), {grid}, {std::as_const(field).view()}, std::size_t(0), double(0));

  cfe::CartesianGrid<float> grid_f;
  grid_f.nx = grid.nx;
  grid_f.ny = grid.ny;
  grid_f.nz = grid.nz;
  grid_f.ngx = grid.ngx;
  grid_f.ngy = grid.ngy;
  grid_f.ngz = grid.ngz;
  grid_f.dx = static_cast<float>(grid.dx);
  grid_f.dy = static_cast<float>(grid.dy);
  grid_f.origin_x = static_cast<float>(grid.origin_x);
  grid_f.origin_y = static_cast<float>(grid.origin_y);
  cfe::Field<float, 1> loaded(grid_f.n_cells_total());

  bool threw = false;
  try {
    cfe::io::read_restart_checkpoint<float, 1, cfe::AoSLayout>(scratch.path.string(), {grid_f}, {loaded.view()});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CFE_CHECK(threw);
}

CFE_TEST(test_restart_file_rejects_mismatched_block_count)
{
  const ScratchFile scratch("block_count_mismatch");

  // Written with TWO blocks, read back expecting only one -- `n_blocks`
  // read from the file header must be checked against `grids.size()`
  // before any per-block shape/data read is attempted.
  cfe::CartesianGrid<Scalar> grid_a = make_grid(3, 3, 1);
  cfe::CartesianGrid<Scalar> grid_b = make_grid(4, 2, 1);
  cfe::Field<Scalar, 1> field_a(grid_a.n_cells_total());
  cfe::Field<Scalar, 1> field_b(grid_b.n_cells_total());
  cfe::io::write_restart_checkpoint<Scalar, 1, cfe::AoSLayout>(
      scratch.path.string(), {grid_a, grid_b}, {std::as_const(field_a).view(), std::as_const(field_b).view()},
      std::size_t(0), Scalar(0));

  cfe::Field<Scalar, 1> loaded_a(grid_a.n_cells_total());
  bool threw = false;
  try {
    cfe::io::read_restart_checkpoint<Scalar, 1, cfe::AoSLayout>(scratch.path.string(), {grid_a},
                                                                 {loaded_a.view()});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CFE_CHECK(threw);
}

CFE_TEST(test_restart_file_rejects_truncated_file)
{
  const ScratchFile scratch("truncated_file");

  cfe::CartesianGrid<Scalar> grid = make_grid(4, 4, 1);
  cfe::Field<Scalar, 2> field(grid.n_cells_total());
  cfe::io::write_restart_checkpoint<Scalar, 2, cfe::AoSLayout>(
      scratch.path.string(), {grid}, {std::as_const(field).view()}, std::size_t(0), Scalar(0));

  // Chop the file down to just its header, well short of a full block's
  // worth of real-cell data -- `detail::read_exact`'s own short-read
  // check must catch this, not silently return zero-filled/partial data.
  fs::resize_file(scratch.path, 32);

  cfe::Field<Scalar, 2> loaded(grid.n_cells_total());
  bool threw = false;
  try {
    cfe::io::read_restart_checkpoint<Scalar, 2, cfe::AoSLayout>(scratch.path.string(), {grid}, {loaded.view()});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CFE_CHECK(threw);
}

CFE_TEST(test_restart_file_round_trips_multiple_blocks_in_order)
{
  constexpr std::size_t N = 2;
  const ScratchFile scratch("two_blocks");

  cfe::CartesianGrid<Scalar> grid_a = make_grid(3, 2, 2);
  cfe::CartesianGrid<Scalar> grid_b = make_grid(4, 3, 2);
  cfe::Field<Scalar, N> field_a(grid_a.n_cells_total());
  cfe::Field<Scalar, N> field_b(grid_b.n_cells_total());
  for (std::size_t cell = 0; cell < grid_a.n_cells_total(); ++cell) {
    for (std::size_t c = 0; c < N; ++c) field_a(cell, c) = static_cast<Scalar>(cell + c) + Scalar(0.1);
  }
  for (std::size_t cell = 0; cell < grid_b.n_cells_total(); ++cell) {
    for (std::size_t c = 0; c < N; ++c) field_b(cell, c) = static_cast<Scalar>(1000 + cell + c) + Scalar(0.2);
  }

  cfe::io::write_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
      scratch.path.string(), {grid_a, grid_b}, {std::as_const(field_a).view(), std::as_const(field_b).view()},
      std::size_t(42), Scalar(1.0));

  cfe::Field<Scalar, N> loaded_a(grid_a.n_cells_total());
  cfe::Field<Scalar, N> loaded_b(grid_b.n_cells_total());
  const cfe::io::RestartCheckpointInfo<Scalar> info = cfe::io::read_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
      scratch.path.string(), {grid_a, grid_b}, {loaded_a.view(), loaded_b.view()});

  CFE_CHECK(info.step == 42);
  for (std::size_t i = 0; i < grid_a.nx; ++i) {
    for (std::size_t j = 0; j < grid_a.ny; ++j) {
      const std::size_t cell = grid_a.flat_index(grid_a.ngx + i, grid_a.ngy + j, 0);
      for (std::size_t c = 0; c < N; ++c) CFE_CHECK(loaded_a(cell, c) == field_a(cell, c));
    }
  }
  for (std::size_t i = 0; i < grid_b.nx; ++i) {
    for (std::size_t j = 0; j < grid_b.ny; ++j) {
      const std::size_t cell = grid_b.flat_index(grid_b.ngx + i, grid_b.ngy + j, 0);
      for (std::size_t c = 0; c < N; ++c) CFE_CHECK(loaded_b(cell, c) == field_b(cell, c));
    }
  }
}
