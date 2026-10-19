// One contiguous allocation backing several same-kind `Field`s (AGENTS.md
// #10). See ARCHITECTURE.md #21 ("Memory layout: ownership, views, and
// block-structured grids") for the full picture this fits into --
// summary here:
//
// `ForwardStepBlocks` (tests/verification/forward_step_setup.hpp) needs
// one "state" buffer, one "stage1" buffer, and one "scratch" buffer per
// block of a block-structured grid (two blocks today -- see ADR 0011).
// Giving each block its own independent `Field` means six unrelated heap
// allocations for what is conceptually three buffers, each just
// decomposed across more than one block. `BlockSet` instead owns ONE
// `std::vector<Scalar>` sized for the sum of every block's cell count,
// and hands out a `FieldView` per block at the right pointer offset into
// that single allocation -- block `b`'s view starts immediately after
// block `b-1`'s last scalar, so block `b+1`'s data is the next thing in
// memory after block `b`'s.
//
// `FieldView` already has zero assumption that it owns, or sits at the
// start of, its own allocation (see field.hpp) -- it is exactly
// `(Scalar* data, Index n_cells)`. That is what makes this
// possible with no change to `FieldView` itself, and no change at all to
// any code that only ever consumes a `FieldView` (`FvmSolver`,
// `BlockInterfaceBoundary`, `ssp_rk2_combine_stage1/2`, ...): they cannot
// tell a `BlockSet`-provided view from a `Field`-provided one.
//
// Python/NumPy analogy: this is like allocating ONE big NumPy array
// (`np.zeros(total_cells)`) and handing out two slices of it
// (`arr[0:n1]`, `arr[n1:n1+n2]`) instead of two separate small arrays --
// each slice is a VIEW into the same backing buffer, not a copy, so it
// is genuinely the same memory, just addressed through a different
// pointer/offset (the two slices never overlap each other's cells).
#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/field/layout.hpp"

namespace cfe {

template <class Scalar, std::size_t NComponents, class Layout = AoSLayout>
class BlockSet
{
 public:
  using View = FieldView<Scalar, NComponents, Layout>;
  using ConstView = FieldView<const Scalar, NComponents, Layout>;

  // `block_cell_counts`: one entry per block, in the order `view(block)`
  // will address them. The backing allocation is sized once, up front,
  // to the sum of these -- never resized afterward.
  //
  // `std::initializer_list<Index>` (rather than, say, two
  // `Index` parameters) is what lets brace-init syntax --
  // `BlockSet<...> x{10, 20}`, or `{10, 20, 30}` for three blocks --
  // bind to THIS one constructor regardless of how many entries are
  // listed: a braced list preferentially matches an initializer-list
  // constructor over any other overload, and the list's length is a
  // runtime property of the call, not part of the type (unlike
  // `NComponents`, which IS a compile-time template parameter). Adding a
  // third block to a future case is just adding a third entry at the
  // call site -- no new constructor needed here. One real limitation:
  // an initializer list must be written as a literal braced list at the
  // call site; it cannot bind to an already-built
  // `std::vector<Index>`. A future caller needing a genuinely
  // runtime-determined block count would need a different constructor
  // overload (e.g. one taking a `std::vector<Index>` directly) --
  // not needed by anything in this codebase today.
  explicit BlockSet(std::initializer_list<Index> block_cell_counts)
      : block_cell_counts_(block_cell_counts)
  {
    scalar_offsets_.reserve(block_cell_counts_.size());
    Index total_scalars = 0;
    for (const Index n_cells : block_cell_counts_) {
      scalar_offsets_.push_back(total_scalars);
      total_scalars += n_cells * NComponents;
    }
    storage_.resize(total_scalars);
  }

  // Mirrors `Field`'s own `n_cells()`/`size()` shape (field.hpp) for a
  // reader already familiar with it; not currently called by
  // `ForwardStepBlocks` (which gets a block's cell count from its own
  // `CartesianGrid` already), kept for the same reason `Field` exposes
  // them -- a natural, near-zero-cost accessor pair for a type whose
  // whole job is "how many cells, and where are they."
  Index n_blocks() const { return block_cell_counts_.size(); }
  Index n_cells(Index block) const { return block_cell_counts_[block]; }
  // Total scalars across every block -- the one allocation's size.
  Index size() const { return storage_.size(); }

  View view(Index block)
  {
    return View(storage_.data() + scalar_offsets_[block], block_cell_counts_[block]);
  }
  // See `Field::view() const`'s own comment: `storage_.data()` on a
  // const `BlockSet` is already `const Scalar*`, exactly what
  // `ConstView` expects -- no `const_cast` needed.
  ConstView view(Index block) const
  {
    return ConstView(storage_.data() + scalar_offsets_[block], block_cell_counts_[block]);
  }

 private:
  std::vector<Index> block_cell_counts_;
  // Scalar (not cell) offset of each block's first element into
  // `storage_` -- i.e. already multiplied by `NComponents`, so `view()`
  // can index `storage_.data()` directly.
  std::vector<Index> scalar_offsets_;
  std::vector<Scalar> storage_;
};

}  // namespace cfe
