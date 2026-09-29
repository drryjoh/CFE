// Single call-site dispatch for filling one axis's ghost layer.
//
// This function is the actual "swappable neighbor provider" seam
// referenced in tasks/0002-phase1-cartesian-grid-scalar-transport.md's
// AMR-readiness constraint: solver code always calls
// `fill_ghost_cells(field, grid, axis, boundary)`, never touches
// `(i,j,k)+-1` indexing itself. Swapping `boundary`'s TYPE -- static,
// periodic, and later an MPI halo-exchange provider or a coarse-fine AMR
// interpolation provider -- is the entire extension point; call sites
// never change.
#pragma once

#include "cfe/backend/parallel_for.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe {

template <class Scalar, std::size_t N, class Layout, class Boundary, class Backend = CpuParallelFor>
void fill_ghost_cells(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar>& grid, Axis axis,
                      const Boundary& boundary)
{
  switch (axis) {
    case Axis::X:
      boundary.template fill_x<Backend>(field, grid);
      return;
    case Axis::Y:
      boundary.template fill_y<Backend>(field, grid);
      return;
    case Axis::Z:
      boundary.template fill_z<Backend>(field, grid);
      return;
  }
}

}  // namespace cfe
