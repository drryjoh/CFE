// Reflecting ("slip wall") boundary condition (task spec item 3,
// ROADMAP.md Phase 3, AGENTS.md #17's "reflective/slip" category).
//
// Mirrors the interior real-cell values into the ghost layer, but
// negates the single momentum component normal to the wall -- density,
// energy, and any tangential momentum component are copied unchanged.
// This is the standard FVM ghost-cell reflecting BC for an inviscid
// wall (e.g. Toro, "Riemann Solvers and Numerical Methods for Fluid
// Dynamics," Sec. 6.3.2; LeVeque, "Finite Volume Methods for Hyperbolic
// Problems," Sec. 7.9): combined with any physics-agnostic two-state
// numerical flux (Rusanov, HLLC, ...), the mirrored state makes the
// reconstructed normal velocity exactly zero at the wall face by
// symmetry -- the no-penetration condition a slip wall enforces --
// while a tangential velocity is left completely free to slide along
// the wall (the "slip" in "slip wall").
//
// This is CFE's first Field-aware BoundaryCondition (AGENTS.md #17,
// ARCHITECTURE.md #11). Every existing type in boundary_condition.hpp
// (PeriodicBoundary, StaticBoundary, InflowOutflowBoundary) only ever
// copies or writes raw component values -- it has no notion of what any
// component MEANS, so it can stay physics-agnostic. This type needs to
// know which raw component index is "momentum normal to this axis" for
// whichever multi-component Field is actually in play (component 1 for
// an X-normal wall in 2D/3D, but NOT in general -- see
// fields/euler/field.hpp's conserved-state layout comment), so it takes
// a `Field` template parameter and calls its static
// `momentum_index(Axis)` rather than hardcoding an index. This is
// deliberately the *lighter-weight* Field-aware shape ARCHITECTURE.md
// #11 anticipates (a static component-index lookup plus a sign flip),
// not the characteristic-based partial-specification subsonic
// inflow/outflow BC that same section also describes (which needs an
// equation of state and wave-speed information, and reads the interior
// state through the Field's Calculator rather than just indexing it) --
// confirmed against that distinction with Amelia/Violet before
// implementation; see docs/adr/0010-euler-field-and-shock-capturing.md.
//
// Works for any Field exposing `static constexpr Index
// momentum_index(Axis)` (currently just EulerField, fields/euler/field.hpp)
// -- nothing here is Euler-specific by name.
#pragma once

#include <cstddef>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe {

template <class Field>
struct SlipWallBoundary
{
  template <class Backend = CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    constexpr Index normal_component = Field::momentum_index(Axis::X);
    const Index py = grid.padded_ny();
    const Index pz = grid.padded_nz();
    Backend::run(grid.ngx * py * pz, [=] CFE_HOST_DEVICE(Index idx) mutable {
      const Index g = idx % grid.ngx;
      const Index rem = idx / grid.ngx;
      const Index j = rem % py;
      const Index k = rem / py;

      // Mirror image about the wall face: ghost layer g (0 = nearest
      // the wall) reads the real cell g layers in from the wall on the
      // interior side.
      const Index low_ghost = grid.flat_index(grid.ngx - 1 - g, j, k);
      const Index low_source = grid.flat_index(grid.ngx + g, j, k);
      const Index high_ghost = grid.flat_index(grid.ngx + grid.nx + g, j, k);
      const Index high_source = grid.flat_index(grid.ngx + grid.nx - 1 - g, j, k);

      for (Index c = 0; c < N; ++c) {
        const Scalar sign = (c == normal_component) ? Scalar(-1) : Scalar(1);
        field(low_ghost, c) = sign * field(low_source, c);
        field(high_ghost, c) = sign * field(high_source, c);
      }
    });
  }

  template <class Backend = CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    constexpr Index normal_component = Field::momentum_index(Axis::Y);
    const Index px = grid.padded_nx();
    const Index pz = grid.padded_nz();
    Backend::run(grid.ngy * px * pz, [=] CFE_HOST_DEVICE(Index idx) mutable {
      const Index g = idx % grid.ngy;
      const Index rem = idx / grid.ngy;
      const Index i = rem % px;
      const Index k = rem / px;

      const Index low_ghost = grid.flat_index(i, grid.ngy - 1 - g, k);
      const Index low_source = grid.flat_index(i, grid.ngy + g, k);
      const Index high_ghost = grid.flat_index(i, grid.ngy + grid.ny + g, k);
      const Index high_source = grid.flat_index(i, grid.ngy + grid.ny - 1 - g, k);

      for (Index c = 0; c < N; ++c) {
        const Scalar sign = (c == normal_component) ? Scalar(-1) : Scalar(1);
        field(low_ghost, c) = sign * field(low_source, c);
        field(high_ghost, c) = sign * field(high_source, c);
      }
    });
  }

  template <class Backend = CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    constexpr Index normal_component = Field::momentum_index(Axis::Z);
    const Index px = grid.padded_nx();
    const Index py = grid.padded_ny();
    Backend::run(grid.ngz * px * py, [=] CFE_HOST_DEVICE(Index idx) mutable {
      const Index g = idx % grid.ngz;
      const Index rem = idx / grid.ngz;
      const Index i = rem % px;
      const Index j = rem / px;

      const Index low_ghost = grid.flat_index(i, j, grid.ngz - 1 - g);
      const Index low_source = grid.flat_index(i, j, grid.ngz + g);
      const Index high_ghost = grid.flat_index(i, j, grid.ngz + grid.nz + g);
      const Index high_source = grid.flat_index(i, j, grid.ngz + grid.nz - 1 - g);

      for (Index c = 0; c < N; ++c) {
        const Scalar sign = (c == normal_component) ? Scalar(-1) : Scalar(1);
        field(low_ghost, c) = sign * field(low_source, c);
        field(high_ghost, c) = sign * field(high_source, c);
      }
    });
  }
};

}  // namespace cfe
