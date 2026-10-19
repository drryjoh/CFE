// Field storage layout policies (AGENTS.md #10, ARCHITECTURE.md #4, ADR 0002).
//
// AoS vs SoA vs AoSoA must not be assumed optimal without measurement. Both
// layouts below expose the same `index(cell, component)` interface so a
// `Field` can be re-benchmarked under either policy without touching call
// sites, and additional layouts (e.g. AoSoA) can be added later without
// changing `Field` itself.
#pragma once

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"

namespace cfe {

// Components of a single cell are contiguous:
//   index = cell * n_components + component
struct AoSLayout
{
  CFE_HOST_DEVICE
  static Index index(Index cell,
                      Index component,
                      Index n_cells,
                      Index n_components)
  {
    (void)n_cells;
    return cell * n_components + component;
  }
};

// A given component is contiguous across all cells:
//   index = component * n_cells + cell
struct SoALayout
{
  CFE_HOST_DEVICE
  static Index index(Index cell,
                      Index component,
                      Index n_cells,
                      Index n_components)
  {
    (void)n_components;
    return component * n_cells + cell;
  }
};

}  // namespace cfe
