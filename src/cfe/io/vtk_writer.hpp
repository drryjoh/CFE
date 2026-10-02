// Minimal legacy-VTK (STRUCTURED_POINTS, ASCII) writer for a single
// cell-centered scalar on a CartesianGrid, per ARCHITECTURE.md #19 ("do
// not design a custom binary file format prematurely -- place a CFE API
// over an established portable format"): this is that API, not a new
// format, and it does not link the VTK library itself.
//
// Writes only the *real* (non-ghost) cells, in the same i-fastest,
// j-next, k-slowest order CartesianGrid::flat_index already uses --
// which is also the order VTK's STRUCTURED_POINTS format expects, so no
// reordering is needed. CELL_DATA (not POINT_DATA) matches this
// codebase's FVM cell-average semantics: `DIMENSIONS` describes the
// (nx+1, ny+1, nz+1) *point* lattice bounding those `nx*ny*nz` cells.
//
// Host-only, ASCII, and not performance-sensitive by design: this exists
// for producing a handful of visualization snapshots (see
// tutorials/scalar_advection_3d_visualization/), not for checkpointing a
// production-scale run -- a binary/XML (.vti) writer would replace this
// if that need arises later.
#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe::io {

template <class Scalar, class Layout>
void write_vtk_structured_points_cell_scalar(const std::string& path, const CartesianGrid<Scalar>& grid,
                                              FieldView<Scalar, 1, Layout> field, const char* scalar_name)
{
  std::ofstream out(path);
  if (!out) throw std::runtime_error("write_vtk_structured_points_cell_scalar: could not open " + path);

  out << "# vtk DataFile Version 3.0\n";
  out << "CFE scalar field\n";
  out << "ASCII\n";
  out << "DATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << (grid.nx + 1) << " " << (grid.ny + 1) << " " << (grid.nz + 1) << "\n";
  out << "ORIGIN " << grid.origin_x << " " << grid.origin_y << " " << grid.origin_z << "\n";
  out << "SPACING " << grid.dx << " " << grid.dy << " " << grid.dz << "\n";
  out << "CELL_DATA " << (grid.nx * grid.ny * grid.nz) << "\n";
  out << "SCALARS " << scalar_name << " double 1\n";
  out << "LOOKUP_TABLE default\n";

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        out << field(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0) << "\n";
      }
    }
  }
}

}  // namespace cfe::io
