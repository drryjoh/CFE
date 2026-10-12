#!/usr/bin/env python3
"""Renders an isosurface animation (MP4) of this tutorial's 3D scalar-
advection run, from the per-rank VTK time series
(vtk_output/frame_NNNN_rankNN.vtk) any single run of the tutorial
writes. Same mechanism as the sibling Burgers tutorial's own
`render_isosurface_video.py` (kept independently self-contained rather
than shared, consistent with every other pair of tutorials in this
repo) -- read that file's header comment for the full reasoning on the
per-rank tile stitching and why it must happen before any cell-to-point
averaging.

Unlike the Burgers sibling, this equation is LINEAR: the bump
translates rigidly at the fixed velocity (1,1,1), unchanged in shape,
so the isosurface here stays a plain sphere throughout -- only its
*position* moves (and wraps around the periodic domain once it crosses
an edge). That contrast is the point: run both videos side by side and
the difference between "it just moves" and "it steepens into a shock"
is immediately visible, not just asserted in prose.

Dependencies: pyvista (`pip install pyvista`). ffmpeg is pulled in
automatically via `imageio-ffmpeg`, no separate system install needed.

Usage: run the tutorial first (see README.md) so vtk_output/ exists,
then:

    python3 render_isosurface_video.py

Writes figures/isosurface.mp4.
"""
import pathlib
import re

import numpy as np
import pyvista as pv

HERE = pathlib.Path(__file__).resolve().parent
VTK_DIR = HERE / "vtk_output"
OUT_PATH = HERE / "figures" / "isosurface.mp4"

# Fixed across every frame. Background=1.0, peak=1.5, unchanged over
# time (no steepening here) -- any value strictly between the two shows
# the same sphere, just translating/wrapping.
ISO_VALUE = 1.25
FPS = 10
TOTAL_ORBIT_DEGREES = 75.0  # slow, subtle -- enough to read as 3D, not a dizzying spin


def discover_frame_indices():
    pattern = re.compile(r"frame_(\d+)_rank\d+\.vtk")
    indices = set()
    for f in VTK_DIR.glob("frame_*_rank*.vtk"):
        m = pattern.match(f.name)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def load_merged_frame(frame_idx):
    """Stitches every rank's own VTK piece for one frame back into a
    single full-domain `pv.ImageData`, by copying each piece's raw cell
    data into the right slice of one combined numpy array BEFORE any
    cell-to-point averaging happens -- doing the merge after averaging
    would leave a visible seam at rank boundaries, since each piece
    would otherwise compute its own boundary-face point values only
    from the cells it can see on its own side.
    """
    paths = sorted(VTK_DIR.glob(f"frame_{frame_idx:04d}_rank*.vtk"))
    pieces = [pv.read(str(p)) for p in paths]

    dx, dy, dz = pieces[0].spacing
    bounds = np.array([p.bounds for p in pieces])
    gx0, gx1 = bounds[:, 0].min(), bounds[:, 1].max()
    gy0, gy1 = bounds[:, 2].min(), bounds[:, 3].max()
    gz0, gz1 = bounds[:, 4].min(), bounds[:, 5].max()
    nx_g = round((gx1 - gx0) / dx)
    ny_g = round((gy1 - gy0) / dy)
    nz_g = round((gz1 - gz0) / dz)

    merged = np.full((nz_g, ny_g, nx_g), np.nan)
    for p in pieces:
        nx_l, ny_l, nz_l = (np.array(p.dimensions) - 1).astype(int)
        # VTK's own flat ordering for ImageData/STRUCTURED_POINTS cell
        # data is X-fastest, Y-next, Z-slowest -- the same convention
        # `cfe::io::write_vtk_structured_points_cell_scalar` itself
        # documents and writes in, so this reshape matches it exactly.
        vals = np.asarray(p.cell_data["state"]).reshape(nz_l, ny_l, nx_l)
        ox = round((p.bounds[0] - gx0) / dx)
        oy = round((p.bounds[2] - gy0) / dy)
        oz = round((p.bounds[4] - gz0) / dz)
        merged[oz:oz + nz_l, oy:oy + ny_l, ox:ox + nx_l] = vals

    if np.isnan(merged).any():
        raise RuntimeError(
            f"frame {frame_idx}: gap detected while stitching per-rank tiles together -- "
            "expected every rank's piece to exactly tile the global domain with no gaps."
        )

    grid = pv.ImageData(dimensions=(nx_g + 1, ny_g + 1, nz_g + 1), spacing=(dx, dy, dz),
                         origin=(gx0, gy0, gz0))
    grid.cell_data["state"] = merged.flatten(order="C")
    return grid


def main():
    frame_indices = discover_frame_indices()
    if not frame_indices:
        print(f"No frames found in {VTK_DIR} -- run the tutorial first (see README.md).")
        return

    plotter = pv.Plotter(off_screen=True, window_size=(960, 720))
    plotter.open_movie(str(OUT_PATH), framerate=FPS)

    orbit_step = TOTAL_ORBIT_DEGREES / max(1, len(frame_indices) - 1)

    for n, frame_idx in enumerate(frame_indices):
        grid = load_merged_frame(frame_idx)
        point_grid = grid.cell_data_to_point_data()
        surface = point_grid.contour(isosurfaces=[ISO_VALUE], scalars="state")

        plotter.clear_actors()
        plotter.add_mesh(surface, color="steelblue", smooth_shading=True, specular=0.3)
        plotter.add_mesh(grid.outline(), color="black", line_width=2)
        if n == 0:
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.3)
        else:
            plotter.camera.azimuth += orbit_step
        plotter.write_frame()
        print(f"frame {frame_idx} ({n + 1}/{len(frame_indices)}): {surface.n_points} isosurface points")

    plotter.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
