#!/usr/bin/env python3
"""Renders an isosurface animation (MP4) of this tutorial's 3D Burgers
run, from the per-rank VTK time series (vtk_output/frame_NNNN_rankNN.vtk)
any single run of the tutorial writes.

Each frame's 8 (or however many) per-rank tiles are stitched back into
one full-domain grid (the exact same tiling already verified numerically
elsewhere in this tutorial's own README -- no gaps/overlaps), then one
isosurface is extracted via VTK's marching-cubes filter (through
PyVista, a Python wrapper around VTK) at a FIXED threshold shared by
every frame, so the surface's shape -- not just its existence -- is
comparable across the whole video. Rendered from a fixed-elevation,
slowly-orbiting camera positioned outside the simulated cube (the
cube's own wireframe outline is drawn for spatial reference), so the
viewer can watch the shock-formation deformation happen in 3D rather
than read it off a single static picture.

Why the periodic "ghost" fragments at the box corners are real, not a
bug: this equation's background state (1.0) self-advects too (Burgers'
characteristic speed is the state itself, never zero here), so the
*entire* domain drifts at roughly that speed and wraps around the
periodic box -- the same correctness property
`tests/mpi/test_mpi_burgers_steepening.cpp` already verifies
numerically at the rank-to-rank boundaries holds at the domain's own
periodic boundary too. Left in deliberately: it's accurate physics, and
a visual demonstration that periodic wrap-around works, not just
rank-to-rank halo exchange.

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

# Fixed across every frame (not re-centered on each frame's own peak):
# chosen so the surface intersects the steepening/rarefacting region
# (not just the smooth cap near the peak) throughout the whole run,
# including its last frame's lower, decayed peak -- see this
# tutorial's own README for how this value was picked.
ISO_VALUE = 1.2
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
        plotter.add_mesh(surface, color="orangered", smooth_shading=True, specular=0.3)
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
