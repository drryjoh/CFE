#!/usr/bin/env python3
"""Renders figures for the forward-facing-step benchmark (task 0010)
from the VTK snapshots `forward_step_2d` writes under data/res*/.

Usage:
    python3 -m venv .venv && source .venv/bin/activate
    pip install numpy matplotlib
    python3 plot_results.py

Writes figures/ under this directory:
  - density_t{0,1,2,4}.png     : density field at several times, finest
                                 resolution (res80) -- bow shock, growing
                                 Mach stem, trailing contact.
  - schlieren_t{1,2,4}.png     : |grad(density)| proxy ("numerical
                                 schlieren"), finest resolution -- makes
                                 the shock/contact structure much more
                                 visible than raw density alone.
  - resolution_comparison.png : density at t~4 across all three
                                 resolutions, for the convergence-
                                 illustration qualitative check.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)


def read_vtk_structured_points_cell_scalar(path):
    """Minimal reader for cfe::io::write_vtk_structured_points_cell_scalar's
    ASCII legacy STRUCTURED_POINTS/CELL_DATA output."""
    with open(path, "r") as f:
        lines = f.read().split("\n")
    dims = origin = spacing = None
    values_start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in line.split()[1:4])
        elif line.startswith("ORIGIN"):
            origin = tuple(float(x) for x in line.split()[1:4])
        elif line.startswith("SPACING"):
            spacing = tuple(float(x) for x in line.split()[1:4])
        elif line.startswith("LOOKUP_TABLE"):
            values_start = i + 1
            break
    nx, ny, nz = dims[0] - 1, dims[1] - 1, dims[2] - 1
    n = nx * ny * nz
    values = np.array([float(v) for v in lines[values_start:values_start + n]], dtype=float)
    # k-slowest, j-next, i-fastest (matches CartesianGrid::flat_index order)
    field = values.reshape((nz, ny, nx))
    return field[0], origin, spacing  # 2D only here: take the single z-slice


def load_frame(res_dir, frame_index, quantity):
    b1 = read_vtk_structured_points_cell_scalar(
        os.path.join(res_dir, f"frame_{frame_index:04d}_{quantity}_block1.vtk"))
    b2 = read_vtk_structured_points_cell_scalar(
        os.path.join(res_dir, f"frame_{frame_index:04d}_{quantity}_block2.vtk"))
    return b1, b2


def plot_two_blocks(ax, b1, b2, **imshow_kwargs):
    (f1, origin1, spacing1), (f2, origin2, spacing2) = b1, b2
    ny1, nx1 = f1.shape
    ny2, nx2 = f2.shape
    extent1 = [origin1[0], origin1[0] + nx1 * spacing1[0], origin1[1], origin1[1] + ny1 * spacing1[1]]
    extent2 = [origin2[0], origin2[0] + nx2 * spacing2[0], origin2[1], origin2[1] + ny2 * spacing2[1]]
    im = ax.imshow(f1, origin="lower", extent=extent1, aspect="equal", **imshow_kwargs)
    ax.imshow(f2, origin="lower", extent=extent2, aspect="equal", **imshow_kwargs)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.0)
    # The step itself (solid, not part of either block's real domain):
    ax.fill_between([0.6, 3.0], 0, 0.2, color="0.25", zorder=5)
    return im


def schlieren_proxy(field, spacing):
    gy, gx = np.gradient(field, spacing[1], spacing[0])
    return np.sqrt(gx ** 2 + gy ** 2)


def frame_times(res_dir, snapshot_dt=0.1):
    n_frames = len(glob.glob(os.path.join(res_dir, "frame_*_density_block1.vtk")))
    return n_frames, [i * snapshot_dt for i in range(n_frames)]


def main():
    res_dir_80 = os.path.join(DATA, "res80")
    n_frames, times = frame_times(res_dir_80)
    print(f"res80: {n_frames} frames, t up to {times[-1]:.2f}")

    # --- density field at several times, finest resolution -----------------
    target_times = [0.0, 1.0, 2.0, times[-1]]
    fig, axes = plt.subplots(len(target_times), 1, figsize=(9, 10), sharex=True)
    for ax, tt in zip(axes, target_times):
        idx = min(range(n_frames), key=lambda i: abs(times[i] - tt))
        b1, b2 = load_frame(res_dir_80, idx, "density")
        im = plot_two_blocks(ax, b1, b2, cmap="viridis", vmin=1.0, vmax=6.5)
        ax.set_title(f"density, t={times[idx]:.2f} (res80)")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.colorbar(im, ax=axes, shrink=0.8, label="density")
    fig.savefig(os.path.join(FIGURES, "density_time_series.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/density_time_series.png")

    # --- numerical schlieren at a few times ---------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, tt in zip(axes, [1.0, 2.0, times[-1]]):
        idx = min(range(n_frames), key=lambda i: abs(times[i] - tt))
        (f1, o1, s1), (f2, o2, s2) = load_frame(res_dir_80, idx, "density")
        g1 = schlieren_proxy(f1, s1)
        g2 = schlieren_proxy(f2, s2)
        im = plot_two_blocks(ax, (g1, o1, s1), (g2, o2, s2), cmap="gray_r", vmin=0, vmax=np.percentile(g1, 99))
        ax.set_title(f"|grad(density)| (numerical schlieren), t={times[idx]:.2f} (res80)")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.savefig(os.path.join(FIGURES, "schlieren_time_series.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/schlieren_time_series.png")

    # --- resolution comparison at final time --------------------------------
    resolutions = [20, 40, 80, 96, 144]
    fig, axes = plt.subplots(len(resolutions), 1, figsize=(9, 13), sharex=True)
    for ax, res in zip(axes, resolutions):
        res_dir = os.path.join(DATA, f"res{res}")
        nf, ts = frame_times(res_dir)
        idx = nf - 1
        b1, b2 = load_frame(res_dir, idx, "density")
        im = plot_two_blocks(ax, b1, b2, cmap="viridis", vmin=1.0, vmax=6.5)
        ax.set_title(f"density, t={ts[idx]:.2f}, res{res} ({b1[0].shape[1]}x{b1[0].shape[0]} / "
                     f"{b2[0].shape[1]}x{b2[0].shape[0]} cells)")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.colorbar(im, ax=axes, shrink=0.8, label="density")
    fig.savefig(os.path.join(FIGURES, "resolution_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/resolution_comparison.png")


if __name__ == "__main__":
    main()
