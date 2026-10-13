#!/usr/bin/env python3
"""Renders the "see it work" figure for the 3D CPU extrusion
(task 0010): a mid-span Z-slice of the final-time density field,
compared directly against the 2D primary run at the same resolution
(cells-per-unit-height=40) -- confirming visually, not just via the
printed max|mom_z|=0 diagnostic, that the extrusion stays exactly 2D.

Usage:
    python3 -m venv .venv && source .venv/bin/activate
    pip install numpy matplotlib
    python3 plot_results.py
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGURES = os.path.join(HERE, "figures")
TWIN_2D_DATA = os.path.join(HERE, "..", "euler_2d_forward_step", "data", "res40")
os.makedirs(FIGURES, exist_ok=True)


def read_vtk(path):
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
    return values.reshape((nz, ny, nx)), origin, spacing


def plot_two_blocks(ax, b1, b2, **kwargs):
    (f1, o1, s1), (f2, o2, s2) = b1, b2
    ny1, nx1 = f1.shape
    ny2, nx2 = f2.shape
    e1 = [o1[0], o1[0] + nx1 * s1[0], o1[1], o1[1] + ny1 * s1[1]]
    e2 = [o2[0], o2[0] + nx2 * s2[0], o2[1], o2[1] + ny2 * s2[1]]
    im = ax.imshow(f1, origin="lower", extent=e1, aspect="equal", **kwargs)
    ax.imshow(f2, origin="lower", extent=e2, aspect="equal", **kwargs)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.0)
    ax.fill_between([0.6, 3.0], 0, 0.2, color="0.25", zorder=5)
    return im


def last_frame_index(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    return len(files) - 1


def main():
    idx3d = last_frame_index(DATA, "frame_*_density_block1.vtk")
    f1_3d, o1, s1 = read_vtk(os.path.join(DATA, f"frame_{idx3d:04d}_density_block1.vtk"))
    f2_3d, o2, s2 = read_vtk(os.path.join(DATA, f"frame_{idx3d:04d}_density_block2.vtk"))
    mid1, mid2 = f1_3d.shape[0] // 2, f2_3d.shape[0] // 2

    idx2d = last_frame_index(TWIN_2D_DATA, "frame_*_density_block1.vtk")
    f1_2d, _, _ = read_vtk(os.path.join(TWIN_2D_DATA, f"frame_{idx2d:04d}_density_block1.vtk"))
    f2_2d, _, _ = read_vtk(os.path.join(TWIN_2D_DATA, f"frame_{idx2d:04d}_density_block2.vtk"))

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    im = plot_two_blocks(axes[0], (f1_3d[mid1], o1, s1), (f2_3d[mid2], o2, s2), cmap="viridis", vmin=1.0, vmax=6.5)
    axes[0].set_title("3D CPU extrusion, mid-span Z slice (cells-per-unit-height=40, nz=8)")
    plot_two_blocks(axes[1], (f1_2d[0], o1, s1), (f2_2d[0], o2, s2), cmap="viridis", vmin=1.0, vmax=6.5)
    axes[1].set_title("2D primary run, same resolution (res40) -- for comparison")
    for ax in axes:
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.colorbar(im, ax=axes, shrink=0.8, label="density")
    fig.savefig(os.path.join(FIGURES, "3d_vs_2d_final_slice.png"), dpi=130, bbox_inches="tight")
    print("wrote figures/3d_vs_2d_final_slice.png")

    # Max abs difference between the mid-Z 3D slice and the 2D run, both
    # blocks -- a quantitative version of the same "stays exactly 2D"
    # check, independent of the solver's own printed max|mom_z|.
    diff1 = np.max(np.abs(f1_3d[mid1] - f1_2d[0]))
    diff2 = np.max(np.abs(f2_3d[mid2] - f2_2d[0]))
    print(f"max|density_3d_slice - density_2d| : block1={diff1:.3e}, block2={diff2:.3e}")


if __name__ == "__main__":
    main()
