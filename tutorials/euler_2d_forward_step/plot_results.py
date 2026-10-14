#!/usr/bin/env python3
"""Renders figures for the forward-facing-step benchmark (task 0010)
from the VTK snapshots `forward_step_2d` writes under data/res*_<flux>/.

Usage:
    python3 -m venv .venv && source .venv/bin/activate
    pip install numpy matplotlib
    python3 plot_results.py

Writes figures/ under this directory:
  - density_t{0,1,2,4}.png     : density field at several times, finest
                                 resolution (res80, Rusanov) -- bow
                                 shock, growing Mach stem, trailing
                                 contact.
  - schlieren_t{1,2,4}.png     : |grad(density)| proxy ("numerical
                                 schlieren"), finest resolution
                                 (Rusanov) -- makes the shock/contact
                                 structure much more visible than raw
                                 density alone.
  - resolution_comparison.png : density at t~4 across all seven
                                 resolutions (Rusanov), for the
                                 convergence-illustration qualitative
                                 check.
  - contact_kh_detail.png     : zoomed trailing-contact schlieren,
                                 Rusanov, res80 vs res288.

Task 0011 (docs/adr/0012-hllc-numerical-flux.md) re-ran the whole sweep
with HLLC (data/res*_hllc/) specifically to check for the paper's
"mild waviness" Kelvin-Helmholtz signature on the trailing contact,
which Rusanov showed no trace of at any resolution -- adds:
  - contact_kh_rusanov_vs_hllc.png : zoomed trailing-contact schlieren,
                                 Rusanov vs HLLC, at res80 AND res288 --
                                 the direct comparison this task exists
                                 to produce.
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
    ax.set_xlim(extent1[0], extent2[1])
    ax.set_ylim(extent1[2], extent1[3])
    # The step itself (solid, not part of either block's real domain):
    # drawn from block2's OWN actual origin (where its real cells start in
    # x and y), not hardcoded (0.6, 0.2) -- at a resolution that doesn't
    # divide the nominal step position/height evenly (96, 144: see
    # ForwardStepBlocks's constructor comment on why), the real rounded
    # origin written into block2's VTK metadata differs from the nominal
    # constants by a sub-cell amount, so hardcoding the nominal values
    # here would draw the solid step slightly off from where the two
    # blocks' real cells actually meet.
    ax.fill_between([origin2[0], extent2[1]], extent1[2], origin2[1], color="0.25", zorder=5)
    return im


def schlieren_proxy(field, spacing):
    gy, gx = np.gradient(field, spacing[1], spacing[0])
    return np.sqrt(gx ** 2 + gy ** 2)


def schlieren_proxy_two_blocks(b1, b2):
    """Like `schlieren_proxy`, but for the two-block forward-step case:
    `np.gradient` called separately on each block's own array (as a naive
    per-block `schlieren_proxy(f1, s1)` / `schlieren_proxy(f2, s2)` would
    do) falls back to a one-sided difference at the two columns adjacent
    to the shared block interface, since each block's array has no
    knowledge of the other's real cells there. That one-sided estimate is
    measurably worse than a centered difference using the true neighbor
    value (which is exactly what the real simulation's own
    `BlockInterfaceBoundary` ghost-fill provides to the solver) --
    verified by comparison: at one case this produced schlieren values of
    106.6/32.0 at the two edge columns where the properly-stitched
    centered difference gives 70.9/34.8, a large enough discrepancy to
    show up as a spurious second line next to the real shock in the
    rendered image. The underlying density field is NOT discontinuous
    there; this was purely an artifact of computing each block's gradient
    in isolation. Fixes it by correcting just the two edge columns to use
    a true cross-block centered difference wherever the two blocks share
    height (the forward step's own riser-face region, where block2 does
    not exist at that height, is unaffected -- there is no neighbor to
    stitch in there, so those rows keep the same one-sided estimate
    `schlieren_proxy` already gives them, "respecting the solid step"
    rather than inventing a value across it).

    No smoothing anywhere: every interior column (all but the two edge
    columns, which is all `np.gradient`'s default `edge_order=1` ever
    treats differently to begin with) is untouched, and the two edge
    columns get a real neighbor CELL VALUE substituted into the same
    centered-difference formula, not a filtered/averaged value."""
    (f1, origin1, spacing1), (f2, origin2, spacing2) = b1, b2
    dx1, dy1 = spacing1[0], spacing1[1]
    dx2, dy2 = spacing2[0], spacing2[1]
    g1y, g1x = np.gradient(f1, dy1, dx1)
    g2y, g2x = np.gradient(f2, dy2, dx2)

    # Row j2 (block2's local row) shares height with block1's local row
    # j2 + step_rows -- same relationship `forward_step_setup.hpp`'s
    # `j_offset_to_neighbor` encodes for the real simulation's ghost fill,
    # re-derived here independently for the plotting script. Uses the
    # blocks' ACTUAL origins (now self-consistent after the
    # ForwardStepBlocks constructor fix), not the nominal step height
    # constant, so this stays correct at 96/144 too.
    step_rows = int(round((origin2[1] - origin1[1]) / dy1))
    ny2 = f2.shape[0]
    j1 = np.arange(ny2) + step_rows
    j2 = np.arange(ny2)
    g1x[j1, -1] = (f2[j2, 0] - f1[j1, -2]) / (2.0 * dx1)
    g2x[j2, 0] = (f2[j2, 1] - f1[j1, -1]) / (2.0 * dx2)

    schlieren1 = np.sqrt(g1x ** 2 + g1y ** 2)
    schlieren2 = np.sqrt(g2x ** 2 + g2y ** 2)
    return schlieren1, schlieren2


def frame_times(res_dir, snapshot_dt=0.1):
    n_frames = len(glob.glob(os.path.join(res_dir, "frame_*_density_block1.vtk")))
    return n_frames, [i * snapshot_dt for i in range(n_frames)]


def res_dir_for(res, flux):
    return os.path.join(DATA, f"res{res}_{flux}")


def main():
    res_dir_80 = res_dir_for(80, "rusanov")
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
        g1, g2 = schlieren_proxy_two_blocks((f1, o1, s1), (f2, o2, s2))
        im = plot_two_blocks(ax, (g1, o1, s1), (g2, o2, s2), cmap="gray_r", vmin=0, vmax=np.percentile(g1, 99))
        ax.set_title(f"|grad(density)| (numerical schlieren), t={times[idx]:.2f} (res80)")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.savefig(os.path.join(FIGURES, "schlieren_time_series.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/schlieren_time_series.png")

    # --- resolution comparison at final time --------------------------------
    resolutions = [20, 40, 80, 96, 144, 216, 288]
    fig, axes = plt.subplots(len(resolutions), 1, figsize=(9, 13), sharex=True)
    for ax, res in zip(axes, resolutions):
        res_dir = res_dir_for(res, "rusanov")
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

    # --- zoomed trailing-contact-discontinuity detail, res80 vs res288 ------
    # Specifically looking for Kelvin-Helmholtz rollup on the slip line,
    # which the README describes as only "mild waviness, not dramatic
    # rollup" at res80 -- check whether finer resolution reveals more.
    # Rusanov only (original task 0010 figure, unchanged) -- see the
    # Rusanov-vs-HLLC comparison figure below for task 0011's own check.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, res in zip(axes, [80, 288]):
        res_dir = res_dir_for(res, "rusanov")
        nf, ts = frame_times(res_dir)
        idx = nf - 1
        (f1, o1, s1), (f2, o2, s2) = load_frame(res_dir, idx, "density")
        g1, g2 = schlieren_proxy_two_blocks((f1, o1, s1), (f2, o2, s2))
        im = plot_two_blocks(ax, (g1, o1, s1), (g2, o2, s2), cmap="gray_r", vmin=0, vmax=np.percentile(g1, 99))
        ax.set_xlim(1.0, 2.2)
        ax.set_ylim(0.1, 0.7)
        ax.set_title(f"res{res}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(f"Trailing contact discontinuity, zoomed, t={ts[idx]:.2f} -- res80 vs res288 (Rusanov)")
    fig.savefig(os.path.join(FIGURES, "contact_kh_detail.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/contact_kh_detail.png")

    # --- Rusanov vs. HLLC, same zoom, same resolutions (task 0011) ---------
    # The direct comparison this task exists to produce: does switching to
    # HLLC (sharper contact resolution, per the Sod-tube evidence) reveal
    # the paper's "mild waviness... visible at times 2.5 and 3" that
    # Rusanov shows no trace of at any resolution?
    #
    # vmax is computed from BLOCK 2's own gradient at a moderate (75th)
    # percentile, not the original figure's `np.percentile(g1, 99)` --
    # the bow shock (always the single largest gradient in the domain)
    # otherwise dominates the color scale and washes out the much
    # weaker trailing-contact structure this figure exists to show.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    vmax_global = None
    panels = []
    for row, flux in enumerate(["rusanov", "hllc"]):
        for col, res in enumerate([80, 288]):
            res_dir = res_dir_for(res, flux)
            nf, ts = frame_times(res_dir)
            idx = nf - 1
            (f1, o1, s1), (f2, o2, s2) = load_frame(res_dir, idx, "density")
            g1, g2 = schlieren_proxy_two_blocks((f1, o1, s1), (f2, o2, s2))
            panels.append((row, col, flux, res, (g1, o1, s1), (g2, o2, s2), ts[idx]))
            if vmax_global is None:
                vmax_global = np.percentile(g2, 90)
    for row, col, flux, res, b1g, b2g, t_last in panels:
        ax = axes[row, col]
        im = plot_two_blocks(ax, b1g, b2g, cmap="gray_r", vmin=0, vmax=vmax_global)
        ax.set_xlim(0.9, 2.4)
        ax.set_ylim(0.15, 0.45)
        ax.set_title(f"{flux}, res{res}")
        if row == 1:
            ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("y")
    fig.suptitle(f"Trailing contact discontinuity: Rusanov vs. HLLC, t={t_last:.2f} (task 0011)")
    fig.savefig(os.path.join(FIGURES, "contact_kh_rusanov_vs_hllc.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/contact_kh_rusanov_vs_hllc.png")

    # --- HLLC time evolution at the paper's own named times -----------------
    # Woodward & Colella report the mild KH "water-wave" instability
    # "visible at times 2.5 and 3" -- show the HLLC trailing-contact
    # region at exactly those times (plus 1.5/2.0 for before/onset
    # context and 4.0 for the final state), at the finest resolution
    # (res288), to let a reader see the SAME growth-over-time signature
    # directly rather than only a single final-time snapshot.
    res_dir_288_hllc = res_dir_for(288, "hllc")
    nf288, ts288 = frame_times(res_dir_288_hllc)
    target_times = [1.5, 2.0, 2.5, 3.0, 4.0]
    fig, axes = plt.subplots(len(target_times), 1, figsize=(12, 15), sharex=True)
    vmax_time = None
    for ax, tt in zip(axes, target_times):
        idx = min(range(nf288), key=lambda i: abs(ts288[i] - tt))
        (f1, o1, s1), (f2, o2, s2) = load_frame(res_dir_288_hllc, idx, "density")
        g1, g2 = schlieren_proxy_two_blocks((f1, o1, s1), (f2, o2, s2))
        if vmax_time is None:
            vmax_time = np.percentile(g2, 90)
        im = plot_two_blocks(ax, (g1, o1, s1), (g2, o2, s2), cmap="gray_r", vmin=0, vmax=vmax_time)
        ax.set_xlim(0.9, 2.4)
        ax.set_ylim(0.15, 0.45)
        ax.set_title(f"HLLC res288, t={ts288[idx]:.2f}")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.suptitle("HLLC trailing-contact instability: growth over time (res288)")
    fig.savefig(os.path.join(FIGURES, "contact_kh_hllc_time_evolution.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/contact_kh_hllc_time_evolution.png")


if __name__ == "__main__":
    main()
