#!/usr/bin/env python3
"""Plots for the 2D Burgers diagonal-shock tutorial. Reads
data/summary.csv plus the representative field/diagonal-profile CSVs
burgers_2d writes for the smallest grid (100^2, chosen to keep committed
file size down -- see README.md) and second-order (limited)
reconstruction.

Dependencies: numpy, pandas, matplotlib (`pip install numpy pandas
matplotlib`).

Usage: run from this directory, after running the C++ tutorial at least
once (so data/ is populated):

    python3 plot_results.py

Writes figures/colormaps.png, figures/diagonal_profiles.png,
figures/convergence.png.
"""
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

GRID = 100
RECONSTRUCTION = "second_order_limited"
TIMES = [0.00, 0.25, 0.50]


def load_field(t):
    path = DATA / f"field_nx{GRID:04d}_{RECONSTRUCTION}_t{t:.2f}.csv"
    df = pd.read_csv(path)
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    num = df.pivot(index="y", columns="x", values="u_numerical").loc[ys, xs].to_numpy()
    return xs, ys, num


def plot_colormaps():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, t in zip(axes, TIMES):
        xs, ys, num = load_field(t)
        mesh = ax.pcolormesh(xs, ys, num, vmin=0.0, vmax=1.0, cmap="viridis", shading="nearest")
        # Exact shock line: x + y = 0.5 + t.
        line_x = np.array([0.0, 1.0])
        line_y = (0.5 + t) - line_x
        ax.plot(line_x, line_y, color="red", linewidth=1.5, linestyle="--", label="exact shock: x+y=0.5+t")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=axes, label="u (numerical)", shrink=0.8)
    fig.suptitle(f"2D diagonal shock: numerical field vs. exact shock line (nx=ny={GRID}, limited 2nd order)")
    fig.savefig(FIGURES / "colormaps.png", dpi=150)
    plt.close(fig)


def plot_diagonal_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, t in zip(axes, TIMES):
        path = DATA / f"diagonal_nx{GRID:04d}_{RECONSTRUCTION}_t{t:.2f}.csv"
        df = pd.read_csv(path)
        ax.plot(df["s"], df["u_exact"], color="black", linewidth=1.5, label="exact (cell average)")
        ax.plot(df["s"], df["u_numerical"], color="tab:red", marker=".", markersize=3, linewidth=1.0,
                 label="numerical")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("s (along x=y)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Profile along the diagonal x=y (nx=ny={GRID}, limited 2nd order)")
    fig.tight_layout()
    fig.savefig(FIGURES / "diagonal_profiles.png", dpi=150)
    plt.close(fig)


def plot_convergence():
    df = pd.read_csv(DATA / "summary.csv")
    df = df[df["time"] == TIMES[-1]].copy()
    df["dx"] = 1.0 / df["grid"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, label, color in [
        ("first_order", "first-order (unlimited)", "tab:blue"),
        ("second_order_limited", "limited second-order (minmod)", "tab:red"),
    ]:
        sub = df[df["reconstruction"] == name].sort_values("dx")
        ax.loglog(sub["dx"], sub["l1_error"], marker="o", color=color, label=label)

    ref_dx = np.array(sorted(df["dx"].unique()))
    first_order_coarse = df[(df["reconstruction"] == "first_order") & (df["dx"] == ref_dx.max())][
        "l1_error"
    ].iloc[0]
    ref_line = first_order_coarse * (ref_dx / ref_dx.max())
    ax.loglog(ref_dx, ref_line, linestyle="--", color="gray", label="O(dx) reference")

    ax.set_xlabel("grid spacing dx")
    ax.set_ylabel(f"L1 error vs. exact cell average (t={TIMES[-1]:.2f})")
    ax.set_title("2D diagonal shock: L1 error vs. grid spacing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    plot_colormaps()
    plot_diagonal_profiles()
    plot_convergence()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
