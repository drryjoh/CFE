#!/usr/bin/env python3
"""Plots for the 1D Burgers tutorial (case A: moving shock, case B:
sinusoidal steepening). Reads data/summary.csv plus the representative
per-time field CSVs (x, u_numerical, u_exact) burgers_1d writes for the
finest grid (400 cells) and second-order (limited) reconstruction --
see README.md for exactly which files are committed vs. regenerated.

Dependencies: numpy, pandas, matplotlib (`pip install numpy pandas
matplotlib`).

Usage: run from this directory, after running the C++ tutorial at least
once (so data/ is populated):

    python3 plot_results.py

Writes figures/case_a_profiles.png, figures/case_a_convergence.png,
figures/case_a_shock_zoom.png, figures/case_b_profiles.png.
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


def plot_case_a_profiles():
    times = [0.00, 0.25, 0.50, 1.00]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, times):
        path = DATA / f"case_shock_nx0400_second_order_limited_t{t:.2f}.csv"
        df = pd.read_csv(path)
        ax.plot(df["x"], df["u_exact"], color="black", linewidth=1.5, label="exact (cell average)")
        ax.plot(df["x"], df["u_numerical"], color="tab:red", linewidth=1.0, marker=".", markersize=2,
                 label="numerical (limited 2nd order, nx=400)")
        ax.set_title(f"t = {t:.2f}")
        ax.grid(alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("u")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Case A: moving shock -- numerical vs. exact Rankine-Hugoniot solution")
    fig.tight_layout()
    fig.savefig(FIGURES / "case_a_profiles.png", dpi=150)
    plt.close(fig)


def plot_case_a_convergence():
    df = pd.read_csv(DATA / "summary.csv")
    df = df[(df["case"] == "shock") & (df["time"] == 1.00)].copy()
    df["dx"] = 1.0 / df["grid"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, label, color in [
        ("first_order", "first-order (unlimited)", "tab:blue"),
        ("second_order_limited", "limited second-order (minmod)", "tab:red"),
    ]:
        sub = df[df["reconstruction"] == name].sort_values("dx")
        ax.loglog(sub["dx"], sub["l1_error"], marker="o", color=color, label=label)

    # A reference O(dx) line anchored at the first-order scheme's coarsest
    # point -- a captured shock is expected to converge at first order
    # regardless of the reconstruction's away-from-shock accuracy (see
    # tutorial README / docs/adr/0008-burgers-shock-capturing-scheme.md).
    ref_dx = np.array(sorted(df["dx"].unique()))
    first_order_coarse = df[(df["reconstruction"] == "first_order") & (df["dx"] == ref_dx.max())][
        "l1_error"
    ].iloc[0]
    ref_line = first_order_coarse * (ref_dx / ref_dx.max())
    ax.loglog(ref_dx, ref_line, linestyle="--", color="gray", label="O(dx) reference")

    ax.set_xlabel("grid spacing dx")
    ax.set_ylabel("L1 error vs. exact cell average (t=1.0)")
    ax.set_title("Case A: L1 error vs. grid spacing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "case_a_convergence.png", dpi=150)
    plt.close(fig)


def plot_case_a_shock_zoom():
    """Zooms in on the captured front at the final time (t=1.0, the
    most-developed state), re-centering each reconstruction on ITS OWN
    numerically-detected shock location (summary.csv's
    shock_position_numerical -- the same value burgers_1d.cpp reports,
    not recomputed here) so x=0 is "the shock" for that scheme and the
    fixed +/-0.05 window is directly comparable between schemes despite
    their slightly different actual shock positions. This is the plot
    that actually shows what "shock capturing" costs: how many cells
    wide the smeared transition is, not just that the overall L1 error
    is small.
    """
    summary = pd.read_csv(DATA / "summary.csv")
    t = 1.00
    half_window = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (name, label) in zip(
        axes,
        [("first_order", "first-order (unlimited)"), ("second_order_limited", "limited second-order (minmod)")],
    ):
        path = DATA / f"case_shock_nx0400_{name}_t{t:.2f}.csv"
        df = pd.read_csv(path)
        shock_x = summary[
            (summary["case"] == "shock") & (summary["grid"] == 400) & (summary["reconstruction"] == name)
            & (summary["time"] == t)
        ]["shock_position_numerical"].iloc[0]

        shifted_x = df["x"] - shock_x
        mask = (shifted_x >= -half_window) & (shifted_x <= half_window)

        ax.plot(shifted_x[mask], df["u_exact"][mask], color="black", linewidth=1.2, label="exact (step at x=0)")
        ax.step(shifted_x[mask], df["u_numerical"][mask], where="mid", color="tab:red", linewidth=1.2,
                marker="o", markersize=4, label="numerical (per-cell value)")
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(-half_window, half_window)
        ax.set_title(f"{label}\n(nx=400, t={t:.2f}, shock at x={shock_x:.4f})")
        ax.set_xlabel("x - shock_position_numerical")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Case A: zoom on the captured front -- how many cells wide is the smeared shock?")
    fig.tight_layout()
    fig.savefig(FIGURES / "case_a_shock_zoom.png", dpi=150)
    plt.close(fig)


def plot_case_b_profiles():
    times = [0.00, 0.15, 0.30, 0.40, 0.60]
    break_time = 1.0 / (2.0 * np.pi * 0.5)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2), sharey=True)
    for ax, t in zip(axes, times):
        path = DATA / f"case_steepening_t{t:.2f}.csv"
        df = pd.read_csv(path)
        pre_shock = t < break_time
        ax.plot(df["x"], df["u_numerical"], color="tab:red", linewidth=1.2, label="numerical")
        if pre_shock:
            ax.plot(df["x"], df["u_exact"], color="black", linestyle="--", linewidth=1.0,
                     label="exact (characteristics)")
            ax.set_title(f"t={t:.2f}\n(pre-shock)")
        else:
            ax.set_title(f"t={t:.2f}\n(post-shock -- numerical only)")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Case B: sinusoidal steepening (breaking time t_s={break_time:.5f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "case_b_profiles.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    plot_case_a_profiles()
    plot_case_a_convergence()
    plot_case_a_shock_zoom()
    plot_case_b_profiles()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
