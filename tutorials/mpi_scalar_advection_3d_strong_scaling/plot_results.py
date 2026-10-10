#!/usr/bin/env python3
"""Strong-scaling plot for the 3D MPI scalar-advection tutorial. Reads
data/summary.csv (one row per rank count, assembled by running the
tutorial once per rank count -- see README.md) and plots wall-clock
time vs. rank count against an ideal-strong-scaling reference line,
plus a speedup/parallel-efficiency plot.

Dependencies: numpy, pandas, matplotlib (`pip install numpy pandas
matplotlib`).

Usage: run from this directory, after assembling data/summary.csv:

    python3 plot_results.py

Writes figures/strong_scaling_time.png, figures/strong_scaling_speedup.png,
figures/expected_scaling_reference.png.
"""
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent


def plot_expected_scaling_reference():
    """A purely conceptual reference plot -- NOT derived from
    data/summary.csv, and does not require the tutorial to have been
    run at all. Illustrates what "good" vs. "overhead-limited" strong
    scaling looks like in general, using a standard simple model:
    efficiency(N) = 1 / (1 + alpha*(N-1)), where `alpha` is the fraction
    of each rank's work that is communication/overhead rather than
    useful computation (alpha=0 is the unreachable ideal; larger alpha
    means overhead eats into the benefit of more ranks faster). This is
    the same qualitative shape task 0004's `bench_mpi_halo_exchange`
    and this tutorial's own measured curves both follow -- see
    README.md's "What the numbers show" section for the actual
    measured data this reference plot is meant to help interpret.
    """
    ranks = np.linspace(1, 64, 400)
    alphas = {
        "ideal (no overhead)": 0.0,
        "low overhead (alpha=0.01)": 0.01,
        "moderate overhead (alpha=0.05)": 0.05,
        "high overhead (alpha=0.20)": 0.20,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for label, alpha in alphas.items():
        efficiency = 1.0 / (1.0 + alpha * (ranks - 1.0))
        speedup = ranks * efficiency
        style = "--" if alpha == 0.0 else "-"
        ax1.plot(ranks, speedup, style, label=label)
        ax2.plot(ranks, efficiency * 100, style, label=label)

    ax1.set_xlabel("MPI ranks")
    ax1.set_ylabel("speedup (T(1)/T(N))")
    ax1.set_title("Expected strong scaling: speedup")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("parallel efficiency (%)")
    ax2.set_title("Expected strong scaling: efficiency")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=8)

    fig.suptitle("What to expect from strong scaling (illustrative, not measured)")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "expected_scaling_reference.png", dpi=150)
    plt.close(fig)
    print("Wrote figures/expected_scaling_reference.png (conceptual reference, no data needed)")


def main():
    # Always produced, independent of any measured data -- this is the
    # one figure a reader can look at before ever running the tutorial.
    plot_expected_scaling_reference()

    summary_path = HERE / "data" / "summary.csv"
    if not summary_path.exists():
        print(f"{summary_path} not found yet -- run the tutorial first (see README.md's "
              '"Build and run" section) to generate the measured strong-scaling plots too.')
        return

    df = pd.read_csv(summary_path).sort_values("ranks")
    ranks = df["ranks"].to_numpy()
    time_s = df["wall_clock_s"].to_numpy()

    t1 = time_s[ranks == 1][0] if 1 in ranks else time_s[0] * ranks[0]
    ideal_ranks = np.linspace(ranks.min(), ranks.max(), 200)
    ideal_time = t1 / ideal_ranks

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ranks, time_s, "o-", label="measured")
    ax.loglog(ideal_ranks, ideal_time, "--", color="gray", label="ideal (T(1)/ranks)")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("3D scalar advection: strong scaling (time)")
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "strong_scaling_time.png", dpi=150)
    plt.close(fig)

    speedup = t1 / time_s
    efficiency = speedup / ranks

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.plot(ranks, speedup, "o-", label="measured")
    ax1.plot(ranks, ranks, "--", color="gray", label="ideal")
    ax1.set_xlabel("MPI ranks")
    ax1.set_ylabel("speedup (T(1)/T(N))")
    ax1.set_title("Speedup")
    ax1.set_xticks(ranks)
    ax1.legend()

    ax2.plot(ranks, efficiency * 100, "o-", color="darkorange")
    ax2.axhline(100, linestyle="--", color="gray")
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("parallel efficiency (%)")
    ax2.set_title("Efficiency")
    ax2.set_xticks(ranks)
    ax2.set_ylim(0, max(110, efficiency.max() * 110))

    fig.tight_layout()
    fig.savefig(HERE / "figures" / "strong_scaling_speedup.png", dpi=150)
    plt.close(fig)

    print("Wrote figures/strong_scaling_time.png and figures/strong_scaling_speedup.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
