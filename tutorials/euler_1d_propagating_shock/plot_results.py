#!/usr/bin/env python3
"""Plot the propagating-shock tutorial's output (data/*.csv -> figures/*.png).

Run after the `cfe_euler_1d_propagating_shock` executable (see this
directory's README.md). Requires matplotlib + pandas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data"
FIGS = "figures"

summary = pd.read_csv(f"{DATA}/summary.csv")
tracks = pd.read_csv(f"{DATA}/tracked_positions.csv")
analytic_speed = float(summary["analytic_speed"].iloc[0])
finest_nx = int(summary["nx"].max())

# --- Figure 1: tracked shock position vs. time, with the linear fit,
# at the finest resolution, both fluxes. -------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
for flux, marker, color in [("rusanov", "o", "C0"), ("hllc", "s", "C1")]:
    rows = tracks[(tracks["flux"] == flux) & (tracks["nx"] == finest_nx)].sort_values("t")
    ax.plot(rows["t"], rows["x_shock"], marker, color=color, label=f"tracked ({flux})")
    speed = float(summary.loc[(summary["flux"] == flux) & (summary["nx"] == finest_nx), "measured_speed"].iloc[0])
    t_fit = np.array([rows["t"].min(), rows["t"].max()])
    x0_fit = rows["x_shock"].iloc[0] - speed * rows["t"].iloc[0]
    ax.plot(t_fit, x0_fit + speed * t_fit, "--", color=color, lw=1, label=f"fit ({flux}), v={speed:.4f}")
ax.set_xlabel("t")
ax.set_ylabel("Tracked shock position x")
ax.set_title(f"Propagating-shock position vs. time (nx={finest_nx})\nanalytic speed={analytic_speed:.4f}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGS}/propagating_shock_position_vs_time.png", dpi=150)

# --- Figure 2: measured-speed error vs. resolution, both fluxes ------------
fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
for flux, marker, color in [("rusanov", "o", "C0"), ("hllc", "s", "C1")]:
    rows = summary[summary["flux"] == flux].sort_values("nx")
    ax2.loglog(rows["nx"], rows["abs_error"], f"{marker}-", color=color, label=flux)
ax2.set_xlabel("nx")
ax2.set_ylabel("|measured speed - analytic speed|")
ax2.set_title("Propagation-speed error vs. resolution")
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{FIGS}/propagating_shock_speed_error.png", dpi=150)

print("Wrote figures/propagating_shock_position_vs_time.png, propagating_shock_speed_error.png")
