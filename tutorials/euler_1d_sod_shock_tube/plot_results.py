#!/usr/bin/env python3
"""Plot the Sod shock tube tutorial's output (data/*.csv -> figures/*.png).

Run after the `cfe_euler_1d_sod_shock_tube` executable (see this
directory's README.md). Requires matplotlib + pandas, like the Burgers
tutorials' own plot scripts.
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data"
FIGS = "figures"

summary = pd.read_csv(f"{DATA}/summary.csv")

# --- Figure 1: profiles at the finest resolution vs. exact -----------------
finest_nx = int(summary["nx"].max())
profile = pd.read_csv(f"{DATA}/case_nx{finest_nx}.csv")

fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
for ax, (num_col, exact_col, label) in zip(
    axes,
    [
        ("rho_numerical", "rho_exact", "Density"),
        ("u_numerical", "u_exact", "Velocity"),
        ("p_numerical", "p_exact", "Pressure"),
    ],
):
    ax.plot(profile["x"], profile[exact_col], "k-", lw=1.5, label="exact")
    ax.plot(profile["x"], profile[num_col], "C0.", ms=3, label=f"MUSCL+Rusanov (nx={finest_nx})")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8)
axes[-1].set_xlabel("x")
fig.suptitle(f"Sod shock tube at t=0.2 (nx={finest_nx})")
fig.tight_layout()
fig.savefig(f"{FIGS}/sod_profiles.png", dpi=150)

# --- Figure 2: L1 error vs. resolution --------------------------------------
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.loglog(summary["nx"], summary["l1_rho"], "o-", label="L1(rho)")
ax2.loglog(summary["nx"], summary["l1_u"], "s-", label="L1(u)")
ax2.loglog(summary["nx"], summary["l1_p"], "^-", label="L1(p)")
ax2.set_xlabel("nx")
ax2.set_ylabel("L1 error vs. exact solution")
ax2.set_title("Sod shock tube: error vs. resolution\n(discontinuities cap the rate below 2nd order)")
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{FIGS}/sod_l1_convergence.png", dpi=150)

print("Wrote figures/sod_profiles.png and figures/sod_l1_convergence.png")
