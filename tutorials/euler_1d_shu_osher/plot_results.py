#!/usr/bin/env python3
"""Plot the Shu-Osher tutorial's output (data/*.csv -> figures/*.png).

Run after the `cfe_euler_1d_shu_osher` executable (see this directory's
README.md). Requires matplotlib + pandas, like the Sod/Burgers
tutorials' own plot scripts.
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data"
FIGS = "figures"

summary = pd.read_csv(f"{DATA}/summary.csv")
finest_nx = int(summary["nx"].max())

profile_rusanov = pd.read_csv(f"{DATA}/case_nx{finest_nx}_rusanov.csv")
profile_hllc = pd.read_csv(f"{DATA}/case_nx{finest_nx}_hllc.csv")
fine_rusanov = pd.read_csv(f"{DATA}/case_fine_rusanov.csv")
fine_hllc = pd.read_csv(f"{DATA}/case_fine_hllc.csv")

# --- Figure 1: full density profile at the finest test resolution vs. the
# converged fine reference, both fluxes. ------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(fine_hllc["x"], fine_hllc["rho"], "k-", lw=1, alpha=0.6, label=f"HLLC reference (nx={len(fine_hllc)})")
ax.plot(profile_rusanov["x"], profile_rusanov["rho"], "C0.", ms=3, label=f"MUSCL+Rusanov (nx={finest_nx})")
ax.plot(profile_hllc["x"], profile_hllc["rho"], "C1x", ms=4, label=f"MUSCL+HLLC (nx={finest_nx})")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title(f"Shu-Osher density profile at t=1.8 (nx={finest_nx})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGS}/shu_osher_profiles.png", dpi=150)

# --- Figure 2: zoomed-in post-shock oscillatory wake -----------------------
front = summary.attrs.get("front", None)
# Estimate the front visually from the fine HLLC reference: rightmost x
# where density still deviates from 1+0.2*sin(5x) by more than 0.1 (same
# definition tests/verification/test_euler_shu_osher.cpp uses).
import numpy as np

exact_ahead = 1.0 + 0.2 * np.sin(5.0 * fine_hllc["x"])
disturbed = (fine_hllc["rho"] - exact_ahead).abs() > 0.1
front_est = fine_hllc.loc[disturbed, "x"].max() if disturbed.any() else fine_hllc["x"].max()

window_lo, window_hi = front_est - 3.0, front_est - 0.5
mask_fine = (fine_hllc["x"] >= window_lo) & (fine_hllc["x"] <= window_hi)
mask_r = (profile_rusanov["x"] >= window_lo) & (profile_rusanov["x"] <= window_hi)
mask_h = (profile_hllc["x"] >= window_lo) & (profile_hllc["x"] <= window_hi)

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(fine_hllc.loc[mask_fine, "x"], fine_hllc.loc[mask_fine, "rho"], "k-", lw=1, alpha=0.6,
          label="HLLC reference")
ax2.plot(profile_rusanov.loc[mask_r, "x"], profile_rusanov.loc[mask_r, "rho"], "C0.-", ms=3,
          label=f"MUSCL+Rusanov (nx={finest_nx})")
ax2.plot(profile_hllc.loc[mask_h, "x"], profile_hllc.loc[mask_h, "rho"], "C1x-", ms=4,
          label=f"MUSCL+HLLC (nx={finest_nx})")
ax2.set_xlabel("x")
ax2.set_ylabel("Density")
ax2.set_title("Post-shock oscillatory wake close-up")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{FIGS}/shu_osher_wake_closeup.png", dpi=150)

# --- Figure 3: ahead-of-shock region, showing Rusanov's diffusion vs.
# HLLC's exact preservation -- the full window (bulk + boundary-adjacent)
# is plotted deliberately, with a dashed line marking where the bulk
# window ends, so the plot is honest about BOTH error sources visible in
# it rather than only the one the summary table's "bulk" numbers isolate
# (Ilia's independent final review on this task: an earlier version of
# this figure/caption attributed the sharp divergence visible right at
# the domain edge to the same bulk mechanism visible through the rest of
# the window, when it is actually a distinct, boundary-localized
# artifact -- see shu_osher_max_ahead_deviation's own header comment).
# -----------------------------------------------------------------------
ahead_lo = front_est + 1.0
domain_x_max = 5.0
boundary_margin = 0.5
bulk_x_hi = domain_x_max - boundary_margin
mask_ahead_r = profile_rusanov["x"] >= ahead_lo
mask_ahead_h = profile_hllc["x"] >= ahead_lo

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
x_ahead = profile_rusanov.loc[mask_ahead_r, "x"]
ax3.plot(x_ahead, profile_rusanov.loc[mask_ahead_r, "rho_exact_ahead"], "k-", lw=1.5, label="exact (stationary)")
ax3.plot(x_ahead, profile_rusanov.loc[mask_ahead_r, "rho"], "C0.", ms=3, label="MUSCL+Rusanov")
ax3.plot(profile_hllc.loc[mask_ahead_h, "x"], profile_hllc.loc[mask_ahead_h, "rho"], "C1x", ms=4,
          label="MUSCL+HLLC")
ax3.axvline(bulk_x_hi, color="gray", ls=":", lw=1,
            label=f"bulk window ends here (last {boundary_margin} units is a separate BC artifact)")
ax3.set_xlabel("x")
ax3.set_ylabel("Density")
ax3.set_title("Ahead-of-shock region: Rusanov's bulk diffusion (left of the\ndotted line) vs. a separate, larger boundary-adjacent artifact (right of it)")
ax3.legend(fontsize=7)
fig3.tight_layout()
fig3.savefig(f"{FIGS}/shu_osher_ahead_diffusion.png", dpi=150)

# --- Figure 4: wake L1 error and ahead-of-shock max deviation vs. nx -------
# The ahead-of-shock panel plots the BULK mechanism (boundary-adjacent
# region excluded, both fluxes) plus Rusanov's own SEPARATE
# boundary-adjacent artifact as a third, explicitly-labeled dashed line
# -- NOT combined into one number (Ilia's independent final review
# caught an earlier version of this tutorial doing that and
# overstating the bulk mechanism's own magnitude 3-8x).
fig4, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4.5))
for flux, marker, color in [("rusanov", "o", "C0"), ("hllc", "s", "C1")]:
    rows = summary[summary["flux"] == flux].sort_values("nx")
    axa.plot(rows["nx"], rows["wake_l1_error"], f"{marker}-", color=color, label=flux)
    axb.semilogy(rows["nx"], rows["ahead_bulk_max_deviation"].clip(lower=1e-12), f"{marker}-", color=color,
                 label=f"{flux} (bulk)")
rusanov_rows = summary[summary["flux"] == "rusanov"].sort_values("nx")
axb.semilogy(rusanov_rows["nx"], rusanov_rows["ahead_boundary_max_deviation"].clip(lower=1e-12), "o--",
             color="C3", label="rusanov (boundary-adjacent, separate artifact)")
axa.set_xlabel("nx")
axa.set_ylabel("Post-shock wake L1 error vs. own fine reference")
axa.set_title("Wake fidelity vs. resolution")
axa.legend(fontsize=8)
axb.set_xlabel("nx")
axb.set_ylabel("Max ahead-of-shock deviation (log scale)")
axb.set_title("Ahead-of-shock error vs. resolution\n(HLLC stays at float roundoff)")
axb.legend(fontsize=8)
fig4.tight_layout()
fig4.savefig(f"{FIGS}/shu_osher_error_vs_resolution.png", dpi=150)

print(
    "Wrote figures/shu_osher_profiles.png, shu_osher_wake_closeup.png, "
    "shu_osher_ahead_diffusion.png, shu_osher_error_vs_resolution.png"
)
