#!/usr/bin/env python3
"""Plot the Sod shock tube tutorial's output (data/*.csv -> figures/*.png).

Run after the `cfe_euler_1d_sod_shock_tube` executable (see this
directory's README.md). Requires matplotlib + pandas, like the Burgers
tutorials' own plot scripts.

Task 0011 (docs/adr/0012-hllc-numerical-flux.md) extended this tutorial
to run BOTH MUSCL+Rusanov and MUSCL+HLLC at every resolution -- this
script now plots them side by side, including a zoomed-in view right at
the contact discontinuity (the one place the two fluxes are expected to
differ visibly) and the measured contact-width-vs-resolution comparison
the task's own acceptance criteria asked for.
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data"
FIGS = "figures"

summary = pd.read_csv(f"{DATA}/summary.csv")
finest_nx = int(summary["nx"].max())

profile_rusanov = pd.read_csv(f"{DATA}/case_nx{finest_nx}_rusanov.csv")
profile_hllc = pd.read_csv(f"{DATA}/case_nx{finest_nx}_hllc.csv")

# --- Figure 1: full profiles at the finest resolution vs. exact ------------
fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
for ax, (num_col, exact_col, label) in zip(
    axes,
    [
        ("rho_numerical", "rho_exact", "Density"),
        ("u_numerical", "u_exact", "Velocity"),
        ("p_numerical", "p_exact", "Pressure"),
    ],
):
    ax.plot(profile_rusanov["x"], profile_rusanov[exact_col], "k-", lw=1.5, label="exact")
    ax.plot(profile_rusanov["x"], profile_rusanov[num_col], "C0.", ms=3, label=f"MUSCL+Rusanov (nx={finest_nx})")
    ax.plot(profile_hllc["x"], profile_hllc[num_col], "C1x", ms=4, label=f"MUSCL+HLLC (nx={finest_nx})")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8)
axes[-1].set_xlabel("x")
fig.suptitle(f"Sod shock tube at t=0.2 (nx={finest_nx})")
fig.tight_layout()
fig.savefig(f"{FIGS}/sod_profiles.png", dpi=150)

# --- Figure 2: zoomed-in density right at the contact discontinuity -------
x_contact = 0.5 + 0.927453 * 0.2  # exact contact location (Toro Table 4.1 u*)
window = 0.08
mask_r = (profile_rusanov["x"] > x_contact - window) & (profile_rusanov["x"] < x_contact + window)
mask_h = (profile_hllc["x"] > x_contact - window) & (profile_hllc["x"] < x_contact + window)

fig_zoom, ax_zoom = plt.subplots(figsize=(6, 4.5))
ax_zoom.plot(profile_rusanov["x"], profile_rusanov["rho_exact"], "k-", lw=1.5, label="exact")
ax_zoom.plot(profile_rusanov.loc[mask_r, "x"], profile_rusanov.loc[mask_r, "rho_numerical"], "C0o-", ms=4,
             label="MUSCL+Rusanov")
ax_zoom.plot(profile_hllc.loc[mask_h, "x"], profile_hllc.loc[mask_h, "rho_numerical"], "C1s-", ms=4,
             label="MUSCL+HLLC")
ax_zoom.axvline(x_contact, color="gray", ls=":", lw=1, label="exact contact position")
ax_zoom.set_xlim(x_contact - window, x_contact + window)
ax_zoom.set_xlabel("x")
ax_zoom.set_ylabel("Density")
ax_zoom.set_title(f"Contact discontinuity close-up (nx={finest_nx})")
ax_zoom.legend(fontsize=8)
fig_zoom.tight_layout()
fig_zoom.savefig(f"{FIGS}/sod_contact_closeup.png", dpi=150)

# --- Figure 3: L1 error vs. resolution, both fluxes -------------------------
fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
for flux, marker, color in [("rusanov", "o", "C0"), ("hllc", "s", "C1")]:
    rows = summary[summary["flux"] == flux]
    ax2.loglog(rows["nx"], rows["l1_rho"], f"{marker}-", color=color, label=f"L1(rho), {flux}")
ax2.set_xlabel("nx")
ax2.set_ylabel("L1 error vs. exact solution")
ax2.set_title("Sod shock tube: density L1 error vs. resolution\n(discontinuities cap the rate below 2nd order)")
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{FIGS}/sod_l1_convergence.png", dpi=150)

# --- Figure 4: measured contact width vs. resolution, both fluxes ----------
fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
for flux, marker, color in [("rusanov", "o", "C0"), ("hllc", "s", "C1")]:
    rows = summary[summary["flux"] == flux].sort_values("nx")
    ax3.plot(rows["nx"], rows["contact_width_cells"], f"{marker}-", color=color, label=flux)
ax3.set_xlabel("nx")
ax3.set_ylabel("Contact width (cells, 5%-95% crossing)")
ax3.set_title("Measured contact-discontinuity width: HLLC vs. Rusanov")
ax3.legend()
fig3.tight_layout()
fig3.savefig(f"{FIGS}/sod_contact_width.png", dpi=150)

print("Wrote figures/sod_profiles.png, sod_contact_closeup.png, sod_l1_convergence.png, "
      "sod_contact_width.png")
