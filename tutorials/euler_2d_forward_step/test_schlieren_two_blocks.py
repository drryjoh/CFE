#!/usr/bin/env python3
"""Regression test for `schlieren_proxy_two_blocks` (plot_results.py).

Exists specifically because this function was implemented once (commit
eeea040), then silently reverted back to the broken naive per-block
`np.gradient` call by a later, unrelated commit (6a754fe) that happened
to touch the same function's call site while extending the resolution
sweep -- a plain visual check of the regenerated figure did not catch
it at the time. This test makes that regression impossible to
reintroduce silently: it fails loudly (nonzero exit, printed diffs) if
`schlieren_proxy_two_blocks` ever again computes each block's gradient
in isolation instead of stitching in the true cross-block neighbor.

Uses a synthetic SMOOTH two-block field with a known analytic gradient
(no actual simulation data needed) so the expected answer is exact, not
just "whatever the solver happens to produce". Tests the function's
actual return value directly (the schlieren magnitude), not a
reimplementation of its internals:
  - In the shared-height band (where block2 physically exists),
    `schlieren_proxy_two_blocks`'s two edge columns must be close to
    the analytic |grad(f)|, and substantially closer than the naive
    per-block `schlieren_proxy` applied to each block in isolation.
  - Below the step height (where block2 does not exist), the stitched
    result must be IDENTICAL to the naive one-sided estimate --
    "respecting the solid step" rather than inventing a cross-block
    value where there is no real neighbor.

Run directly: `python3 test_schlieren_two_blocks.py` (no pytest
dependency, consistent with this tutorial's existing numpy/matplotlib
tooling -- see README.md's "Regenerating the figures" section).
"""
import math
import sys

import numpy as np

from plot_results import schlieren_proxy, schlieren_proxy_two_blocks


def f(x, y):
    """Smooth, nonlinear, nonzero-gradient-in-both-axes test function."""
    return math.sin(3.0 * x) + 0.5 * math.cos(2.0 * y) + 0.2 * x * y


def grad_f(x, y):
    dfdx = 3.0 * math.cos(3.0 * x) + 0.2 * y
    dfdy = -1.0 * math.sin(2.0 * y) + 0.2 * x
    return math.hypot(dfdx, dfdy)


def build_two_block_case(dx=0.05, nx1=30, ny1=50, step_rows=10, nx2=40):
    """Mirrors the real forward-step geometry: block1 is full height,
    block2 is shorter and offset upward by `step_rows` cells, with
    origin2_x exactly abutting block1's real right edge (the same
    invariant `ForwardStepBlocks`'s constructor now enforces on the
    C++ side)."""
    dy = dx
    ny2 = ny1 - step_rows
    origin1 = (0.0, 0.0)
    origin2 = (nx1 * dx, step_rows * dy)

    f1 = np.empty((ny1, nx1))
    for j in range(ny1):
        y = origin1[1] + (j + 0.5) * dy
        for i in range(nx1):
            x = origin1[0] + (i + 0.5) * dx
            f1[j, i] = f(x, y)

    f2 = np.empty((ny2, nx2))
    for j in range(ny2):
        y = origin2[1] + (j + 0.5) * dy
        for i in range(nx2):
            x = origin2[0] + (i + 0.5) * dx
            f2[j, i] = f(x, y)

    b1 = (f1, origin1, (dx, dy))
    b2 = (f2, origin2, (dx, dy))
    return b1, b2, step_rows, dx, dy


def main():
    b1, b2, step_rows, dx, dy = build_two_block_case()
    (f1, origin1, s1) = b1
    (f2, origin2, s2) = b2
    ny2 = f2.shape[0]

    # Naive: each block's gradient MAGNITUDE computed in isolation --
    # exactly what the reverted code at commit 6a754fe did.
    g1_naive = schlieren_proxy(f1, s1)
    g2_naive = schlieren_proxy(f2, s2)

    # The function under test.
    g1_stitched, g2_stitched = schlieren_proxy_two_blocks(b1, b2)

    failures = []

    # --- Shared-height band: stitched must be close to the analytic
    # gradient magnitude, and substantially better than naive, at both
    # edge columns. -----------------------------------------------------
    worst_naive_err = 0.0
    worst_stitched_err = 0.0
    x_edge1 = origin1[0] + (f1.shape[1] - 0.5) * dx  # block1's last column center
    x_edge2 = origin2[0] + 0.5 * dx  # block2's first column center
    for j2 in range(ny2):
        j1 = j2 + step_rows
        y = origin2[1] + (j2 + 0.5) * dy

        analytic1 = grad_f(x_edge1, y)
        analytic2 = grad_f(x_edge2, y)

        naive_err = max(abs(g1_naive[j1, -1] - analytic1), abs(g2_naive[j2, 0] - analytic2))
        stitched_err = max(abs(g1_stitched[j1, -1] - analytic1), abs(g2_stitched[j2, 0] - analytic2))
        worst_naive_err = max(worst_naive_err, naive_err)
        worst_stitched_err = max(worst_stitched_err, stitched_err)

    print(f"worst naive (per-block, one-sided) edge-column error vs. analytic |grad(f)|: {worst_naive_err:.6f}")
    print(f"worst stitched (cross-block, centered) edge-column error vs. analytic |grad(f)|: {worst_stitched_err:.6f}")

    if not (worst_stitched_err < 0.05):
        failures.append(
            f"stitched edge-column gradient error {worst_stitched_err:.6f} is too large for a smooth "
            "field with a centered-difference stitch -- schlieren_proxy_two_blocks may be broken, not "
            "just reverted to the naive per-block form.")
    if not (worst_stitched_err < worst_naive_err / 3.0):
        failures.append(
            f"stitched error ({worst_stitched_err:.6f}) is not substantially better than naive error "
            f"({worst_naive_err:.6f}) -- this is exactly the regression where schlieren_proxy_two_blocks "
            "silently falls back to independent per-block np.gradient calls (commit 6a754fe's bug).")

    # --- Below the step: no neighbor exists, so the stitched function's
    # OWN return value must be IDENTICAL to the naive one there (not
    # inventing a cross-block value across the solid step). ------------
    if step_rows > 0:
        below_step_naive = g1_naive[:step_rows, -1]
        below_step_stitched = g1_stitched[:step_rows, -1]
        if not np.array_equal(below_step_naive, below_step_stitched):
            failures.append(
                "rows below the step height (no real cross-block neighbor) were changed by the "
                "stitching -- should be left exactly as the one-sided estimate, not have a value "
                "invented across the solid step.")
        else:
            print(f"below-step rows (0..{step_rows - 1}): correctly left unchanged (no neighbor to stitch).")

    # --- Interior columns (away from the interface, both blocks): must
    # be completely untouched -- no smoothing anywhere except the two
    # edge columns actually adjacent to the interface. ------------------
    if not np.array_equal(g1_naive[:, :-1], g1_stitched[:, :-1]):
        failures.append("block1 interior columns (all but the last) were changed by stitching -- should "
                         "be untouched.")
    if not np.array_equal(g2_naive[:, 1:], g2_stitched[:, 1:]):
        failures.append("block2 interior columns (all but the first) were changed by stitching -- should "
                         "be untouched.")

    if failures:
        print("\nFAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nPASS: schlieren_proxy_two_blocks stitches the true cross-block neighbor (not a per-block "
          "one-sided fallback), touches only the two interface-adjacent columns, and respects the solid "
          "step where no neighbor exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
