# Phase 2 (first slice): Burgers Equation and Shock Capturing

*PR #3 — progress review. Phase 1 proved the engine (grid, ghost cells,*
*time-stepping) on the simplest possible physics: a quantity drifting at*
*a constant speed. This PR asks a harder question of that same engine:*
*what happens when the thing being transported can catch up with itself?*

---

## Slide: What's actually new here, physically?

- Phase 1's quantity moved at a **fixed** speed — like a leaf on a river
  with a constant current. The shape of the leaf never changed, it just
  slid along.
- This PR's equation (**Burgers' equation**, the standard toy problem
  for this in the field) is different in one crucial way: **the speed a
  point moves at is its own value.** A taller part of the wave moves
  faster than a shorter part next to it.
- That one change has a dramatic consequence: a smooth, gentle wave can
  **steepen itself into a cliff** in finite time, with no outside cause
  — the fast part simply catches up to the slow part ahead of it. That
  cliff is called a **shock** — a genuine, physical discontinuity, not a
  numerical glitch. This is the first time this project has had to deal
  with one.

---

## Slide: Why a shock is hard for a computer to represent

- Ordinary smooth-math techniques assume you can zoom in on any point
  and see something smooth. At a shock, that assumption is simply false
  — there's a real jump, and zooming in doesn't make it go away.
- A naive scheme (the same "look at my neighbors and draw a smooth line
  through them" trick Phase 1 used) reacts badly to this: right at the
  jump, it tries to draw a smooth line through a cliff, and overshoots —
  producing wiggles and values that were never physically present (e.g.
  slightly above the true maximum, or slightly below the true minimum).
  This is a well-known failure mode called **Gibbs oscillation**.
- The fix is a technique called a **slope limiter**: before drawing that
  smooth line through a cell's neighbors, check whether doing so would
  create a new high or low that didn't exist in the data. If it would,
  flatten the slope instead — sacrifice a little smoothness exactly
  where there's a jump, in exchange for never inventing values that
  aren't really there. This project uses the simplest, most
  conservative member of this family, called **minmod**.

---

## Slide: Proving it actually works — the moving-shock test

- The acceptance bar, same philosophy as every phase before this one:
  compare against the exact mathematical answer, not "it looks like a
  shock."
- Burgers' equation has an exact formula for how fast a shock moves
  (**Rankine-Hugoniot speed**: the average of the values on either
  side) — so we can set up a shock, let it travel, and check the
  simulation's shock lands exactly where the formula says it should.

| Grid cells | Shock position (exact) | Shock position (simulated) |
|---|---|---|
| 100 | 0.7500 | 0.7511 |
| 200 | 0.7500 | 0.7505 |
| 400 | 0.7500 | 0.7503 |

- The simulated position gets closer to exact as the grid refines, and
  at every resolution tested, the simulated value **never overshot or
  undershot** the true bounds of the data — the limiter doing its job,
  measured directly, not assumed.
- We also zoomed in on the shock itself at the finest grid to directly
  *see* what the limiter buys: without it, the jump is smeared across
  roughly 4-5 cells; with it, roughly 2-3. Fewer cells smeared = a
  sharper, more trustworthy answer at the same grid resolution.

---

## Slide: A smooth wave steepening into a shock, live

- Separately from the moving-shock test, we also started from a smooth,
  gentle wave (no jump at all) and watched it steepen on its own.
- Burgers' equation gives an exact formula for *when* the first shock
  must appear (the **breaking time**), derived from how fast the wave's
  steepest point is catching up to the point ahead of it. We ran the
  simulation right up to that moment and compared against the exact
  smooth-wave answer — matching closely, including in the final
  instants just before the shock forms.
- A precise distinction worth stating plainly, caught in review: the
  limiter from two slides ago is designed to be highly accurate on
  smooth data like this wave — that's its *textbook* rating. But because
  this specific wave happens to have a high point and a low point
  somewhere in the middle (not just a jump), the limiter's safety check
  quietly kicks in at exactly those two points too, every time, no
  matter how fine the grid — and that measurably drags down the
  *overall* accuracy we actually measured for this problem, from the
  textbook number to a real, consistently lower one. Both facts are
  true at once and don't contradict each other; we now report and test
  the number we actually measured, not the textbook one, and renamed the
  underlying test so its name says that too.
- After that moment, there is no simple exact formula to compare
  against anymore (the honest smooth-math answer becomes multi-valued,
  which is physically meaningless) — so past that point, the plots are
  labeled plainly as "what the simulation computed," not compared
  against anything, rather than quietly claiming a comparison that
  doesn't actually exist.
- One genuine bug surfaced and fixed during this: the comparison
  formula's own root-finding method became unstable in the last instant
  before the shock (where, mathematically, it's supposed to get
  difficult) and produced a visibly wrong wiggle in what should have
  been a smooth reference curve. Caught by actually looking at the
  generated plot, not just checking that the program ran without
  crashing — fixed with a more robust (if slightly slower) method that
  can't misbehave the same way.

---

## Slide: A diagonal shock in 2D — does the shock stay straight?

- Burgers' equation also works in two dimensions, where the quantity
  can move diagonally. We set up a shock along a 45-degree line and
  checked that it both (a) stays perfectly straight and (b) moves at
  exactly the speed the same exact formula predicts — no drift, no
  curving.
- This case is also the one place in this project so far where the
  "fake extra cells at the edge" (ghost cells, from Phase 1) needed to
  know not just "what's my neighbor's value" but "what time is it right
  now" — because here, the correct edge value itself changes as the
  shock sweeps past. Handled by directly writing out the two-step
  time-advance by hand for this one case, instead of using the
  project's shared one-step helper, specifically so the correct instant
  could be threaded through to the edges at each of the two steps.
- A genuine inconsistency was caught and fixed here too: cells sitting
  exactly on the shock line were initially being told "you're either
  fully in or fully out" at the edges, when every other cell in the
  simulation (including the very same cells on the interior) was
  correctly told its exact fractional share. Fixing this measurably
  improved the accuracy of the whole simulation by roughly 40% at the
  same grid resolution — a real, not cosmetic, correction.

---

## Slide: First-order vs. the limiter, side by side

- To make sure the limiter is actually buying something — not just
  "different," but *better* — every test above was also run with the
  simplest possible scheme (no slope at all, the crudest, most
  conservative choice) and compared directly.
- Result, consistent across the 1D and 2D cases: the limited scheme's
  error is consistently **about half** the simple scheme's error at the
  same grid resolution, and visibly sharper right at the shock itself
  (see the zoom-in plot above). Both still only improve at the same
  *rate* as the grid refines, right at the shock specifically — a
  known, accepted property of any scheme trying to represent a genuine
  discontinuity with a finite number of cells, not a flaw in this one.

---

## Slide: CPU first, GPU second, 3D proven properly — same discipline as Phase 1

- Same sequencing as every phase before: prove it on ordinary hardware,
  then the GPU, then make sure "3D" actually means all three directions
  are doing real work, not just going through the motions.
- A real gap was caught mid-review here: the first 3D tests only varied
  the quantity along one direction, leaving the other two directions
  with nothing to transport at all — meaning a bug specific to those
  two directions could have hidden in plain sight, undetected by any
  test. Fixed by adding a case that varies in all three directions at
  once, built to be perfectly symmetric between two of those
  directions, so a real bug would visibly break that symmetry rather
  than disappearing.
- GPU results, verified correct to a tight tolerance against the CPU
  answer: a 100-million-cell 1D line in about 13 milliseconds per step,
  and a 134-million-cell (512x512x512) 3D volume in about 33
  milliseconds per step — slightly slower than Phase 1's simpler
  equation at the same scale, because this scheme genuinely does more
  work per cell (the limiter's decision, plus the shock-safe combining
  formula), not because anything is wrong.

---

## Slide: Something you can actually watch

- Beyond tables and plots, this PR also produced a short animation: a
  smooth 3D bump that visibly steepens into a shock on its leading edge
  while spreading out smoothly on its trailing edge — the same
  self-steepening behavior from the very first slide, but now watchable
  in 3D rather than described in words.
- Checked, not just rendered: the simulated values never left the
  bounds the starting shape itself set, confirmed directly from the
  program's own output at every frame of the animation, for the whole
  run.

---

## Slide: Bottom line / what's next

- Built: the project's first genuinely self-steepening equation, paired
  with a shock-capturing scheme that is provably non-oscillatory, slotted
  into the exact same grid/time-stepping engine Phase 1 built — zero
  changes needed to that shared engine itself.
- Verified: exact shock-speed agreement in 1D and 2D, a smooth-wave
  comparison right up to the mathematically exact moment a shock must
  form, conservation, a genuine (not degenerate) 3D correctness check,
  and CPU/GPU agreement at the same scale Phase 1 established.
- Produced: two fully reproducible, plotted verification tutorials
  (anyone can re-run the exact numbers behind every plot and number in
  this presentation) and a new 3D animation.
- Caught and fixed, via independent review, three real issues before
  merge: a sign bug that only shows up when this project's newer flux
  scheme is paired with its older, linear equation; the 2D shock-edge
  inconsistency above; and the 3D blind-spot above — the project's
  review process catching exactly the kind of thing a quick self-check
  would miss.
- Next: this is the first genuinely nonlinear equation handled end to
  end — the remaining pieces of Phase 2 (splitting the work across
  multiple machines, and a first prototype of the project's other
  intended numerical method) build on having proven that out.
