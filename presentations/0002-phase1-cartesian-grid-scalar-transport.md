# Phase 1: Cartesian Grid and Scalar Transport

*PR #2 — progress review. This is the first PR that solves an actual*
*physics problem, however simple, on top of Phase 0's engine room.*

---

## Slide: What problem is this PR actually solving?

- Phase 0 built the "engine room" — storage and parallel execution — but
  computed nothing physical. This PR writes the first real piece of CFD:
  **taking one quantity and moving it through a grid at a constant
  speed**, the way a dye or a puff of smoke drifts along with a steady
  wind.
- In fluids jargon this is called **scalar advection/transport** — a
  single transported quantity carried along by a known velocity, with no
  chemistry, no pressure, no turbulence yet. It's the simplest real PDE
  in the roadmap, chosen specifically so we can prove the *method* works
  before adding any of that complexity on top.

---

## Slide: The grid, and the "fake extra cell" trick

- Space is split into a grid of cells, same idea as Phase 0. Each cell
  now also needs to know its **neighbors**, because computing how much
  of the quantity flows out of a cell requires knowing what's next door.
- Every real cell needs a neighbor on every side — including cells at
  the very edge of the domain, which don't have a real neighbor there.
- The fix: surround the real grid with a thin ring of extra cells called
  **ghost cells**. They aren't part of the real simulation — they're
  filled in with whatever value the boundary condition says they should
  have (e.g. "wrap around to the other side" for a periodic domain, or
  "always hold this fixed value" for a wall). Once they're filled, every
  real cell — edge or interior — can run the *exact same* calculation
  with no special-casing. That's the whole point of ghost cells: they
  turn "handle edges differently" into "there are no edges."

---

## Slide: How a cell "talks" to its neighbor (without hardcoding the physics)

- Computing the flow across the boundary between two cells happens in
  two separate steps, on purpose:
  1. Each cell independently produces its own best guess at "what's my
     value right at this shared boundary?", using only its own nearby
     data.
  2. A separate, tiny piece of logic then combines the two neighboring
     cells' guesses into one final answer for how much material actually
     crosses that boundary.
- Why split it into two steps instead of one? Because step 1 (how a cell
  represents its own data) and step 2 (how two guesses get combined) are
  independent ideas, and future, more sophisticated methods (the roadmap
  eventually wants a completely different way of representing data
  inside a cell, called DG) only need to replace step 1 — step 2 doesn't
  care how step 1 got its answer, only what number it produced. Keeping
  them separate now means we don't have to rewrite this later.

---

## Slide: Why we picked a 2-stage time-stepper, not a 3-stage one

- Advancing the simulation forward in time is done with a small family of
  methods called **Runge-Kutta** steppers — think of them as "take a
  trial step, check how things are changing, then take a corrected real
  step," possibly with more than one round of trial-and-correct.
- We had a choice between a 2-stage version and a 3-stage version. The
  3-stage version is more accurate in isolation, but our spatial method
  (step 1 above) is only accurate to a certain degree itself — pairing a
  more-accurate time-stepper with a less-accurate spatial method doesn't
  actually make the overall answer any better, since the *weaker* of the
  two links in the chain is what limits the final accuracy.
- So we picked the 2-stage version: same overall accuracy, one less step
  of work every single time-step. We verified directly that this
  reasoning holds (see the refinement study below) rather than just
  assuming it.

---

## Slide: The real acceptance test — does refining the grid help exactly as much as it should?

- The core scientific claim to prove: this method is **"2nd-order
  accurate,"** meaning if you cut the cell size in half, the error in
  the answer should shrink by a factor of about 4 (half squared).
- "It ran and looked reasonable" is explicitly not good enough — we ran
  the actual test: start with a known smooth wave shape, run the
  simulation, compare against the exact mathematical answer, then repeat
  at a finer grid and check the error actually shrank by ~4x.

| Grid cells | Error vs. exact answer | Error shrank by... |
|---|---|---|
| 20 | 0.0339 | — |
| 40 | 0.00846 | 4.01x |
| 80 | 0.00211 | 4.01x |
| 160 | 0.000528 | 4.00x |
| 320 | 0.000132 | 4.00x |

- Every single doubling shrinks the error by almost exactly 4x, and gets
  closer to exactly 4x as the grid gets finer — textbook 2nd-order
  behavior, measured, not assumed.
- We also checked a second, independent property: on a domain that wraps
  around (periodic), the *total amount* of the transported quantity
  should never change, no matter how it sloshes around. Confirmed to
  floating-point precision after 200 time-steps.

---

## Slide: Why test 1D and a flat 2D case before a real 3D case

- Same philosophy as Phase 0: prove the smallest, easiest-to-check case
  first. A 1D line of cells is small enough that a bug is obvious; a full
  3D volume is not.
- One easy-to-miss trap we specifically caught: our first "at scale, on
  the GPU" demonstration only ever moved a wave along one direction (1D),
  even though the code was written to handle 3D. A big, impressive cell
  count in 1D does not actually prove the 3D machinery works — so before
  calling this done, we added a genuine 3D case (a bump moving
  diagonally through a cube, all three directions active at once) and
  re-verified everything on the GPU. This is worth stating plainly
  because it's an easy mistake to make in good faith: a large number
  alone is not evidence of the specific thing you actually need proven.

---

## Slide: CPU first, GPU second — same reasoning as Phase 0

- Once more: prove correctness on ordinary hardware everyone already has
  access to, then move to the GPU.
- CPU results: this solver's per-cell work is heavier than Phase 0's
  simple kernel (it needs to look at four neighboring cells, not just
  itself), so multi-core threading starts paying off at a smaller grid
  size than before — profitable by around 100,000 cells, and about 1.25x
  faster than single-core by 10 million cells.
- GPU results (once we got real hardware access again): the same
  solver, verified correct on the GPU to the same tight tolerance as
  before, running on:
  - a **100-million-cell 1D line** in about 11 milliseconds per
    time-step, and
  - a **134-million-cell (512x512x512) 3D volume** in about 31
    milliseconds per full time-step.
- For context: that 3D case is a grid large enough that if you tried to
  print out every cell's number, one per line, you'd need more pages
  than most printers will ever produce in their lifetime — and the GPU
  updates every single one of those cells, from every direction, in
  about the time it takes to blink.

---

## Slide: A bug worth explaining, because it's a genuinely sneaky category

- While getting the GPU version correct, we hit a bug where the GPU's
  answer and a "reference" CPU answer disagreed. Chasing it down
  revealed something worse than a simple math mistake: a piece of code
  that was supposed to work on *either* the CPU or the GPU had
  accidentally been marked "GPU only."
- On an ordinary CPU-only build, that mistake is invisible — the "GPU
  only" marking simply does nothing there, so all our CPU tests kept
  passing the entire time, hiding the problem completely.
- It only became visible in the specific situation of compiling
  CPU-and-GPU-shared code *using the GPU compiler*, where the mislabeled
  code silently produced all-zero answers instead of raising an error.
  We tracked it down with a very targeted, minimal side-experiment (run
  just one calculation, no time-stepping, and compare cell-by-cell)
  rather than staring at the full failing test — the same "test on the
  smallest complete story you can" discipline behind everything else in
  this project.

---

## Slide: Something new — an actual picture, not just numbers

- Every result above is a number in a table. For this PR we also
  produced something you can actually *look at*: a short animation of a
  blob of the transported quantity drifting diagonally through a 3D box
  and wrapping around the edges, using a free, standard visualization
  format any of us can open.
- We double-checked it isn't just a pretty animation of nonsense: the
  blob's tracked position across the whole animation lines up exactly
  with where simple algebra says it should be at each moment in time,
  including the wrap-around.

---

## Slide: Bottom line / what's next

- Built: a working, dimension-generic (1D/2D/3D, same code) scalar
  transport solver, running correctly and fast on both CPU and GPU, with
  formal proof it hits its required accuracy — not just "it compiled."
- Verified: 2nd-order convergence (measured, not assumed), conservation,
  CPU/GPU agreement in both 1D and a genuine 3D case, at grid sizes up to
  ~134 million cells.
- Produced: an actual animation you can open and watch, not just a
  spreadsheet of numbers.
- Next: this same two-step "each side speaks for itself, then combine"
  design is what lets the roadmap's next real physics (a nonlinear
  equation, then eventually full compressible flow) plug in without
  redesigning this layer — that's what Phase 2 builds on.
