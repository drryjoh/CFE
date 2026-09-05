# Phase 0: Execution Foundation

*PR #1 — progress review. No simulation physics yet; this PR is about how the*
*code will eventually run fast, not what it computes.*

---

## Slide: What are we actually building?

- CMU-CFE is a simulation engine — eventually it will model things like
  reacting fluid flow (think: combustion, multiple chemical species mixing
  and reacting inside a flow field).
- Before writing any of that science, we need the engine's "engine room" to
  work: how data is stored, and how work gets spread across hardware so it
  runs fast.
- This PR is that engine room. Zero physics. All plumbing.

---

## Slide: The mesh, in one picture

- A simulation splits space into a grid of small pieces called **cells**.
- Each cell holds its own little bundle of numbers — its **state** (e.g. how
  fast is the fluid moving here, how hot is it, how much of each chemical
  species is present).
- Picture a spreadsheet: one row per cell, one column per quantity. That's
  basically what we're storing.

---

## Slide: The core pattern — "one worker per cell"

- Much of CFD can be organized as: **assign one worker to each cell and
  have it perform the same operation.**
- That one small operation, run once per cell, is called a **kernel**.
- A worker is allowed to *read* data from neighboring cells — a real flow
  calculation almost always needs that. What makes this parallel-friendly
  is that we organize the work so each worker only ever *writes* its own
  cell's result, never someone else's — so workers never step on each
  other, even while reading each other's data.
- That's the whole reason GPUs matter here: a GPU is built from thousands
  of small workers (**threads**), and this "read freely, write only your
  own" structure is exactly the shape of problem they're built to chew
  through.

---

## Slide: Why test on a laptop CPU before touching a GPU?

- We wrote the "run this kernel on every cell" logic three ways: one
  worker doing everything in order (**serial**), several CPU workers
  splitting the cells (**threaded**), and a GPU version (**CUDA**).
- The CPU versions can be built and tested on any ordinary laptop — no
  special hardware needed. The GPU version needs an actual NVIDIA GPU and
  toolkit, which we didn't have yet for this PR.
- Getting the *pattern* proven correct on hardware we already have, before
  touching hardware we don't, is just good practice: fewer variables to
  debug at once. This is also why the GPU code is written but explicitly
  marked "unverified" everywhere — it's not being claimed as working, only
  as reviewed and ready to test.

---

## Slide: The other question — how should a cell's data sit in memory?

- Say each cell stores 20 numbers. There are two obvious ways to arrange
  a whole mesh of these in memory:
  - **Group by cell** — cell 1's 20 numbers together, then cell 2's 20
    numbers together, and so on. (Shorthand in the code: **AoS**.)
  - **Group by quantity** — every cell's 1st number together, then every
    cell's 2nd number together, and so on. (Shorthand: **SoA**.)
- Neither is obviously right. It depends on what the hardware likes to
  read — the same way it's faster to read one book cover-to-cover than to
  read page 1 of 50 different books.
- Since the whole point of this project is not guessing about performance,
  we built both and measured.

---

## Slide: What we found (CPU only, so far)

- Grouping by cell (AoS) won in every single test we ran on CPU — and the
  advantage grows the more numbers each cell stores.
- At 100 numbers per cell, AoS was up to **8x faster** than grouping by
  quantity (SoA).
- Why: the kernel touches all of a cell's numbers together, so keeping them
  physically next to each other in memory means the CPU grabs them in one
  trip instead of many scattered ones.

---

## Slide: Why this isn't the final answer yet

- This result is CPU-only. GPUs often read memory differently than CPUs
  do, and it's a known possibility that **SoA could actually win on GPU**
  — the opposite of what we just found on CPU.
- So: we're not picking a winner yet. The project's own rule is "don't
  assume, measure" — and we're missing half the measurement.
- This is exactly what the upcoming GPU access (CMU's Orchard system) will
  let us finish.

---

## Slide: What review caught and fixed on this PR

- A spot where read-only data could accidentally be edited — closed, and a
  test now guards against it coming back.
- The GPU code was pausing to wait after every single step even when it
  didn't need to yet — changed so it only waits when a result is actually
  needed, which matters once we're doing this thousands of times per
  simulated timestep.
- Added checks so that if something goes wrong on the GPU (bad memory
  copy, failed launch), it fails loudly and immediately instead of quietly
  producing wrong numbers.

---

## Slide: Bottom line / what's next

- Built: the plumbing (storage, CPU execution, GPU code pending hardware),
  all backed by 23 automated tests that check correctness, not just "it
  compiled."
- Measured: CPU prefers grouping data by cell, by a wide and growing
  margin.
- Next: run all of this on a real GPU (Orchard access incoming) and see if
  the CPU's answer holds up, changes, or needs a different answer per
  hardware type.

<!-- Next update: append CUDA/Orchard results here once available. -->
