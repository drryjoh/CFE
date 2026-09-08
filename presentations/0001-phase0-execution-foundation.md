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

---
---

# Update: the GPU results are in

*Everything below happened after the slides above. We got real GPU access*
*(PSC Bridges-2, a national supercomputing center) and finally ran the code*
*that was "written but unverified." Same PR, same story — this is the*
*ending.*

---

## Slide: What we actually tested this round

- Last time, the GPU ("CUDA") version of the code existed but had never
  been compiled or run — no GPU was available yet.
- The earlier slides expected this to happen on CMU's Orchard system; it
  ended up happening sooner, on PSC Bridges-2 instead (a national
  supercomputing center's cluster, not CMU's) — same goal, different
  machine.
- This round: we got time on an NVIDIA V100 GPU (one of the workhorse GPUs
  used in scientific computing) on that cluster.
- We compiled the GPU code for the first time, ran all of its automated
  correctness tests, ran the same performance benchmark we ran on CPU, and
  used a profiling tool (Nsight Compute — think of it as an X-ray machine
  for what's actually happening inside the GPU while code runs) to look
  under the hood.

---

## Slide: Why this matters for the actual CFD science

- Every future piece of physics in this project — compressible flow,
  combustion chemistry, turbulence — will eventually be written as exactly
  the same shape of calculation this PR tests: one small operation,
  repeated across every cell in the mesh, over and over, once per
  simulated timestep.
- A real reacting-flow simulation might do this millions of times per run
  (millions of cells x thousands of timesteps). If the underlying
  "engine room" is inefficient, a simulation that should take hours could
  take days or weeks instead — or need hardware nobody can afford.
- That's why this unglamorous plumbing work happens *before* any physics
  gets written: every future kernel inherits whatever decisions get made
  here about how data is stored and how work is spread across hardware.

---

## Slide: Good news first — the GPU code is correct

- All of the GPU code's automated tests passed, on real hardware, for the
  first time ever in this project's history.
- Practically: the GPU version of "do this calculation on every cell"
  produces the same answer as the already-trusted CPU version, within the
  tiny rounding differences you always expect when comparing two different
  pieces of hardware doing the same math.

---

## Slide: The big surprise — GPU flips the CPU's answer

- Recall from before: on CPU, grouping a cell's numbers together (AoS)
  beat spreading them out by quantity (SoA) — by up to 8x at 100 numbers
  per cell.
- On the GPU, the opposite happens, and by a much bigger margin: spreading
  the numbers out (SoA) beat grouping them together (AoS) by up to **33x**
  at 100 numbers per cell.
- Why the flip? A GPU doesn't have one worker grabbing one cell's worth of
  data at a time — it has thousands of workers (threads) all grabbing data
  *at the same instant*, one cell each. If those simultaneous workers are
  each reaching for numbers that sit right next to each other in memory,
  the hardware can serve all of them in one trip. If instead each worker
  is reaching for numbers scattered far apart (which is exactly what
  "grouped by cell" looks like from thousands of workers' point of view,
  all firing at once), the hardware has to make many separate trips
  instead of one. SoA happens to be the layout that lines those
  simultaneous requests up neatly; AoS is the layout that scatters them.
- We didn't just guess this — the X-ray-machine profiler measured it
  directly: at 100 numbers per cell, the grouped-by-cell version was only
  using about a quarter of each chunk of memory it fetched (the rest was
  wasted overhead), while the spread-out version was using nearly all of
  it.
- Bottom line: **there is no single best answer — CPU wants data grouped
  by cell, GPU wants it spread out by quantity, and now we have real
  measurements proving both halves of that instead of assuming either
  one.**

---

## Slide: The other thing we checked — does the GPU run out of scratch space?

- Each GPU worker has a small amount of ultra-fast personal scratch space
  (called a *register*) to hold the numbers it's actively working with
  right now. If a calculation needs more scratch space than a worker has,
  the extra has to spill into much slower memory — quietly making
  everything run worse without any error being raised.
- This matters a lot for where this project is headed: a real reacting
  flow with many chemical species might need on the order of 100 numbers
  per cell (one per species, roughly), and we need to know now whether
  that many numbers overflows a GPU worker's scratch space.
- We checked, for every state size we'll need (1 all the way up to 100
  numbers per cell): **zero spilling, every time.** The GPU workers never
  ran out of scratch space, even at the largest size.
- This is good news specifically for the chemistry/combustion work later
  in the project's roadmap — it's one less thing to worry about when that
  work starts.

---

## Slide: Two bugs found and fixed along the way

- **A GPU-compiler quirk**: the GPU compiler is picky about one specific
  code shape (a particular way of nesting two pieces of code) and flatly
  refused to compile it. This wasn't a math error — just a "the compiler
  wants this written slightly differently" issue, fixed by restructuring
  the code the way the compiler wanted.
- **A test that only failed on one kind of computer**: one automated test
  failed consistently on the Bridges-2 machine but never on a Mac. Rather
  than assume the test or the code was broken, we tracked down the actual
  cause: two different, equally valid ways of rounding a floating-point
  calculation, which different compilers choose between differently. Once
  confirmed (using tools designed to catch memory bugs, which came back
  clean), the test's expectation was loosened to the correct standard
  instead of an accidentally-too-strict one. This is exactly the kind of
  thing that's easy to wave away as "flaky" and move on from — we didn't,
  and it turned out to be a real, explainable, fixable thing rather than
  a mystery.

---

## Slide: Bottom line / what's next

- Phase 0's biggest open question — "does the GPU code even work, and
  which way should data be organized on each type of hardware?" — is now
  answered with real evidence on both CPU and GPU, not assumptions.
- Both the CPU and GPU "engine rooms" are now proven correct and measured,
  with a documented answer for how data should be organized on each.
- This PR is done. Next up: Phase 1, which starts writing the first real
  piece of CFD physics (a simple transported quantity moving through a
  grid) on top of this now fully-verified foundation.
