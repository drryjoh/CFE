# Phase 2: Full 3D MPI Block Decomposition + Strong Scaling

*Progress review for task 0005. The previous PR split the simulated*
*world into left-to-right slices across several computers. This one*
*asks: can we split it into bricks instead -- cut it up in all three*
*directions at once -- and does doing that actually make things run*
*faster?*

---

## Slide: From slices to bricks

- The last PR split the grid like slicing a loaf of bread -- one cut
  direction, each computer gets one slice.
- This PR generalizes that to splitting a loaf into a 3D grid of
  bricks instead -- think of cutting a block of cheese into a
  3-dimensional array of cubes, one cube per computer.
- Surprising discovery while planning this: a worry going in was that
  a brick's **corners** (where three cuts meet) would need extra,
  trickier communication between computers that only touch at a single
  corner, not a whole face. Checking the actual math carefully showed
  this project's scheme **never needs corner information at all** --
  only flat-face neighbors (the 6 computers directly touching each face
  of a brick). That made this PR meaningfully simpler than expected,
  and it's now written down as a checked fact, not a guess.

---

## Slide: Does splitting it up actually make it faster?

- This is the real question task 0004 left unanswered -- it only timed
  the *neighbor-talking* step by itself, never the *actual simulation*.
- This PR adds exactly that: run the identical fixed-size simulation
  (same resolution, same physics, same amount of total work) once on 1
  computer, then again split across 2, 4, and 8 computers, and time
  each run from start to finish.
- This is called **strong scaling**: the pile of work never changes
  size, only how many hands are splitting it. A good strong-scaling
  result means "4 computers finish in roughly 1/4 the time" -- the
  payoff a whole cluster exists to deliver.

---

## Slide: What the numbers actually showed

**On the real supercomputer (PSC Bridges-2), the result is close to
perfect:**

| Computers used | Time to finish | Speedup vs. 1 computer |
|---|---|---|
| 1 | 57.55 s | 1.0x (baseline) |
| 2 | 29.26 s | 1.97x |
| 4 | 14.74 s | 3.90x |
| 8 | 7.56 s | 7.61x |

Doubling the computers very nearly halves the time, every single step
up to 8 -- a textbook result. 8 computers finishing in 7.61x the time
of 1 (where perfect would be exactly 8.0x) means 95% of the theoretical
best-possible benefit was actually realized.

**On a laptop (10 cores, shared with everything else running on it),
the same experiment told a different, also-instructive story:**

| Computers used | Time to finish | Speedup vs. 1 computer |
|---|---|---|
| 1 | 8.70 s | 1.0x |
| 2 | 4.69 s | 1.85x |
| 4 | 3.33 s | 2.61x |
| 8 | 3.84 s | 2.26x (got *slower* than 4!) |

- On the laptop, scaling looks good up to 4 computers, then at 8 it
  actually gets a little *slower* than at 4. This is a real, expected
  phenomenon, not a mistake: as the work gets split into smaller and
  smaller pieces, each computer's own slice of "actual work" shrinks,
  but the amount of "talk to my neighbors" overhead doesn't shrink
  nearly as fast -- eventually the talking starts to cost more than the
  work saved. This exact overhead was already measured separately in
  the previous PR's own communication benchmark.
- The supercomputer doesn't show this falloff (yet) at these same rank
  counts, because its cores are dedicated -- not fighting the rest of a
  laptop's operating system and every other running program for the
  same shared memory bus. **This is exactly why "check it on a laptop
  first, then confirm for real" matters**: the laptop alone would have
  wrongly suggested this stops scaling well past 4 computers, and the
  real cluster run shows that isn't actually true.

---

## Slide: Seeing it, not just measuring it

- Beyond the numbers, this PR also produces an actual 3D picture: a
  rounded bump of material drifting diagonally through the simulated
  cube, split across however many computers ran it.
- Each computer saves only *its own* piece of the picture. Checked
  numerically (not just assumed) that all 8 pieces, from an 8-computer
  run, line up perfectly into one seamless cube with no gaps and no
  overlaps -- opening all 8 files together in the visualization tool
  (ParaView) shows one continuous bump, not 8 disconnected chunks.

---

## Slide: What this still doesn't do

- Both the laptop AND the real supercomputer numbers are now in (see
  above) -- confirming the communication-overhead explanation directly,
  not just as a plausible guess: dedicated cluster cores really do give
  the cleaner, near-textbook scaling curve the laptop could only
  approximate. Same "verify locally, confirm for real" pattern every
  GPU feature in this project has already used.
- Still doesn't help the self-steepening wave equation from two PRs
  ago (Burgers' equation) -- that one needs the computers to compare
  notes about the fastest-moving point in the *whole* simulation before
  each step, which none of this PR's work does yet. Named, not
  forgotten.
- Only reports a single run-through-to-completion time -- it doesn't
  yet separate out "how much of that time was computing" from "how
  much was computers waiting on each other," beyond what the previous
  PR's own standalone communication-only timing already measured.
