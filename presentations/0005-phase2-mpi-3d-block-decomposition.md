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

| Computers used | Time to finish | Speedup vs. 1 computer |
|---|---|---|
| 1 | 8.70 s | 1.0x (baseline) |
| 2 | 4.69 s | 1.85x |
| 4 | 3.33 s | 2.61x |
| 8 | 3.84 s | 2.26x (got *slower* than 4!) |

*(laptop numbers above -- see the tutorial's own README for the*
*PSC Bridges-2 supercomputer numbers, the authoritative result)*

- Scaling looks great up to 4 computers -- not quite perfect (perfect
  would be exactly 4.0x), but close, and clearly worth doing.
- At 8 computers, it actually got a little *slower* than at 4. This is
  a real, expected phenomenon, not a mistake: as the work gets split
  into smaller and smaller pieces, each computer's own slice of
  "actual work" shrinks, but the amount of "talk to my neighbors"
  overhead doesn't shrink nearly as fast -- eventually the talking
  starts to cost more than the work saved. This exact overhead was
  already measured separately in the previous PR's own communication
  benchmark, so this PR's result isn't a surprise -- it's a direct
  confirmation of what that number predicted.

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

- The communication-overhead explanation above is strong evidence, but
  this PR's laptop numbers are still just a laptop -- a real
  supercomputer's dedicated cores (not shared with everything else
  running on a laptop) should give a cleaner, more textbook-looking
  scaling curve. Both are reported, laptop first, cluster to follow the
  same "verify locally, confirm for real" pattern every GPU feature in
  this project has already used.
- Still doesn't help the self-steepening wave equation from two PRs
  ago (Burgers' equation) -- that one needs the computers to compare
  notes about the fastest-moving point in the *whole* simulation before
  each step, which none of this PR's work does yet. Named, not
  forgotten.
- Only reports a single run-through-to-completion time -- it doesn't
  yet separate out "how much of that time was computing" from "how
  much was computers waiting on each other," beyond what the previous
  PR's own standalone communication-only timing already measured.
