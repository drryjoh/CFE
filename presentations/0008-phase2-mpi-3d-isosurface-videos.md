# Phase 2: Videos of the Two 3D Simulations

*Progress review for task 0008. The last two PRs produced numbers and*
*a handful of static pictures from the two 3D split-across-computers*
*simulations. This one adds something easier to actually look at: a*
*short video of each one, filmed from outside the simulated box.*

---

## Slide: What's new here

- Both 3D simulations now automatically produce a short video (an MP4)
  showing a 3D surface -- think of it like a weather map's "cloud
  boundary" line, but in 3D -- as the simulation runs, viewed from
  outside the simulated cube with the camera slowly circling around it.
- No manual work required: point a small script at the simulation's
  saved output files and it builds the whole video by itself.

---

## Slide: Side by side, the difference is obvious

- The "simple" simulation (drifts at a constant speed): the surface
  stays a perfect sphere the whole video, just sliding across the box
  and wrapping around the edge once it reaches the far side.
- The "self-steepening" simulation (Burgers' equation, from three PRs
  ago): the SAME starting sphere visibly warps over time -- bulging,
  flattening, developing sharp faceted edges on one side -- because
  that equation's whole point is that different parts of the material
  move at different speeds and catch up with each other.
- Watching both videos back to back makes that difference immediately
  obvious in a way a written description or a single still picture
  can't.

---

## Slide: Running the self-steepening one longer

- Checking the actual math, the self-steepening simulation's previous
  run length only *just barely* crossed the point where a real "shock"
  (a sudden jump, not a smooth slope) first forms -- like stopping a
  video right as something interesting starts, not letting it play out.
- Lengthened the run (about 3x further past that threshold) so the
  video actually shows a clearly-formed, fully-developed shock, not a
  borderline one -- re-confirmed the simulation still obeys its own
  promise (never inventing a value higher or lower than what started
  in the scene) at this new, longer length before trusting the result.
- This did mean re-measuring the scaling numbers from the last PR,
  since a longer run takes longer to finish -- done, same honest
  "laptop first, then the real supercomputer" reporting as always.

---

## Slide: One real, interesting detail worth explaining

- In the self-steepening video, watch the corners of the box: small
  fragments of the surface appear and grow there over time. This looks
  at first like a rendering glitch -- it isn't.
- This simulation's background "canvas" isn't perfectly still -- under
  this particular equation, even the flat background drifts over time.
  Since the box wraps around at its edges (what goes off one side comes
  back on the opposite side), the drifting background eventually carries
  part of the feature across that wraparound point, and it reappears on
  the far side -- hence the fragments growing in the corners.
- This is the same "wraparound" correctness this project's tests
  already numerically verified happens correctly between separate
  computers sharing the work -- this video is a visual bonus
  confirmation that it also happens correctly at the edges of the whole
  simulated world, not just between computers.

---

## Slide: What this still doesn't do

- No new correctness proof -- this PR is pictures and tuning, built
  entirely on top of what the last three PRs already proved
  numerically.
- The "simple" simulation's own run length was left unchanged (only
  the self-steepening one needed lengthening, since only it has a
  "shock forms over time" story to tell).
