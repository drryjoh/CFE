# Finding the Missing Wave Pattern

*Progress review for task 0011. The previous PR (Phase 4) got a
simulated Mach-3 wind tunnel working, including a step in the floor.
It showed the big, obvious features correctly -- but one subtle,
specifically-named feature from the original 1984 paper was stubbornly
missing, no matter how much detail we threw at the simulation. This PR
explains what that missing feature was, why it was missing, what we
changed to go looking for it, two real mistakes we caught and fixed
along the way, and what we actually found.*

---

## Slide: The mystery we started with

- Our wind-tunnel simulation (last PR) got the big picture right: a
  curved shock wave peeling off a step, a secondary reflection pattern
  forming and sliding upstream, exactly like the real experiment and
  the original 1984 paper describe.
- But the paper also describes one smaller, specific thing: a very
  mild "flag-waving" wobble on a trailing boundary line in the flow,
  visible at a particular point in time. We ran our simulation at
  resolutions far finer than the paper's own best result and never saw
  even a hint of it.
- Was this a real gap in our simulation, or just something that needed
  even more detail to show up? We couldn't tell yet -- we needed a
  different piece of the puzzle to find out, not just a finer picture
  of the same piece.

---

## Slide: Two different ways to "smooth over" a sharp jump

- Under the hood, every cell-by-cell simulation like this has to decide
  what happens at the boundary between two neighboring cells with very
  different values -- it can't just teleport information instantly, so
  it uses a formula (a "numerical flux") to figure out what crosses
  that boundary each tiny time step.
- The formula we'd been using (nicknamed "Rusanov") is simple and
  robust, but it's deliberately cautious: it smooths out EVERY kind of
  boundary by the same, fairly generous amount, including the one kind
  of boundary (a "contact surface" -- like the edge between two
  different colors of smoke, with no pressure difference across it)
  that needed to stay sharp for that wobble to ever show up and grow.
- So before this PR, we added a second formula ("HLLC") that treats
  that one kind of boundary specially -- letting it stay much sharper --
  while still handling the loud, obvious shock waves the same reliable
  way as before. The plan: keep everything else about the simulation
  identical, swap in only this new formula, and see if the missing
  wobble appears.

---

## Slide: Two real bugs, caught by actually running the harder test

- Before trusting the new formula on the wind-tunnel case, we checked
  it on a much simpler, textbook problem first -- the equivalent of
  testing a new recipe on a small batch before cooking for a crowd. It
  passed there.
- But the FIRST time we ran it on the real, harder wind-tunnel problem,
  the entire simulated wind tunnel froze solid at the very first
  instant and never moved again -- for the whole simulated time, at
  every level of detail. No crash, no error message, nothing. We traced
  it to the formula's own internal math quietly picking the wrong
  answer at exactly the spot where air slams hardest into a wall (three
  times the speed of sound, hitting the step's vertical face) -- like a
  calculator that silently returns "0" instead of an error when asked
  to do something it can't actually handle. Fixed by making that
  internal math iterate a few extra times toward a much more exact
  answer, which we verified against hand-worked exact math for the
  wall-collision case specifically.
- After that fix, a SECOND problem showed up, but only at finer detail
  levels: the simulation would run fine for a while and then suddenly
  produce nonsense numbers partway through. This one traced to a
  different, subtler cause: right where multiple shock waves meet and
  cross, the simulation's own reconstruction step (figuring out what
  each cell's edge value should be) can very occasionally produce an
  almost-vacuum value that's still technically valid but numerically
  fragile -- close enough to "nothing there" that the next calculation
  step can tip it over into actual nonsense. Fixed by adding a small,
  targeted safety floor exactly where this could happen.
- Both bugs were found specifically BECAUSE we ran the harder,
  realistic wind-tunnel test before declaring the new formula "done" --
  the simpler textbook test alone would never have caught either one.
  That's the whole reason this project's rules insist on testing new
  numerical methods against a real, hard case, not just a clean
  textbook one.

---

## Slide: What we found, once both bugs were fixed

- With the new formula working correctly, we re-ran the SAME
  wind-tunnel simulation, at every level of detail already explored,
  changing nothing else.
- **The wobble showed up.** At the finer detail levels, the trailing
  boundary line develops a wavy, crossing pattern that grows more
  pronounced over time -- and it starts becoming visible at almost
  exactly the same point in simulated time the original 1984 paper
  reports for this same effect. The old formula, run at the identical
  settings, never shows even a trace of it, at any level of detail we
  tried.
- This matches the paper's own description well: mild, wavy, not a
  dramatic rolling/curling pattern -- which is exactly what the
  original paper itself found too, even using its own best method. We
  are not claiming more than that, and we checked carefully that we
  weren't fooling ourselves (comparing side-by-side, same detail level,
  same instant in time, with and without the new formula).
- One other thing turned up along the way, worth reporting honestly
  rather than hiding: right at the spot where the shock waves meet (not
  the trailing wobble itself), the new formula produces a small,
  localized value spike that the old formula doesn't -- still a valid,
  finite number, but a real, known side effect of using a less-
  cautious formula combined with how we currently handle
  reconstruction. This was already a named, anticipated risk from
  earlier work, now confirmed with real evidence rather than just
  suspected.

---

## Slide: What's still outstanding

- We did NOT reach for an even bigger, more complex upgrade (a
  higher-order reconstruction scheme called PPM, also mentioned in the
  original paper) before trying this smaller, more targeted fix first --
  and it turned out we didn't need to. That bigger piece of work stays
  on the shelf for later, not because it wouldn't also work, but
  because this smaller fix already answered the question.
- The small value-spike mentioned above is a known, open item -- not
  fixed in this round, tracked for whenever this project takes on a
  bigger upgrade to how it reconstructs values near sharp features.
- As with every new simulation feature in this project, the new formula
  was also verified correctly and bit-for-bit matching on real
  graphics-card hardware (a separate remote research computer with
  GPUs), not just on this laptop -- that check passed cleanly.
- A precise cost comparison (how much more expensive the new formula
  is per calculation) wasn't cleanly isolated yet -- our timing numbers
  came from one long run, not a tightly controlled repeated experiment,
  so we're reporting the honest direction (the new formula costs more)
  without overstating a precise number from data that could be
  muddied by the computer's own background noise during a long run.
