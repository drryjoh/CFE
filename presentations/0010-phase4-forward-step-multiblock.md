# Phase 4: A Wind Tunnel With a Step In It

*Progress review for task 0010. The previous PR (Phase 3) proved the
compressible-gas simulation worked on a simple, straight tube. This PR
puts that same simulation into a more realistic-shaped room -- a wind
tunnel with a step in the floor partway down -- which turns out to need
a genuinely new piece of engineering, not just a new shape of box.*

---

## Slide: What problem this solves

- A classic, famous test case in this field: blow air at three times
  the speed of sound (Mach 3) down a tunnel that has a step in the
  floor. What happens where the air hits that step?
- The answer involves several things happening together: a curved
  shock wave peeling off the step's corner, that shock bouncing off the
  tunnel's ceiling and creating a second, more complicated kind of
  reflection (a "Mach stem") that grows longer the longer the
  simulation runs, and a faint trailing boundary layer behind it all.
- This is a 1984 benchmark specifically because it's hard enough to
  separate "simulations that are actually good" from "simulations that
  only look good on easy problems" -- a step's sharp corner is a known
  stress test for this kind of code.

---

## Slide: Why a stepped room needed a new trick, not just a new wall

- Every simulated "room" so far in this project has been a plain
  rectangular box (a grid of cells, all the same shape). A tunnel with
  a step in the floor is NOT a rectangle -- it's an L-shape.
- The chosen solution: use TWO separate rectangular rooms instead of
  one -- one for the taller section before the step, one for the
  shorter section after it -- and build a new kind of "shared wall"
  between them that, instead of being solid, lets the gas pass through
  almost like it's one continuous room.
- Picture two side-by-side rooms with a connecting doorway that's only
  as tall as the shorter room -- gas flows freely through the doorway,
  but where the doorway doesn't reach (the step itself), that's just a
  real, solid wall. The new piece of code built for this PR is exactly
  that doorway/wall combination.
- The rejected alternative (one single room with some of its floor
  tiles marked "pretend this doesn't exist") was considered and turned
  down: it would have meant teaching the core simulation engine about
  the room's SHAPE, which this project deliberately keeps separate from
  the engine itself (the engine should never need to know anything
  about the specific problem being solved) -- a consequential design
  choice confirmed with this project's own architecture reviewer before
  writing any code.

---

## Slide: Proving the "doorway" doesn't leak or double-count gas

- A shared wall between two separately-simulated rooms is exactly the
  kind of place a subtle bug could quietly create or destroy gas that
  the simulation never notices. This needed a very direct check, not
  just "the simulation ran and the picture looked right."
- The check: compute how much gas is flowing through the doorway, calculated
  independently from EACH room's own point of view, and confirm the two
  numbers agree -- not approximately, but to the limits of the
  computer's own arithmetic precision.
- They agreed. This is the strongest, most literal evidence available
  that the doorway mechanism is correct, not just "probably fine."

---

## Slide: A wrinkle from running two rooms as one program

- Each room is simulated using a standard two-step "predict, then
  correct" time-stepping recipe. With two separate rooms feeding each
  other through their shared doorway, a subtlety appeared: room A's
  "correct" step needs room B's "predict" result through the doorway
  -- not room B's data from a step ago. Getting this ordering wrong
  would silently desynchronize the two rooms' solutions, a hard bug to
  spot just by looking at a picture.
- This was handled by explicitly interleaving the two rooms' steps by
  hand (compute both rooms' "predict" step first, then re-point the
  doorway at the fresh predict-step data before computing either room's
  "correct" step) rather than assuming the existing single-room
  stepping code would handle it automatically -- it wouldn't have.
- Checked directly: ran both rooms together for a short stretch and
  confirmed nothing went non-physical (no negative gas density,
  no runaway numbers) -- a basic sanity net before trusting any of the
  bigger, longer runs.

---

## Slide: What the actual pictures show

- Ran the simulation at three different levels of detail (the same
  three the original 1984 paper used, for a fair comparison), each one
  all the way to the point where the flow stops changing much over
  time.
- All three levels of detail show the SAME story: a curved shock
  peeling off the step's corner, bending backward and up; that shock
  hitting the ceiling and creating the expected more-complex reflection
  pattern, whose meeting point visibly creeps back UP the tunnel
  (toward the step, not away from it) the longer the simulation runs --
  measured directly from the output, not just eyeballed -- which is
  exactly what the original 1984 paper itself reports happening; and a
  much fainter trailing line behind all of it.
- The finer the detail level, the sharper and cleaner all of this
  looks -- the coarsest run shows a blurry smear where the sharpest run
  shows a crisp, clearly separated structure. That's the expected,
  healthy signature of a scheme that's working correctly, becoming more
  accurate as it's given more resolution to work with.
- One known, honestly-reported wrinkle: right at the step's own sharp
  corner, a small extra feature shows up in the pictures that isn't
  part of the main shock pattern. The step's corner is a genuinely hard
  mathematical point (where the exact answer has its own sharp, special
  behavior), and a handful of published papers on this exact benchmark
  describe dedicated extra fixes specifically for that corner. Building
  that specific fix was intentionally left out of this PR's scope --
  the wrinkle is called out plainly here rather than hidden, and does
  not visibly spread beyond the immediate corner area.

---

## Slide: Did this also work in 3D, and what's still outstanding

- Also ran the whole case as a thin 3D slab (not just a flat 2D
  picture) on this project's regular laptop-class processor, to prove
  the two-room mechanism survives the jump from 2D to 3D.
- Checked something specific: since nothing in this particular 3D setup
  should ever cause motion sideways (into the depth of the slab), any
  sideways motion that appeared would be a red flag. None did -- every
  single measurement of sideways motion came back at EXACTLY zero, and
  a slice through the middle of the 3D run matched the plain 2D run
  number-for-number, not just "looked similar."
- Still outstanding, named plainly rather than quietly skipped: this
  project has a separate, standing rule that any new simulation feature
  must also be proven correct and fast on a real graphics-card
  processor (GPUs, the specialized hardware this project uses for its
  biggest, fastest runs), using a shared research computing cluster
  this project has access to remotely. This laptop has no such
  hardware. That GPU-specific check is real future work, not finished
  here, and is written down as the clear next step.
