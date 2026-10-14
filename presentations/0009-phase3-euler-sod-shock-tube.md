# Phase 3: Simulating a Real Explosion in a Tube

*Progress review for task 0009. Every simulation so far has pushed
around one number per point in space (a concentration, a "how fast is
it piling up" quantity). This PR adds a genuinely different kind of
simulation -- a compressible gas, the physics behind real explosions,
sonic booms, and rocket nozzles -- where each point in space now needs
*multiple* numbers at once, and they all depend on each other.*

---

## Slide: What problem this solves

- Picture a tube with a thin wall in the middle: high-pressure gas on
  one side, low-pressure gas on the other. Pull the wall away instantly
  -- what happens next?
- Three things happen at once, moving at three different speeds: a
  sharp shock wave races into the low-pressure gas, a smooth "fan" of
  gently-dropping pressure spreads into the high-pressure gas, and a
  boundary between the two original gases (like a line of dye marking
  which side a given bit of gas started on) drifts along in the middle.
- This is one of the most famous test problems in all of fluid-dynamics
  software (the "Sod shock tube," 1978) specifically because the exact
  answer is known in closed-form math -- so it is a trustworthy way to
  check a new simulation is actually right, not just "runs without
  crashing."

---

## Slide: Why this needed a new kind of simulation, not just a new equation

- Every previous simulation in this project tracked ONE number per grid
  cell (like one temperature reading per thermometer). This one tracks
  FOUR or FIVE per cell at once -- density, "how much stuff is moving
  and which way" (in up to 3 directions), and total energy -- and
  critically, they're coupled: you can't update one without knowing all
  the others (e.g. computing pressure needs density AND energy AND how
  fast things are moving, together).
- The simulation "engine" that was built for the single-number case
  (fill in neighbors, do the math, advance one step in time) had to be
  taught to do the same recipe for a whole bundle of numbers per cell
  at once, without becoming four times slower or four times more
  complicated to read -- a core piece of engineering work this PR had
  to get right before the physics could even be tried.
- Confirmed this didn't come at a hidden cost: inspected the actual
  low-level instructions the computer ends up running, and the
  single-number simulations (used by every earlier PR) compile down to
  the EXACT same instructions as before -- the new, more general engine
  costs those older simulations nothing.

---

## Slide: A new kind of "wall" at the edges

- Every boundary condition before this PR was "dumb" in a specific,
  deliberate sense: it just copied or fixed raw numbers, with no idea
  what those numbers physically meant.
- A solid wall for a real gas needs something smarter: the gas can
  slide freely ALONG the wall, but cannot pass THROUGH it. That means
  the boundary condition has to know which one of its several numbers
  per cell represents "motion toward the wall" specifically, so it can
  flip just that one and leave the rest alone.
- This is the first boundary condition in the project that actually
  has to understand a little bit of the physics it's enforcing, rather
  than just copying numbers around -- a deliberate, reviewed design
  decision (not an accident), kept as narrow as possible (it still
  needs no more physics knowledge than "which number is motion toward
  me").

---

## Slide: How this was checked, not just "it ran"

- Built an independent, from-scratch reference calculator for the
  exact math answer to the shock-tube problem (not copied from the
  simulation itself), then cross-checked ITS numbers against a second,
  completely separate calculation and against the published textbook
  answer for this exact problem, before trusting it as the referee.
- The simulation's result was then checked against that independent
  referee in several ways: does the total amount of gas and energy stay
  exactly constant (it must -- nothing enters or leaves the tube in
  time); does the total "push" on the gas match the exact, hand-
  calculable force from the pressure difference; and does the simulated
  shock/fan/boundary line up with the exact math at several spots along
  the tube.
- One known, expected imperfection, found and explained rather than
  hidden: the simplest wave-recipe used here smears out the
  dye-line-like boundary between the two gases more than it smears the
  actual shock -- a textbook, well-understood property of this specific
  recipe (not a bug), with a sharper recipe named as clear future work.

---

## Slide: What's deliberately NOT in this PR

- The user's original ask also included a second, much bigger benchmark
  -- a wind-tunnel simulation with air flowing past a physical step,
  famous in the field for the complicated shock patterns it produces.
  That benchmark needs an L-shaped tube, which this project's current
  simulated-grid machinery can't represent directly (it only knows
  rectangular boxes so far).
- Rather than bolt a brand-new, untested "join two boxes together"
  mechanism onto brand-new gas physics in the same change, the project
  direction reviewer was consulted first and recommended splitting the
  work: ship the tested, verified gas physics now, build and review the
  box-joining mechanism as its own separate, focused piece of work next
  -- so if something ever goes wrong later, it's possible to tell which
  of the two was at fault.

---

## Slide: Finishing the story -- running it on an actual GPU

- The GPU testing named as "the clearly-named next step" didn't wait
  for a separate PR: before this one was finalized, it was run for real
  on a supercomputer (PSC Bridges-2, a shared research computer with
  NVIDIA GPUs built for exactly this kind of large simulation).
- Quick reminder of the vocabulary: a GPU (graphics processing unit) is
  a chip built to do the SAME small calculation on thousands of pieces
  of data at once, rather than one calculation very fast the way a
  regular computer chip (CPU) does -- simulating a tube full of
  millions of gas cells, where every cell does the identical update
  recipe, is exactly the kind of job a GPU is built for.
- Before trusting any speed number, correctness came first: the GPU was
  told to solve the EXACT same shock-tube problem as the regular
  computer (CPU) version already proven correct, and the two answers
  were compared cell by cell. They matched to 9 decimal digits, in both
  the simple 1-direction tube AND a fuller 3-direction version. A
  separate safety-checking tool (`compute-sanitizer`) that watches for
  "reading memory before it was ever written" also came back completely
  clean.
- With correctness confirmed, the actual speed test: the GPU chewed
  through a tube of 100 million cells, one full time-step, in about 61
  milliseconds -- roughly 1.6 billion cell-updates every second. The
  fuller 3-direction version (134 million cells arranged in a cube)
  ran at about 350 million cell-updates per second -- slower, because
  each cell now carries 5 numbers instead of 3 and every one of the
  cube's three directions does real work every step, not just one.
- One specific worry going in was checked, not assumed away: this gas
  simulation tracks more numbers per cell than the earlier, simpler
  simulations, so the largest version was a real risk of running out of
  the GPU's memory. It was watched run start to finish rather than
  guessed at -- it fit, using about half the GPU's 32 GB of memory.
- Also not yet done (same honest "what's next" framing as every prior
  PR): a GPU-occupancy profile (a deeper "how efficiently is the GPU's
  hardware actually being used" measurement, beyond the pass/fail and
  speed numbers already gathered) and a version using the GPU's faster,
  lower-precision number format (`float` instead of `double`) --
  reasonable future work, not required for this PR's own claims.
