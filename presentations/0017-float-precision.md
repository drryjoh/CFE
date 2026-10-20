# Checking Our Math Still Works With Half the Digits

*Progress review for task 0017. Computers can store a decimal number two
different ways: a "double precision" number (about 16 correct digits)
or a "single precision" / "float" number (about 7 correct digits).
Float numbers are smaller and, on the right hardware, noticeably
faster to crunch -- which matters a lot once we start running on GPUs.
This PR checks, rigorously, whether our simulations still give correct
answers when we switch every number in the simulation from double
precision down to float -- and measures how much faster that actually
makes things, right now, on the computer we have.*

---

## Slide: Why fewer digits can mean more speed

- Every number our simulation stores and crunches -- density, pressure,
  velocity, at every grid cell, every timestep -- has to live in
  computer memory and get pushed through the processor.
- A "double precision" number takes 8 bytes of memory; a "float" (single
  precision) number takes only 4 -- half as much. Less memory to move
  around generally means less time spent moving it, and some hardware
  (GPUs especially) can do float arithmetic noticeably faster than
  double arithmetic.
- Our code was DESIGNED from day one to support switching between the
  two (a single build setting), specifically so we could eventually
  chase this speed advantage. This task is the first time anyone
  actually checked, end to end, whether that promise was real.

---

## Slide: A promise that had quietly gone unchecked

- When we actually built our full test suite in float mode and looked
  closely, we found something subtle: several of our most important
  tests were SECRETLY still running in double precision, even when we
  told the whole project to build in float mode.
- Why: a test file can accidentally "pin" its own numbers to double,
  overriding the project-wide setting -- like a recipe that calls for
  "today's sugar" but one cook keeps a private jar of their own sugar
  under the counter and uses that instead, no matter what's in the
  pantry. The recipe LOOKS like it's using the pantry's sugar; it
  isn't.
- This wasn't sloppiness -- some test files deliberately and correctly
  pin double on purpose (to get an exact answer to compare against), and
  that's the right call there. The problem was specific files that
  pinned double WITHOUT meaning to, so our most important physics tests
  (a shock tube, a 3D sanity check, the forward-facing-step assembly)
  never actually got float-tested at all, despite looking like they did.

---

## Slide: Finding the real, trustworthy tolerance -- not guessing

- Every physics check compares a computed answer to an expected one
  within some allowed slack ("tolerance") -- e.g. "within one part in a
  billion." Those slack values were originally tuned tight for double
  precision's ~16 digits. Float only has ~7, so some checks needed more
  slack to avoid failing for no real reason.
- Instead of guessing a bigger number, we built a small standalone
  measurement tool that runs the EXACT same physics the real tests run,
  at both precisions, and prints the real difference. Only then did we
  pick a new tolerance -- always with a healthy safety margin over the
  measured number, and always explained in a comment so a future reader
  can see the reasoning, not just a magic number.
- The genuinely interesting finding: MOST checks needed no loosening at
  all. The float and double answers came out nearly identical in most
  cases -- because the error in these simulations mostly comes from
  using a finite grid (a few hundred cells instead of infinitely many),
  not from the number format itself. Only a handful of checks that rely
  on near-perfect bookkeeping (e.g. "total mass must stay EXACTLY
  constant") needed real loosening, and even those still pass with
  10-200x safety margin to spare.

---

## Slide: Does switching to float actually make things faster here?

- We measured it directly, on this machine's own CPU (no GPU available
  here tonight -- that comparison is still ahead of us, flagged as the
  natural next step below).
- Result: it depends on what kind of calculation is being done.
  - A cheap, simple formula for how gas should flow across a cell
    boundary (nicknamed "Rusanov") showed only a modest speedup from
    float (about 4% faster) -- it's a short calculation, so on this
    machine its cost is apparently not dominated by raw number-crunching
    the way a heavier calculation's is.
  - A more elaborate, more accurate formula ("HLLC") -- which does a lot
    more arithmetic (several square roots and branching decisions) per
    cell -- ran about 35% faster in float. That's a real, measured ~1.5x
    speedup, not a guess.
- Takeaway: float's speed advantage isn't automatic everywhere -- it
  shows up most where the calculation itself is the bottleneck, which is
  exactly the situation a GPU (this project's real target) puts every
  calculation in. This is an honest, partial result, reported as
  measured rather than oversold.

---

## Slide: What's next

- The actual GPU-vs-CPU, float-vs-double comparison this project cares
  about most still needs real GPU hardware, which wasn't available for
  this task -- the same standing limitation several recent tasks have
  flagged. ROADMAP.md already calls for exactly this "single/double
  precision study" as a Phase 6 deliverable; this task lays the tested
  groundwork for it rather than completing it.
- Going forward, we've made this a standing project rule: any future
  change to the physics or math itself must be checked at BOTH
  precisions before it's considered done -- the same rule this project
  already has for testing new work on a real GPU. That way, this
  specific gap (tests quietly running in the wrong precision) can't
  quietly reappear.
