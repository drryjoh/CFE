# Clocking a Shock Wave

*Progress review for task 0013. Every shock wave moves at a specific,
predictable speed -- physics tells us exactly how fast, given the gas
conditions on either side. This PR builds the first version of a tool
that watches our simulation, automatically finds where a shock wave
currently is at each moment, and measures how fast it's actually
moving -- then checks that measured speed against the speed physics
predicts. This is also a deliberate first step toward a much bigger
future capability: simulations that can follow a moving feature forever
without needing an ever-growing grid.*

---

## Slide: Why "where is the shock, and how fast" matters

- So far, our simulations have mostly watched a FIXED patch of space
  and let waves move through it. That works fine as long as the
  interesting thing happening stays inside that patch the whole time.
- But some of the most important future simulations (a flame or
  detonation racing down a long pipe, for example) involve a feature
  that just keeps moving in one direction, forever -- eventually running
  off the edge of any fixed patch of space you pick.
- The eventual fix (not built yet, but what this task is a building
  block for) is a simulation that can "follow" the moving feature:
  periodically check where it is and how fast it's going, then shift
  the whole simulated window to keep up, like a camera panning to keep
  a running sprinter in frame. Before building that shifting-camera
  machinery, we need the "check where it is and how fast" part working
  and verified on its own -- which is what this task does.

---

## Slide: A clean, simple test case on purpose

- We set up the simplest possible version of "a shock wave moving
  through gas": a single shock traveling into perfectly still, uniform
  gas, with nothing else going on nearby.
- We deliberately did NOT reuse our existing shock-tube test for this,
  even though it already has a shock wave in it. That test's shock sits
  right next to a different kind of wave (a "contact surface," the
  boundary between two different batches of gas) that our two
  formulas smooth out by different amounts -- which could make it look
  like the shock itself was moving at two different speeds, when really
  it was just the measuring method getting confused by a NEARBY,
  unrelated effect. A clean, standalone shock avoids that risk entirely
  and is simpler to set up besides.

---

## Slide: How we find "where is the shock" automatically

- At each snapshot in time, we scan across the simulated gas looking
  for the point where the density crosses exactly halfway between its
  value far behind the shock and its value far ahead of it -- this
  gives one precise position per snapshot.
- We specifically chose this "halfway crossing" method over an
  alternative (finding the single steepest point) because our two
  formulas smear the shock into slightly different shapes -- the
  halfway-crossing method still lands in close to the same place either
  way, while the steepest-point method could get pulled toward whichever
  shape happens to be a little lopsided. In other words: we picked the
  measuring method least likely to fool itself into reporting a fake
  difference between the two formulas.
- With a position at each of several snapshots in time, a straight-line
  fit through all of them gives us the measured speed.

---

## Slide: The result -- physics predicted it, and we measured it

- For our chosen shock strength, physics (a 19th-century result called
  the Rankine-Hugoniot relations -- the same math that tells you a sonic
  boom's shock speed from the airplane's speed) predicts an exact
  speed: 2.366432 (in our simulation's own units).
- Running our tracking tool at several levels of simulation detail, the
  measured speed landed closer and closer to that predicted number as
  we added detail -- from about 0.4% off at the coarsest level tested
  down to under 0.002% off at the finest.
- Both of our smoothing formulas (Rusanov and HLLC) converged to
  essentially the SAME measured speed, agreeing with each other to
  within a few thousandths of a percent at every level of detail. This
  is exactly what physics says should happen (a real shock wave has one
  true speed, regardless of which approximate formula you use to
  simulate it) -- and we checked it directly rather than just assuming
  it would work out.

---

## Slide: What's next

- The position-finding and speed-fitting tools built here are written
  to be reused directly by the future "shifting camera" simulation
  (described above) when that gets built -- this task's whole point was
  to get that piece working and verified in isolation first.
- Both formulas agreeing on shock speed, independently measured rather
  than assumed, is one more small piece of confidence that our two
  "smoothing formula" options are interchangeable where it matters (the
  big, physical answer) and differ only in the fine detail they
  preserve -- exactly the kind of flexibility the project's longer-term
  goal (letting researchers pick different numerical methods for
  different jobs) depends on.
