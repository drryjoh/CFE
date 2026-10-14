# A Shock Moving Through Ripples

*Progress review for task 0012. Our wind-tunnel simulation (prior PRs)
and our shock-tube simulation (task 0009) have each separately shown
our two "smoothing formulas" (Rusanov and HLLC) behaving differently on
sharp features. This PR runs a THIRD, independent classic test --
a shock moving through a rippled gas -- designed specifically to answer
the question "does one formula really preserve fine detail better than
the other, or did we just get lucky twice?" It also turned up a genuine
surprise nobody predicted going in.*

---

## Slide: The test -- a shock plowing through ripples

- Picture a long tube of gas where the density (how tightly packed the
  gas molecules are) gently ripples up and down like a sine wave --
  dense, then thin, then dense again, over and over.
- Now send a strong shock wave (three times the speed of sound) into
  one end, barreling through that rippled gas.
- What happens behind the shock is genuinely complicated: the shock
  doesn't just flatten the ripples, it interacts with them, creating a
  tangle of smaller waves that keep bouncing around. This is a famous
  1989 test (Shu & Osher) specifically because there is no simple
  formula for the "right answer" here -- unlike our shock-tube test,
  where we can calculate the exact answer by hand, this one is too
  tangled for that. The only way to check our simulation is to run it
  at much higher detail and see whether a coarser run agrees with a
  much finer one.

---

## Slide: What we expected to find (and did)

- We already knew (from the wind-tunnel PR and the shock-tube PR) that
  our sharper formula (HLLC) tends to preserve fine detail better than
  our simpler, more cautious formula (Rusanov) -- but each of those
  tests was its own special case. This test is a genuinely different,
  independent check of the same idea.
- Running both formulas at several levels of detail and comparing each
  one against its OWN much-finer reference run, HLLC consistently
  stayed closer to its reference than Rusanov did, at every level of
  detail we tried -- a modest but real and consistent advantage, the
  same direction as before, now confirmed a third time rather than
  assumed.
- We also counted the actual number of ripples each formula's coarse
  run managed to capture, compared to the fine reference: HLLC
  captured as many or more ripples than Rusanov at every detail level
  tested.

---

## Slide: The surprise nobody predicted

- Ahead of where the shock has reached, the gas is just sitting there
  with its ripples, completely undisturbed -- in fact, we can prove
  mathematically that this part of the gas should stay EXACTLY
  unchanged over time, with no approximation needed, until the shock
  physically arrives.
- We expected both formulas to get this part essentially perfect, since
  there's nothing hard happening there. HLLC did -- it reproduced the
  untouched ripples to the full precision a computer can represent, at
  every level of detail we tried.
- Rusanov did NOT. It slowly, measurably smeared out the ripples even
  in this supposedly "nothing is happening" region -- a real, visible
  effect, not a rounding error. We traced this to exactly how Rusanov's
  own smoothing formula works: it always smooths a LITTLE, proportional
  to how different two neighboring cells are, even when the actual flow
  physics says nothing should be crossing between them at all. HLLC's
  formula, by contrast, happens to recognize this exact situation (no
  flow, same pressure on both sides) and correctly applies zero
  smoothing. This is arguably a clearer, more visually striking
  demonstration of HLLC's advantage than the ripple-counting result
  above.

---

## Slide: A mistake we caught before publishing a wrong number

- When we first measured "how much does Rusanov smear the untouched
  ripples," the number looked like it was barely shrinking as we added
  more detail -- which would have been a worrying, hard-to-explain
  result.
- Our independent reviewer dug in and found the real story: almost all
  of that measured smearing was coming from one small strip right at
  the far edge of our simulated tube, not spread evenly through the
  "untouched" region. The edge of any simulated tube needs a boundary
  rule saying "what's just past the edge," and the simple rule we used
  there (copy the nearest cell) can't perfectly represent a RIPPLING
  edge value the way it can a perfectly flat one -- so it introduces
  its own small, separate error, right at that one edge, on top of the
  smearing effect we were actually trying to measure.
- Once we excluded that one small edge strip and re-measured just the
  genuine smearing effect in the rest of the "untouched" region, the
  result became exactly what we'd expect: a clean, steadily shrinking
  error as we added more simulation detail. We kept BOTH numbers
  visible in our final report (the genuine smearing effect, and the
  separate edge effect) rather than hiding either one -- see the
  updated figure, which now marks exactly where one effect ends and the
  other begins.

---

## Slide: What's next

- This result, the wind-tunnel result, and the shock-tube result now
  all independently point the same direction: our sharper formula
  (HLLC) consistently preserves fine detail better than our simpler one
  (Rusanov), across three genuinely different test problems.
- The edge-strip effect found above is real but small and specific to
  this one test's particular boundary setup -- not investigated further
  in this PR, but recorded for whoever next needs an edge rule that can
  represent a varying (not just constant) value at a tube's far end.
- Next up: the companion "1D propagating shock" task (0013), which
  reuses several of the measurement tools built for this one.
