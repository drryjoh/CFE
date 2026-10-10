# Phase 2: Watching the Self-Steepening Wave Scale and Run Correctly

*Progress review for task 0007. Last PR made the self-steepening wave*
*equation (Burgers' equation) safe to split across several computers.*
*This PR is the payoff: watch it actually run that way, and measure*
*whether splitting it up helps.*

---

## Slide: What's new here

- Same split-across-several-computers trick as two PRs ago, but now
  applied to the equation that forms real shocks -- not the simple
  "drifts at constant speed" one.
- Produces both a picture (a bump of material steepening into a shock
  on one side while spreading out on the other, split across however
  many computers ran it) and a number (how much faster it finishes with
  more computers helping).

---

## Slide: This is where the last PR's fix actually gets used for real

- Last PR's fix was: before picking a step size, ask every computer
  "what's the fastest point you can see?" and agree on the overall
  largest answer.
- This PR's bump is a perfect, non-contrived example of exactly why
  that matters: the bump's peak -- the single fastest point in the
  whole simulation -- sits inside only ONE computer's own slice (or a
  couple, once there are more slices). Every OTHER computer, looking
  only at its own slice, would see a noticeably slower "fastest point"
  than the true one.
- Confirmed directly: the true peak is about 1.499; most individual
  slices, on their own, only reach about 1.15-1.23 at that same moment.
  Without last PR's fix, this exact scenario is precisely the one that
  would make different computers disagree and freeze the whole run.

---

## Slide: Proving it's not just numerically correct, but correct in a way you can verify yourself

- The last PR's test already PROVED (with an exact number-for-number
  check) that this works. This PR adds something that test alone
  doesn't give you: an actual picture you can watch.
- Checked, across every single saved picture frame on every computer
  involved -- not just a spot check -- that the material's value never
  went above its starting peak and never dropped below its starting
  floor. This is a known promise of the shock-capturing method the
  project already uses (it's not supposed to ever invent a new high or
  low that wasn't in the starting picture), now confirmed to hold even
  when the simulation is split across several computers.
- Also re-confirmed (same check as two PRs ago) that all 8 computers'
  individual picture pieces fit together into one seamless whole, no
  gaps, no overlaps.

---

## Slide: Does it scale?

**On the real supercomputer, yes -- nearly perfectly:**

| Computers used | Time to finish | Speedup |
|---|---|---|
| 1 | 252.1 s | 1.0x |
| 2 | 125.8 s | 2.00x |
| 4 | 64.2 s | 3.92x |
| 8 | 34.4 s | 7.33x |

Doubling the computers very nearly halves the time, every step up to
8 -- 100% of the theoretical best-possible benefit realized at 2
computers, still 92% at 8. The same excellent result the first scaling
test (two PRs ago) found, now confirmed for the equation that actually
needed the collective-agreement fix to be safe at all.

**On a laptop (10 cores, shared with everything else running on it),
the same familiar falloff from two PRs ago shows up again:**

| Computers used | Time to finish (this laptop) |
|---|---|
| 1 | 62.6 s |
| 2 | 32.4 s |
| 4 | 18.8 s |
| 8 | 24.8 s (slower than 4 again) |

- Good up to 4, then worse at 8 on this shared laptop -- expected, for
  the same reason as before (talking-to-neighbors overhead growing
  relative to shrinking individual workloads) -- and now directly
  confirmed by the supercomputer numbers showing that same effect
  barely registers on dedicated hardware.
- This equation does more work per point than the simple one from two
  PRs ago (checking for a possible shock at every point, not just
  moving a number along), so every number here is bigger across the
  board on both machines -- but the shape of each machine's own curve
  matches its counterpart from two PRs ago closely.
- One side-note worth flagging plainly: the supercomputer's single
  computer alone was actually SLOWER than the laptop's single computer
  for this specific task (252s vs. 63s) -- a good reminder that a
  supercomputer's value is having many dependable, evenly-matched
  computers that scale predictably together, not necessarily having the
  single fastest computer. What matters for this slide is each
  machine's own speedup curve, not a cross-machine race.

---

## Slide: What this still doesn't do

- Both the laptop and real supercomputer numbers are now in (see
  above) -- this result is complete, not a placeholder.
- Doesn't add any new correctness proof beyond what the previous PR
  already established numerically -- this PR is the picture and the
  speed measurement, not a second proof.
- Only proven for this one way of splitting the work into bricks (a
  roughly cube-shaped arrangement, chosen automatically) -- other
  arrangements, or a much larger number of computers across multiple
  physical machines, remain untested.
