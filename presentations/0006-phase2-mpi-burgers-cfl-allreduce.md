# Phase 2: Making the Self-Steepening Wave Equation Safe to Split Up

*Progress review for task 0006. Two PRs ago, splitting a simulation*
*across several computers was built and proven correct for the*
*"simple" equation (something drifting at a constant speed). That same*
*PR explicitly flagged one equation it did NOT yet make safe to*
*split: Burgers' equation, the one that steepens into a shock. This PR*
*closes that gap.*

---

## Slide: Why the self-steepening equation needed something extra

- Every computer in a split-up simulation has to agree, every single
  step, on exactly how big a time-step to take. If they don't agree,
  they fall out of sync with each other.
- For the "constant speed" equation, picking a safe step size is easy:
  the speed never changes, so every computer can work it out on its
  own and they'll all agree automatically.
- Burgers' equation is different: the step size has to be based on the
  **fastest-moving point anywhere in the whole simulated world** --
  and if the work is split up, no single computer can see the whole
  world. Each one can only see its own slice.

---

## Slide: The fix -- ask everyone, then agree

- The fix is a classic, simple pattern: every computer first checks
  the fastest point *it* can see, then all of them send that number to
  each other and agree on the overall largest one -- a "what's the
  biggest number anyone has?" poll, taken once before the step size is
  chosen.
- This is a single new, small, reusable building block
  (`allreduce_max`) -- not a redesign of anything. Every other file in
  the project that was already working stayed untouched.

---

## Slide: Proving it actually matters -- and a bigger surprise than expected

- Built a test specifically designed so the bug would actually bite:
  a wave shape where the fastest point sits at one specific location,
  deliberately NOT centered on any one computer's slice -- so at least
  two of the computers would, without the fix, see a noticeably
  smaller "fastest point" than the true one (about a third smaller).
- To make sure the test wasn't secretly toothless, the fix was
  temporarily removed and the test re-run.
- **What happened was worse than a wrong answer -- the whole run froze.**
  Because the step size comes from the "fastest point" number, and
  that number was now different on different computers, the computers
  ended up disagreeing not just on step SIZE but on how many steps
  total were needed. One computer finished and stopped talking, while
  its neighbor was still waiting to hear from it -- and since neither
  one was going to speak first, they waited forever.
- This is actually a more convincing argument for the fix than a
  subtly-wrong-number would have been: without it, this isn't a small
  numerical error to squint at, it's a guaranteed hang.

---

## Slide: What this still doesn't do

- Only proven in 1D so far -- the same fix should extend cleanly to
  the already-built 3D decomposition, but that hasn't been re-verified
  with Burgers specifically yet.
- Doesn't yet help a non-looping (true edge-of-the-world) domain split
  across computers -- still the same open item named two PRs ago.
- Doesn't touch GPUs -- still CPU-only, same as every PR in this MPI
  series so far, since a GPU allocation still hasn't become available.
