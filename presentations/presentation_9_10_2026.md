# Group Meeting Talking Points — 2026-09-10

Discussion notes to go with `presentations/0001-phase0-execution-foundation.md`
(the slide deck / `slides.pptx`) for today's group meeting. The deck carries
the "what happened" story; these are the points worth spending live
discussion time on rather than just presenting.

## Show quickly, don't dwell

- Headline: the CUDA backend is verified for real on PSC Bridges-2 (27/27
  automated tests passing), and the AoS-vs-SoA memory-layout reversal —
  CPU wants a cell's numbers grouped together, GPU wants them spread out
  by quantity, by up to 33x — is the one finding worth making sure lands,
  since it affects how anyone writes a physics kernel later.

## Worth actually discussing, not just presenting

1. **The Phase 1 scope calls just made.** Fixed-block AMR is deliberately
   deferred but the grid layer is being kept ready for it in roughly two
   months; general unstructured-grid support is deliberately pushed to
   early 2027. If anyone else in the lab has a project that needs either
   sooner, now is the moment to say so, before more code gets built on
   top of these assumptions.
2. **GPU access is PSC Bridges-2, not Orchard.** The original plan assumed
   CMU's Orchard system; that changed. Anyone else needing GPU time for
   their own work should use `docs/bridges2-setup.md` — it covers the
   whole path, including the `gpuinteract` scheduling trick that turned an
   hours-long queue into under a minute.
3. **Preferred verification case for Phase 1's convergence study?**
   `VERIFICATION.md` lists candidates (scalar translation is the obvious
   first one), but if someone in the group has a specific test problem
   they care about for their own research, this is the cheap moment to
   fold it in — before the convergence-study code is written around a
   specific choice.

## One process note worth reinforcing

A test failure that only reproduced on one compiler (GCC, never Clang)
was not waved off as "flaky." It got root-caused down to a specific
compiler flag (`-ffp-contract`) before the test was touched, using
sanitizers to first rule out a real memory bug. Worth naming explicitly
as the standard to hold for anyone else touching this codebase.
