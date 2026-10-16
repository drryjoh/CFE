# Saving Your Place in a Long Simulation

*Progress review for task 0016. Recent PRs built a resolution sweep for
the wind-tunnel simulation: the same physical setup, run at seven
different levels of detail, with two different formulas for how air
crosses a cell boundary. The most detailed version of that sweep takes a
genuinely long time to run -- long enough that re-running it just to
double-check an unrelated change became the single most annoying part of
working on this project. This PR explains the problem, the fix (a
"save game" for a running simulation), and the real time savings it
produced.*

---

## Slide: The problem -- "are you SURE nothing changed?"

- Our most detailed wind-tunnel simulation (cells-per-unit-height=288,
  the sharper of our two flux formulas) takes thousands of time steps
  and tens of minutes of computer time to finish, start to finish.
- Good engineering practice says: whenever you change the code, re-run
  your test cases to make sure you didn't accidentally break the
  physics. That's the right instinct -- but it means EVERY change,
  including ones that provably cannot touch the physics (renaming a
  variable, say), paid the full tens-of-minutes cost to re-confirm
  "yep, still correct."
- That's like re-cooking an entire multi-hour roast from raw every time
  you want to double-check the last ten minutes of it came out right.

---

## Slide: What a "checkpoint" actually is

- A checkpoint is a saved snapshot of a simulation at some point in
  time -- like a save file in a video game. Instead of replaying a
  whole level from the start, you load the save and pick up right
  where you left off.
- For a simulation, "everything you need to pick up where you left off"
  turns out to be surprisingly little: just the actual physical state
  in every real grid cell (density, momentum, energy -- the numbers the
  simulation is actually solving for) plus a note of what time step
  you're on.
- One subtlety specific to this kind of simulation: every grid has a
  thin border of "ghost cells" around it -- fake cells that exist only
  to tell the real cells at the edge what's happening just outside the
  grid (a wall, an open boundary, or -- for our two-block wind tunnel --
  the OTHER block). We checked carefully and confirmed those ghost cells
  never need to be saved at all: the simulation always recalculates them
  itself, fresh, from the real cells, every single time step, before it
  ever reads them. So the save file only needs the real cells.

---

## Slide: Proving it actually works -- not just "looks right"

- The one thing that would make this whole feature worthless: if
  resuming from a checkpoint produced even a slightly different answer
  than just letting the simulation run straight through. A simulation
  like this has no randomness in it at all -- run it twice with the
  same code, you get the exact same numbers, down to the very last
  decimal a computer can represent. So "resume matches continuous,
  bit-for-bit" is the bar, not "resume is close enough."
- We built a direct test: run a simulation 60 steps straight through.
  Separately, run it 40 steps, save a checkpoint, load that checkpoint
  into a brand-new copy of the simulation, and run 20 more steps. Then
  compare every single number in both final results.
- They matched exactly -- not just the real cells we explicitly saved,
  but also everything downstream that depends on them (total mass in
  the system, for instance). We also directly confirmed the "ghost
  cells don't need saving" claim: a fresh checkpoint load leaves those
  border cells completely untouched, proving they really are excluded
  from the save file, not just coincidentally matching.

---

## Slide: The real speedup, measured

- We ran the actual expensive case this whole feature exists for
  (cells-per-unit-height=288, the sharper flux formula) two ways:
  once all the way through from scratch (writing a checkpoint about
  1000 steps before the end along the way), and once by loading that
  checkpoint fresh and only running those last ~1000 steps.
- Full run from scratch: **~14.8 minutes** (886 seconds, 12,000 steps).
  Resuming just the last 1,000 steps from a checkpoint: **~1.3 minutes**
  (78 seconds -- essentially all of it spent actually computing those
  1,000 steps; loading the checkpoint itself took 3 thousandths of a
  second). That's roughly an **11x speedup** for the exact
  re-verification task that prompted this whole feature -- see this
  task's `agent_history.md` entry and `tutorials/euler_2d_forward_step/
  README.md`'s "Checkpoint/restart" section for the full data.
- Both paths landed on the exact same final answer (`final_mass`
  matched to every digit printed), which is exactly what should happen
  if "pick up where you left off" genuinely means the same thing as
  "never stopped."

---

## Slide: What's still manual, and why

- This only works on the SAME machine/build that wrote the checkpoint
  -- it's not (yet) a format for moving a save file between different
  computers. That wasn't needed yet, so we didn't build it; the file
  format has a version number specifically so we CAN extend it later
  without breaking old save files.
- We also did not touch anything related to running this simulation
  spread across multiple computers (MPI) -- that's explicitly separate,
  already-planned future work, not something this change needed to
  solve.
- We wrote (but did not run) a script for the intended real workflow:
  the expensive full sweep runs unattended overnight on a shared
  research supercomputer (PSC Bridges-2), producing checkpoints; day-to-
  day development on a laptop then uses those checkpoints to re-check
  just the parts that matter, fast. Actually setting up that overnight
  automation on the remote machine is its own separate next step, not
  done here.
