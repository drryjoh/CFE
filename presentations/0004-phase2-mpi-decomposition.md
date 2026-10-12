# Phase 2: First MPI Domain-Decomposition Prototype

*Progress review for task 0004. Every phase so far has run on one*
*computer at a time — one CPU (maybe with several threads), or one*
*GPU. This PR asks: what happens when the simulation is too big for*
*one computer, and we need several computers to work on it together?*

---

## Slide: What problem is this solving?

- Supercomputers aren't one giant computer — they're racks of many
  separate computers (called **nodes**), each with its own memory,
  wired together by a fast network. No single node can see another
  node's memory directly.
- A simulation that needs more memory or more speed than one node can
  give has to be **split up**: each node simulates its own slice of the
  domain, and the nodes talk to each other over the network to stay in
  sync.
- The standard tool for "many separate computers talking to each
  other" in scientific computing is called **MPI** (Message Passing
  Interface) — think of it as a shared postal system: any node can
  address a message to any other node and know it will arrive. This PR
  adds the project's very first use of it.

---

## Slide: How do you even split a grid in half?

- Picture the simulation's grid as a long row of cells, left to right.
  The simplest way to split it across, say, 4 computers is to give the
  first quarter of the row to computer 1, the next quarter to computer
  2, and so on — like slicing a loaf of bread into 4 pieces and giving
  one slice to each of 4 people.
- Each computer (in MPI terms, each **rank**) ends up owning one
  contiguous slice. This PR adds a small piece of bookkeeping (called
  `SlabPartition`) whose only job is: given the total loaf size and how
  many people are sharing it, work out exactly which slice each person
  gets — including the fiddly case where the loaf doesn't divide evenly
  and someone has to get one extra piece.

---

## Slide: The actual hard part — the edges of each slice

- Remember from earlier phases: every cell's new value depends on its
  *neighbors*. That's easy when all the neighbors live in the same
  computer's memory. It's not easy at the cut between two slices — the
  cell right at the edge of computer 1's slice needs to know the value
  of the cell just across the cut, which lives in computer 2's memory,
  which computer 1 cannot simply reach into.
- The fix is the same "ghost cell" trick every phase so far has already
  used for the *edges of the whole simulated world* (e.g. a periodic
  domain wrapping around) — except now, instead of copying from the
  *opposite side of the same computer's own data*, the ghost cells get
  filled by an actual network message from the neighboring computer.
  This PR calls that piece `MpiHaloBoundary`: it slots into exactly the
  same "fill in my ghost cells" spot every other boundary type already
  used, so nothing else in the simulation engine had to change at all.
- Because every computer needs its *neighbor's* edge value at the same
  moment, each pair of neighbors does one send-and-receive together,
  in both directions, every single timestep.

---

## Slide: Why this was tested on a laptop first, not a supercomputer

- Same philosophy as every GPU feature before this one: get it working
  and verified on hardware you can touch and debug quickly, before
  trusting it on a shared cluster with long queue times.
- This project doesn't currently have its own supercomputer time
  reserved, so a free MPI toolchain (OpenMPI) was installed locally to
  simulate "several computers talking to each other" using several
  processes on one laptop instead — this is a completely standard way
  to develop and debug MPI code before ever touching a real cluster.
- The real target machine (PSC Bridges-2, the same supercomputer this
  project's GPU work already runs on) is still the place the official,
  reported numbers will come from — this PR's laptop numbers are a
  sanity check, not the final answer.

---

## Slide: How do we know the split-up version gives the SAME answer?

- The strongest possible test isn't "it runs without crashing" — it's
  "splitting the work up changes nothing about the answer, down to the
  very last bit." Since every cell's math only ever depends on its own
  immediate neighbors (never on some global shortcut), and a
  network-delivered ghost value is an exact copy of the sender's
  number, there's no reason the answer should differ AT ALL between
  one computer doing all the work and several computers splitting it.
- This PR's test does exactly that: run the identical problem two
  ways — once as one big simulation, once split across several
  computers — and check every single cell matches exactly, with zero
  tolerance for even a tiny difference.
- It passed at 1, 2, 3, 4, 5, and 8 computers (including the "doesn't
  divide evenly" cases). To make sure this test wasn't secretly
  toothless, a one-line bug was deliberately planted in the
  edge-exchange code — the test caught it immediately, every single
  cell near a cut came back wrong — and then the bug was removed again.

---

## Slide: What this doesn't do yet

- Only splits the grid one way (left-to-right slices). Splitting it in
  all 3 directions at once (so each computer gets a brick instead of a
  slice) is real, valuable follow-up work, not done here.
- Only works for the kind of problem that wraps around (like a loop of
  track) — not yet for a problem with a genuine hard wall at the very
  outer edge of the whole domain.
- Doesn't yet work with the self-steepening wave equation from the
  previous PR (Burgers' equation) — that equation needs to check the
  *fastest-moving point anywhere in the whole domain* to stay stable,
  which means the computers would need to compare notes first. That's
  a known, named next step, not an oversight.
- The official performance numbers (how much slower/faster this gets as
  more computers join in) still need to be measured on the real
  supercomputer, not just this laptop.
