# Giving "Which Number Is This?" a Name

*Progress review for task 0015. No new physics, no new numerical
method this time -- a cleanup pass across the whole codebase, the kind
of work that matters for a project meant to keep growing for years
rather than get written once and left alone.*

---

## Slide: What problem this solves

- Every simulation in this project is full of whole numbers that mean
  very different things: "which cell am I in," "how many cells are
  there," "how far into this buffer of memory do I need to look,"
  "which grid point is this in a loop."
- In C++, the raw, low-level type almost everyone reaches for to hold
  any of those numbers is called `std::size_t` -- a generic "this is a
  non-negative whole number, probably a count or position" type built
  into the language itself.
- The problem: `std::size_t` appeared over a thousand times across
  about 100 files, and every single one of those appearances looked
  identical on the page, even though some meant "a position in a
  single room's grid," some meant "how many physics quantities does
  this cell store," and some meant something else again. A reader (or
  a new student joining the project) had no way to tell which, without
  tracing through the surrounding code by hand.
- The fix: give the common case -- "a plain position or count, used at
  run time, inside one simulation" -- its own name, `cfe::Index`, so
  code reads as what it means instead of just what it is.

---

## Slide: This is a renaming pass, not a new capability

- `cfe::Index` is not a new kind of number. It is defined as
  `using Index = std::size_t;` -- literally just a second name for the
  exact same built-in type. Nothing about how the computer stores or
  computes with these numbers changed.
- That makes this a different kind of change from everything else in
  this project's history so far: no new physics, no new numerical
  method, no performance claim to prove. The entire goal was "make the
  code easier for a human to read correctly," while guaranteeing the
  *compiled program* is unchanged, down to the exact same bytes.
- This project already had one precedent for exactly this move:
  `cfe::scalar` is the project's existing name for "whichever precision
  (float or double) this build is using" instead of writing `float` or
  `double` directly everywhere. `cfe::Index` follows that same pattern,
  for counting/indexing instead of precision.
- Two existing, narrower-purpose number types -- `local_index` (used
  for positions inside one computer's share of a big simulation) and
  `global_index` (used for positions across every computer in a
  distributed run) -- were deliberately left completely alone. They
  solve a different, more specific problem and were not touched or
  reinterpreted by this change.

---

## Slide: The one real judgment call -- what NOT to rename

- Not every `std::size_t` in the codebase means "a position or count at
  run time." Some of them are a genuinely different thing: a *fixed*
  number baked into the type itself at compile time -- e.g. "this
  particular kind of physics state always has exactly 5 numbers in it"
  (density, three momentum components, energy, for the compressible-gas
  equations this project simulates).
- Those compile-time counts were deliberately left as plain
  `std::size_t`, not renamed to `cfe::Index` -- they answer a different
  question ("how many, fixed forever for this type") than `cfe::Index`
  answers ("which one, different every time this code runs"). This
  distinction was checked with the project's own simplicity/
  architecture reviewer before touching a single file, specifically so
  it would be applied the same way everywhere instead of being decided
  ad hoc, file by file, as the rename went along.

---

## Slide: How "it's just a renamed label" was actually proven, not assumed

- Even though this change should be a complete no-op for the compiled
  program, "should be" is not evidence -- so it was checked the same
  rigorous way a real numerical change would be, per this project's
  own standing rule of letting the compiler and the test suite be the
  referee instead of confidence.
- The whole project was rebuilt from scratch, and its entire automated
  test suite (over 150 individual checks) was run and diffed, line for
  line, against a build of the code from *before* this change. The
  text that came out the other end was byte-for-byte identical.
- Every runnable example/tutorial in the project (ten of them, from a
  simple 1D example up through a 3D supersonic wind-tunnel simulation)
  was re-run and its output files compared the same way. The only
  differences found anywhere were wall-clock timing numbers (how many
  seconds a run took) -- which naturally vary run to run on any
  computer and were never expected to match -- every actual physics
  result (shock positions, conserved mass, pressure values) matched
  exactly.

---

## Slide: What's still open

- This machine has no NVIDIA GPU, so the GPU-specific files were
  checked by careful reading rather than by compiling and running them
  on real GPU hardware -- the same limitation this project has flagged
  honestly on prior work for the same reason, not something new or
  specific to this change.
- A performance reviewer separately checked whether any of the
  thousand-plus renamed spots were actually hiding a place where an
  even smaller, more specific number type would have been the better
  engineering choice -- a question this rename pass could surface but
  was not the right moment to go fix (changing what a number's
  underlying type IS, rather than just what it's called, is a decision
  with real performance consequences, so it belongs in its own
  deliberate review later, not folded silently into a renaming pass).
