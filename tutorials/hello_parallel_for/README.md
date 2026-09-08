# Tutorial: Hello, `cfe::parallel_for`

The full Phase 0 benchmark (`tutorials/phase0_benchmark/`) runs 48
combinations and prints a CSV -- useful for real measurements, not so
useful for "let me just see this thing work." This tutorial is that
second thing: one tiny program, one kernel, numbers small enough to
check by hand.

## What it does

Fills an 8-cell, 3-component field with `q(i,k) = i + k + 1`, runs the
same kernel the real benchmark measures --

```cpp
q_new(i, k) = q(i, k) * q(i, k);
```

-- through `cfe::parallel_for`, and prints the field before, the field
after, and how long the kernel took. That's the entire program; see
`hello_parallel_for.cpp`, it's about 50 lines.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_hello_parallel_for -j
./build/tutorials/hello_parallel_for/cfe_hello_parallel_for
```

## What you'll see

This is real output from an actual run (Apple M5, serial backend, Release
build) -- not a mockup:

```text
CMU-CFE Phase 0 hello-world: q_new(i,k) = q(i,k) * q(i,k)
8 cells x 3 components, backend = serial

Before (q):
  cell 0:   1.00   2.00   3.00
  cell 1:   2.00   3.00   4.00
  cell 2:   3.00   4.00   5.00
  cell 3:   4.00   5.00   6.00
  cell 4:   5.00   6.00   7.00
  cell 5:   6.00   7.00   8.00
  cell 6:   7.00   8.00   9.00
  cell 7:   8.00   9.00  10.00

After (q_new = q * q):
  cell 0:   1.00   4.00   9.00
  cell 1:   4.00   9.00  16.00
  cell 2:   9.00  16.00  25.00
  cell 3:  16.00  25.00  36.00
  cell 4:  25.00  36.00  49.00
  cell 5:  36.00  49.00  64.00
  cell 6:  49.00  64.00  81.00
  cell 7:  64.00  81.00 100.00

Elapsed: 0.12 microseconds for 24 scalar updates.
(This array is tiny on purpose so the numbers above are checkable by hand;
 it is not a performance measurement -- see cfe_bench_field_update for that.)
```

Check a couple of cells by hand: cell 0's first component is `1.00`
before and `1.00 * 1.00 = 1.00` after; cell 7's last component is `10.00`
before and `10.00 * 10.00 = 100.00` after. Every value follows the same
rule, applied independently per cell -- that independence is exactly what
`cfe::parallel_for` is exploiting (see `presentations/0001-phase0-execution-foundation.md`
for the plain-language version of why that matters for performance).

## Things to try

- **Switch backends** and re-run to see the reported backend change (the
  8-cell array is too small for threading to show a timing difference --
  that's the point of the *other* tutorial, which uses arrays sized to
  actually saturate memory bandwidth):

  ```bash
  cmake -S . -B build -DCFE_DEFAULT_BACKEND=threaded
  cmake --build build --target cfe_hello_parallel_for -j
  ./build/tutorials/hello_parallel_for/cfe_hello_parallel_for
  ```

- **Switch precision** (`-DCFE_SCALAR_TYPE=float`) and confirm the
  before/after numbers are unchanged -- they should be, since this kernel
  has no rounding-sensitive behavior at these small values.

- **Edit `n_cells`/`n_components`** in `hello_parallel_for.cpp` directly
  and rebuild, to get a feel for the `Field`/`FieldView` API before
  reading the real benchmark or the unit tests.

## Where to go next

- For real timing measurements across precision/component-count/backend/
  layout: `tutorials/phase0_benchmark/README.md`.
- For what the code underneath this tutorial actually looks like:
  `src/cfe/field/field.hpp`, `src/cfe/backend/parallel_for.hpp`.
- For the non-CS-audience explanation of why any of this matters:
  `presentations/0001-phase0-execution-foundation.md`.
