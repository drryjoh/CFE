# ADR 0006: C++ language standard

**Status:** Accepted for CPU; CUDA compatibility unverified
**Date:** 2026-08-31 (PR #1 review correction)

## Context

The original Phase 0 implementation set `CMAKE_CXX_STANDARD 17` without a
documented rationale -- it was an implementation default, not a decision.
PR #1 review flagged this: the language standard is a project-wide
architectural choice (it bounds every future contributor's available
language/library features and must stay compatible with whatever `nvcc`
version CMU-CFE eventually targets) and should be recorded as one.

## Options considered

### A. C++17

Advantages:

- broadest `nvcc` compatibility across CUDA toolkit versions;
- what Phase 0 shipped with, so zero migration risk.

Disadvantages:

- missing features that would materially help this codebase's own stated
  direction -- e.g. concepts, for cleanly constraining `Scalar`/`Layout`
  template parameters instead of relying on unconstrained templates and
  runtime failure; `consteval`; improved `<numbers>`/`<bit>` support.
- no evidence was ever collected for why C++17 specifically was required;
  it was simply not revisited.

### B. C++20

Advantages:

- concepts, `<numbers>`, three-way comparison, and other features directly
  relevant to a numerics codebase with heavy template use.
- modern `nvcc` (CUDA 12.x class toolkits) documents C++20 host-code
  support; recent AppleClang (21, used throughout Phase 0) supports C++20
  fully.

Disadvantages:

- **not verified against this project's actual CUDA toolchain**, because
  no CUDA toolkit/GPU has been available in any development environment
  used so far (see ADR 0001). The claim that `nvcc` accepts C++20 is a
  general claim about recent CUDA releases, not a measurement taken in
  this repository's CI or dev environment.

### C. C++23

Rejected without detailed evaluation: CUDA toolchain support for C++23 is
newer and less established than for C++20, and adopting it would only
widen the gap between "what the CPU compiler accepts" and "what nvcc has
actually been confirmed to accept" -- the opposite of what this decision
should do.

## Decision

Adopt C++20 (`CMAKE_CXX_STANDARD 20`, `CMAKE_CUDA_STANDARD 20` when CUDA is
enabled) as of this review. No CUDA evidence exists that contradicts this
choice, but none exists that confirms it either -- there is no evidence
either way, because no CUDA compiler has been available to test against.

This ADR's status is "Accepted for CPU" rather than fully "Accepted"
because of that gap, mirroring how ADR 0001 treats the CUDA backend itself:
a decision made without CUDA evidence must say so plainly rather than imply
verification that did not happen.

This is a language-standard bump only. No C++20-specific language feature
(concepts, ranges, etc.) is used by this change -- adopting them is future
work, not something this ADR authorizes implicitly.

## Evidence

Full rebuild and all unit tests re-run under `-std=c++20` (AppleClang 21,
this repository's only available compiler): compiles cleanly with
`-Wall -Wextra`, all tests still pass. See `agent_history.md` for the
specific run this ADR's evidence is drawn from.

No CUDA compilation evidence exists for either C++17 or C++20 -- this
review pass had the same "no CUDA toolkit/GPU available" constraint as the
original Phase 0 work.

## Consequences

Any future environment that finally has CUDA hardware must, as part of
verifying the CUDA backend at all (see ADR 0001's outstanding item), also
confirm that its `nvcc` version actually accepts `-std=c++20` for this
project's CUDA translation units. If it does not, this ADR should be
revisited and either the CUDA-specific standard lowered independently of
the CPU standard, or the whole project standard reconsidered.

## Revisit criteria

Revisit when:

- CUDA hardware becomes available and `nvcc` either confirms or rejects
  C++20 support for this project's `.cu` files;
- a specific C++20 feature (e.g. concepts for backend/layout template
  constraints) becomes something the project actually wants to adopt, at
  which point that adoption should be its own reviewed change, not folded
  silently into this one;
- CUDA toolkit requirements change in a way that affects the achievable
  standard.
