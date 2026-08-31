# ADR 0004: Grid connectivity architecture

**Status:** Proposed  
**Date:** 2026-08-30

## Context

Initial development uses Cartesian box grids, but CMU-CFE must later support unstructured hybrid grids on GPUs.

A purely `(i,j,k)`-centric API could make later unstructured support difficult.

Conversely, implementing general unstructured connectivity immediately would slow initial development.

## Proposed approach

Start with specialized Cartesian indexing and neighbor access.

Keep solver/numerics interfaces from assuming that connectivity must always be analytically reconstructed.

Future unstructured grids should use compact contiguous adjacency structures suitable for GPU execution, likely CSR-like or equivalent.

## Decision

Proposed.

Do not implement general unstructured connectivity during the initial scalar-transport phase.

## Revisit criteria

Revisit when the first unstructured DG prototype begins.
