# ADR 0003: Case-specific compilation

**Status:** Proposed  
**Date:** 2026-08-30

## Context

CMU-CFE should generate performant executables specialized for the selected backend, precision, dimension, field, and major numerical configuration.

Users should not pay runtime generality costs when compile-time specialization materially improves hot kernels.

## Proposed approach

A Python builder reads case configuration and divides parameters into:

### Compile-time

Examples:

- backend;
- precision;
- spatial dimension;
- field;
- species mechanism/count;
- major numerical method;
- DG polynomial order when appropriate;
- derivative support.

### Runtime

Examples:

- CFL;
- final time;
- output interval;
- diagnostic locations;
- initial-condition parameters.

The compile-relevant configuration is hashed.

If the hash is unchanged, reuse the existing binary.

## Key rule

**Configuration determines types. Types do not parse configuration.**

Performance-critical C++ code must not parse YAML or other high-level configuration.

## Decision

Proposed. Implementation should occur after the execution foundation is established.

## Revisit criteria

Revisit if compile times become a dominant workflow problem or if a parameter shows no measurable value as compile-time specialization.
