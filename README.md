# CMU Computational Fluids Environment (CMU-CFE)

CMU-CFE is a performance-oriented computational fluid dynamics research environment for education, numerical-method development, multiphysics research, and high-performance simulation.

Its mission is to provide students and researchers with a malleable CFD environment in which new numerical methods and physical models can be implemented, tested, benchmarked, and compared rapidly.

## Design priorities

1. Correctness
2. Measured performance
3. Architectural clarity
4. Testability
5. Extensibility
6. Portability
7. Ease of experimentation

CMU-CFE is being developed with extensive use of generative AI. AI agents may accelerate implementation, testing, refactoring, documentation, and profiling, but scientific and architectural changes are accepted only through verification and measured evidence.

## Near-term objective

The initial objective is a performant scalar-transport foundation on Cartesian grids that establishes:

- CPU and NVIDIA GPU execution
- contiguous field storage
- compile-time state sizing
- precision portability
- structured-grid communication
- correctness testing
- convergence testing
- performance benchmarking
- register-spill monitoring for large states

This foundation will then support Burgers equations, high-order finite volume, DG, compressible flow, chemistry, moving frames, and multimethod DG/FVM development.

## Repository governance

Before modifying the code, read:

1. [`AGENTS.md`](AGENTS.md)
2. [`ARCHITECTURE.md`](ARCHITECTURE.md)
3. relevant files under [`docs/adr/`](docs/adr/)
4. [`agent_history.md`](agent_history.md)
5. relevant tests and benchmarks

Major architectural decisions must be recorded as Architecture Decision Records.

## Core documents

- `AGENTS.md` — permanent rules for human and AI contributors
- `ARCHITECTURE.md` — current software architecture
- `ROADMAP.md` — development milestones
- `BENCHMARKS.md` — performance methodology and baselines
- `VERIFICATION.md` — scientific verification requirements
- `agent_history.md` — chronological AI-assisted development history
- `REFERENCES.md` — algorithms and literature used by the code
- `docs/adr/` — architectural decisions
- `tasks/` — scoped prompts for implementation agents

## Intended execution model

```text
Distributed domain
    -> MPI partition
        -> local cells/faces
            -> CPU threads or accelerator threads
```

Physics and numerical algorithms should remain independent of execution backends.

## Case philosophy

A future case should resemble:

```text
Case/
    domain/
    configuration/
```

and be runnable using:

```bash
python cfe.py --run Case
```

The builder should generate or reuse a case-specific executable specialized for the required physics, numerical method, precision, dimension, and hardware backend.

## License

To be selected before external distribution.
