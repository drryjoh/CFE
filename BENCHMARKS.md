# CMU-CFE Performance Benchmarking

Performance is a project requirement and must be measured rather than inferred.

## 1. Benchmark objectives

CMU-CFE benchmarks should answer:

- How fast is a kernel?
- How much useful work is completed per unit time?
- Is performance limited by compute, memory, launch latency, or communication?
- Does performance scale with state size?
- Does the compiler spill registers?
- Does a code change cause a measurable regression?
- Does an abstraction impose measurable cost?
- Does performance change across CPU, Apple Silicon, CUDA GPU, and MPI configurations?

## 2. Required state-size sweep

Performance-sensitive field/storage kernels must be tested for:

```text
1
5
10
20
50
100
```

components where the implementation supports them.

This is intended to expose behavior relevant to multicomponent reacting-flow states.

## 3. Required metrics

Where meaningful, record:

- problem size;
- component/state size;
- precision;
- backend;
- hardware;
- compiler;
- compiler version;
- optimization flags;
- wall-clock runtime;
- iterations;
- throughput;
- cell updates/s;
- scalar updates/s;
- effective memory bandwidth;
- kernel launches;
- kernel duration;
- MPI bytes transferred;
- MPI time;
- total communication time.

For NVIDIA GPU kernels also record when practical:

- registers/thread;
- occupancy;
- shared-memory use;
- local-memory use;
- load/store efficiency;
- memory throughput;
- evidence of register spilling.

## 4. Benchmark classes

### Microbenchmarks

Examples:

- `contract`;
- `weight`;
- fixed-size state operations;
- state loads/stores;
- primitive/conserved conversion;
- flux calculator;
- chemistry source evaluation.

### Kernel benchmarks

Examples:

- scalar update;
- residual evaluation;
- reconstruction;
- flux evaluation;
- DG volume term;
- DG surface term;
- chemistry source kernel;
- ML inference.

### Communication benchmarks

Examples:

- halo packing;
- halo transfer;
- unpacking;
- MPI latency;
- MPI bandwidth;
- CPU-GPU staging where relevant.

### End-to-end benchmarks

Examples:

- scalar advection timestep;
- Burgers timestep;
- Euler timestep;
- Navier-Stokes timestep;
- reacting timestep.

## 5. Benchmark protocol

For each benchmark:

1. warm up the execution path;
2. avoid including one-time initialization unless that is the quantity being measured;
3. run enough repetitions to suppress timer noise;
4. record median and a dispersion statistic;
5. record hardware/software environment;
6. compare against a stored baseline when available.

Do not report only the fastest run.

## 6. Performance regressions

A benchmark regression is not automatically a rejected change, but it must be understood.

For meaningful regressions, document:

- affected benchmark;
- baseline;
- new result;
- percentage change;
- cause;
- scientific or architectural benefit;
- decision to accept or reject.

## 7. Memory-layout studies

AoS, SoA, AoSoA, and other layouts should be treated as hypotheses until measured.

Representative studies should include:

- 1, 5, 10, 20, 50, 100 components;
- read-heavy kernels;
- read/write kernels;
- flux-like kernels;
- CPU;
- CUDA.

The selected default layout should receive an ADR.

## 8. GPU spill monitoring

Large-state kernels must be checked for register pressure.

At minimum document a reproducible workflow using NVIDIA compilation/profiling tools to inspect:

- register count;
- local memory;
- occupancy;
- spill loads;
- spill stores.

If a kernel spills, determine whether the spill is material to runtime before redesigning it.

## 9. Peak-performance interpretation

"Peak" should be defined relative to the kernel.

Possible comparisons include:

- achieved/peak memory bandwidth;
- achieved/peak FLOP rate;
- roofline position;
- scaling relative to a simple reference kernel;
- throughput per cell/species.

Do not describe a kernel as near peak without defining the relevant peak.

## 10. Benchmark storage

Benchmark summaries should eventually live under:

```text
docs/performance/
```

Raw machine-readable benchmark results may use a structured format such as JSON or CSV under a dedicated benchmark-results directory that is not necessarily committed for every local run.

Accepted reference baselines should be version-controlled or archived reproducibly.
