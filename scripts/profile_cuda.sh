#!/usr/bin/env bash
# CUDA register/occupancy/local-memory/spill inspection procedure
# (task spec item 12, BENCHMARKS.md #8).
#
# This script documents the workflow; it has NOT been run against real
# hardware because no CUDA toolkit/NVIDIA GPU was available during Phase 0
# development (see docs/adr/0001-execution-backend.md and
# docs/performance/0001-phase0-results.md). Treat every command below as
# reviewed-but-unverified until someone with CUDA hardware runs it and
# records the output in docs/performance/.
#
# Usage (on a machine with the CUDA toolkit and an NVIDIA GPU):
#   CFE_ENABLE_CUDA=ON cmake -S . -B build -DCFE_ENABLE_CUDA=ON
#   cmake --build build --target cfe_bench_field_update_cuda
#   scripts/profile_cuda.sh build/benchmarks/memory/cfe_bench_field_update_cuda
set -euo pipefail

BINARY="${1:-build/benchmarks/memory/cfe_bench_field_update_cuda}"

if [[ ! -x "${BINARY}" ]]; then
  echo "error: ${BINARY} not found or not executable." >&2
  echo "Build it first with CFE_ENABLE_CUDA=ON (see header comment)." >&2
  exit 1
fi

echo "== 1. Static register/local-memory usage per kernel instantiation =="
echo "nvcc --resource-usage reports registers, shared memory, and (critically"
echo "for spill detection) 'stack frame'/'spill stores'/'spill loads' bytes"
echo "per thread, per kernel instantiation. Because parallel_for_kernel is a"
echo "template, this must be run once per (Scalar, N, Layout) instantiation"
echo "of interest -- the required sweep is 1,5,10,20,50,100 components x"
echo "float/double (task spec item 9)."
echo
echo "Example (adjust -arch to the target GPU's compute capability):"
echo "  nvcc -std=c++17 -O3 --resource-usage -arch=sm_80 \\"
echo "       -I src \\"
echo "       -c benchmarks/memory/bench_field_update_cuda.cu -o /dev/null"
echo
echo "Look for lines of the form:"
echo "  ptxas info : Used N registers, M bytes smem, K bytes cmem"
echo "  ptxas info : Function properties for ... 0 bytes stack frame,"
echo "               0 bytes spill stores, 0 bytes spill loads"
echo "A nonzero 'spill stores'/'spill loads' is direct evidence of register"
echo "spilling for that instantiation."
echo

echo "== 2. Occupancy =="
echo "Two independent ways to get achieved occupancy for the running kernel:"
echo "  a) Nsight Compute (preferred, gives achieved + theoretical occupancy,"
echo "     limiter analysis, and memory throughput in one pass):"
echo "       ncu --set full -o phase0_field_update_report ${BINARY}"
echo "     Then inspect the 'Occupancy' and 'Launch Statistics' sections, and"
echo "     the 'Source Counters' section for per-instruction spill traffic."
echo "  b) cudaOccupancyMaxActiveBlocksPerMultiprocessor (programmatic, no"
echo "     profiler needed) -- useful for a quick automated check per"
echo "     instantiation; not wired into this repo's benchmark binary yet."
echo

echo "== 3. Local-memory traffic under load =="
echo "  ncu --metrics l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\\"
echo "               l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum \\"
echo "      ${BINARY}"
echo "Nonzero local-memory load/store bytes here corroborate static spill"
echo "evidence from --resource-usage with actual runtime traffic."
echo

echo "== 4. Achieved memory bandwidth (for roofline context) =="
echo "  ncu --set full ${BINARY}"
echo "and read 'Memory Throughput' / 'DRAM Throughput' from the summary;"
echo "compare against the GPU's published peak bandwidth per BENCHMARKS.md #9"
echo "(never describe a result as 'near peak' without stating that peak)."
echo

echo "== Recording results =="
echo "Append findings to docs/performance/, following the same format as"
echo "docs/performance/0001-phase0-results.md, and update"
echo "docs/adr/0001-execution-backend.md / 0002-state-memory-layout.md if the"
echo "data changes either decision."
