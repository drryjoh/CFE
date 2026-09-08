# Running CMU-CFE's CUDA backend on PSC Bridges-2

A concise, reproducible path from "I have a PSC account" to "I ran the
CUDA tests/benchmarks/profiler myself." This is how the CUDA results in
`docs/performance/0002-phase0-cuda-results.md` and
`presentations/0001-phase0-execution-foundation.md` were produced.

## 1. Prerequisites

- A PSC account with an active allocation that includes GPU time (ask
  whoever manages your group's allocation for the account name, e.g.
  `phy260058p` — yours will differ).
- SSH access to `bridges2.psc.edu` (username + password + Duo 2FA; PSC
  does not support plain key-only login).

## 2. (Optional) set up SSH so you don't re-authenticate every command

Add to `~/.ssh/config`:

```
Host bridges
    HostName bridges2.psc.edu
    User <your-psc-username>
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
```

```bash
mkdir -p ~/.ssh/sockets && chmod 700 ~/.ssh/sockets
ssh bridges true   # authenticate once (password + Duo)
```

After that, `ssh bridges '<command>'` reuses the connection with no
further prompts, for up to 4h (or until you reboot/lose network).

## 3. Get the code onto Bridges-2

The repo is public, so a plain clone works from the login node:

```bash
ssh bridges
git clone https://github.com/drryjoh/CFE.git
cd CFE && git checkout main   # or the branch you want to test
```

## 4. Get a GPU allocation

Plain `salloc`/`interact` on `GPU-shared` can queue for hours (this
partition is heavily contended). If your account has the `gpuinteract`
QOS (check with `sacctmgr show assoc user=<you> format=account,qos`),
use it explicitly — it carries much higher scheduling priority and got
an allocation in under a minute instead of queuing to the next day:

```bash
salloc -A <your-account> -p GPU-shared --qos=gpuinteract \
       --gres=gpu:v100-32:1 -t 30:00 --no-shell
```

This prints a job ID (e.g. `Granted job allocation 12345678`) and returns
immediately — it does **not** drop you into a shell. Every subsequent
command must be prefixed with `srun --jobid=<id>` to actually run on the
allocated GPU node (otherwise it runs on the login node, which has no
GPU and a different CPU).

## 5. Load the toolchain

```bash
export HOSTNAME=$(hostname)   # module load fails without this over SSH
module load cuda-v100/12.9.2 gcc/13.3.1-p20240614
```

(`cmake` needs no module — the system one at `/usr/bin/cmake`, 3.26.5, is
new enough.)

**Never pipe a `module load` through another command** (e.g. `... | tee
log`) — the pipeline runs in a subshell and silently discards the
environment changes.

## 6. Configure and build

The V100 is compute capability 7.0:

```bash
cmake -S . -B build -DCFE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_CUDA_HOST_COMPILER=$(which g++)
cmake --build build -j 8
```

## 7. Run it, on the actual GPU node

```bash
srun --jobid=<id> ./build/tests/cfe_unit_tests                          # correctness
srun --jobid=<id> ./build/benchmarks/memory/cfe_bench_field_update_cuda # benchmark sweep
srun --jobid=<id> nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

## 8. Profile it (register/spill/occupancy)

`scripts/profile_cuda.sh` documents the full procedure. Quick version:

```bash
# Static register/spill check (no GPU allocation needed):
nvcc -std=c++20 --extended-lambda -O3 --resource-usage -arch=sm_70 -I src \
     -c benchmarks/memory/bench_field_update_cuda.cu -o /dev/null

# Runtime occupancy/bandwidth (needs the GPU allocation, use ncu --set full):
srun --jobid=<id> ncu --set full ./build/benchmarks/memory/cfe_bench_field_update_cuda
```

## 9. When you're done

```bash
scancel <id>   # release the allocation for others
```

## Notes / gotchas

- A `GPU-shared --gres=gpu:v100-32:1` allocation grants only a handful of
  CPUs (5 observed), not the whole node — check with `srun --jobid=<id>
  nproc`, not plain `nproc` (which reports the login node's much larger
  count and is misleading).
- If `gpuinteract` isn't in your account's QOS list, `sinfo -p GPU-shared
  -N -o "%N %10T %25G"` shows which nodes are `mixed` (have a free GPU
  share) vs. fully `allocated`, which at least tells you whether a plain
  request should succeed once it reaches the front of the queue.
