#!/usr/bin/env bash
# Regenerates checkpoint/restart files for this tutorial's expensive
# resolution-sweep cases (task tasks/0016-checkpoint-restart.md).
#
# Intended workflow (documented here, NOT executed remotely by this
# script -- see README.md's "Checkpoint/restart" section and this
# task's own file for why): this script is meant to run unattended
# overnight on PSC Bridges-2 (or any machine with CPU time to spare),
# doing a full from-scratch run of each expensive case and leaving
# behind a restart checkpoint taken shortly before that run's end.
# Local day-to-day development then uses
# `cfe_euler_2d_forward_step_restart_demo --mode=resume` against the
# resulting checkpoint file to re-verify just the tail end of that case
# cheaply, instead of re-running it from scratch every time. This
# script does NOT itself SSH into Bridges-2 or set up a cron job --
# actually wiring that remote automation up is separate, explicit
# follow-up work the user would need to request and supervise directly
# (this task only documents the intended shape of it).
#
# Currently covers exactly one case: cells-per-unit-height=288,
# HllcFlux ("res288_hllc") -- named explicitly in this task's own scope
# ("demonstrate on at least res288_hllc, the most expensive case, since
# that's the one that actually hurts") and the one
# `forward_step_2d_restart_demo.cpp` is built for. Extending this to
# more cases (e.g. res216 at either flux) means adding another small,
# dedicated `--mode=checkpoint`/`--mode=resume` executable following
# that same file's pattern (its `Blocks` alias, resolution, and flux are
# compile-time choices, not runtime ones -- consistent with every other
# `ForwardStepBlocks<Scalar, Dim, NumericalFlux>` instantiation in this
# codebase, AGENTS.md #24's "configuration determines types" rule), not
# a generic runtime resolution/flux switch here -- not built now since
# only this one case has a demonstrated need (AGENTS.md #6).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

echo "=== regenerate_restart_checkpoints.sh: configuring/building (Release, threaded backend) ==="
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCFE_DEFAULT_BACKEND=threaded
cmake --build "${BUILD_DIR}" --target cfe_euler_2d_forward_step_restart_demo -j

cd "${SCRIPT_DIR}"
echo "=== regenerate_restart_checkpoints.sh: res288_hllc full run + checkpoint (this is the expensive part -- tens of minutes) ==="
"${BUILD_DIR}/tutorials/euler_2d_forward_step/cfe_euler_2d_forward_step_restart_demo" --mode=checkpoint

echo "=== regenerate_restart_checkpoints.sh: done. Checkpoint at data/res288_hllc_restart/checkpoint.cferst ==="
echo "Local re-verification from here on: cd ${SCRIPT_DIR} && ${BUILD_DIR}/tutorials/euler_2d_forward_step/cfe_euler_2d_forward_step_restart_demo --mode=resume"
