// Checkpoint/restart demonstration for the forward-facing-step
// resolution sweep (task tasks/0016-checkpoint-restart.md). Motivation:
// re-verifying a change against forward_step_2d.cpp's full sweep means
// re-running every case from t=0, including the most expensive one
// (cells-per-unit-height=288, HLLC flux -- "res288_hllc"), even when the
// change being verified provably cannot affect physics (e.g. a pure
// rename). This driver shows the actual payoff of
// src/cfe/io/restart_file.hpp/ForwardStepBlocks::write_restart/
// read_restart on exactly that case: write a checkpoint ~1000 steps
// before a target step count, then re-verify ONLY those last ~1000
// steps instead of the whole run -- and reports REAL measured
// wall-clock time for both paths, not an estimate. See this tutorial's
// README.md "Checkpoint/restart" section for the actual numbers this
// produced and tutorials/euler_2d_forward_step/
// regenerate_restart_checkpoints.sh for the intended day-to-day
// workflow (full sweep + checkpoints generated on PSC Bridges-2
// overnight; local development re-verifies the tail only, using this
// program's "resume" mode).
//
// Step-count-bounded (`ForwardStepBlocks::run_n_steps`), not time-
// bounded (`run_to`) -- see that method's own doc comment in
// forward_step_setup.hpp for why: a restart-resumable run must not
// depend on reproducing `run_to`'s `t_final`-clamping decision, which
// needs the exact accumulated time restored, not just the state.
// `kDemoTotalSteps` below (12000) is chosen to be of the same ORDER as
// the real res288_hllc sweep case's own step count to t=4 (the
// production tutorial's own `summary.csv` after a from-scratch run, see
// README.md) -- a realistic stand-in for "the expensive case," not a
// literal reproduction of that exact time-bounded run.
//
// Two modes, selected by the one required `--mode=` argument:
//   --mode=checkpoint : full run from scratch, writes a restart
//                        checkpoint `kDemoCheckpointLeadSteps` before the
//                        end, continues to completion, reports total
//                        wall-clock time (the "redo everything from
//                        scratch" cost) and the wall-clock time of the
//                        trailing steps computed in-process (an
//                        internal sanity cross-check against the
//                        separate `--mode=resume` measurement below).
//   --mode=resume     : loads that checkpoint into a FRESH
//                        `ForwardStepBlocks` and runs just the final
//                        `kDemoCheckpointLeadSteps`, reporting wall-
//                        clock time for load+run -- the actual
//                        "re-verify the tail" cost this feature exists
//                        to make cheap.
// Run `--mode=checkpoint` once (expensive, tens of minutes), then
// `--mode=resume` as many times as needed afterward (cheap) to re-check
// the tail -- e.g. after a change provably unrelated to physics.
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "forward_step_setup.hpp"

namespace fs = std::filesystem;

namespace {

using Scalar = cfe::scalar;
using Blocks = cfe_verification::ForwardStepBlocks<Scalar, 2, cfe::HllcFlux>;

constexpr std::size_t kResolution = 288;
constexpr Scalar kCfl = Scalar(0.4);
constexpr std::size_t kDemoTotalSteps = 12000;
constexpr std::size_t kDemoCheckpointLeadSteps = 1000;

const fs::path kDataDir = fs::path("data") / "res288_hllc_restart";
const fs::path kCheckpointPath = kDataDir / "checkpoint.cferst";

double seconds_since(const std::chrono::steady_clock::time_point& start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void run_checkpoint_mode()
{
  fs::create_directories(kDataDir);
  const std::size_t steps_before_checkpoint = kDemoTotalSteps - kDemoCheckpointLeadSteps;

  std::printf("=== restart demo: --mode=checkpoint (res%zu_hllc, %zu steps total, checkpoint %zu steps before end) ===\n",
              kResolution, kDemoTotalSteps, kDemoCheckpointLeadSteps);
  std::fflush(stdout);

  const auto wall_start = std::chrono::steady_clock::now();
  Blocks blocks(kResolution);

  Scalar t = blocks.run_n_steps(steps_before_checkpoint, kCfl);
  const double wall_at_checkpoint = seconds_since(wall_start);
  std::printf("  reached step %zu (t=%.4f) in %.2fs wall -- writing checkpoint\n", steps_before_checkpoint,
              static_cast<double>(t), wall_at_checkpoint);
  std::fflush(stdout);

  blocks.write_restart(kCheckpointPath.string(), steps_before_checkpoint, t);

  const auto wall_tail_start = std::chrono::steady_clock::now();
  t += blocks.run_n_steps(kDemoCheckpointLeadSteps, kCfl);
  const double wall_tail = seconds_since(wall_tail_start);
  const double wall_total = seconds_since(wall_start);

  const double final_mass = blocks.total_mass();
  std::printf("  completed step %zu (t=%.4f): total wall=%.2fs (checkpoint-write-to-end tail=%.2fs), final_mass=%.6f\n",
              kDemoTotalSteps, static_cast<double>(t), wall_total, wall_tail, final_mass);

  std::ofstream summary(kDataDir / "checkpoint_mode_summary.csv");
  summary << "resolution,total_steps,checkpoint_at_step,wall_total_s,wall_tail_inprocess_s,final_mass\n";
  summary << kResolution << "," << kDemoTotalSteps << "," << steps_before_checkpoint << "," << wall_total << ","
          << wall_tail << "," << final_mass << "\n";
  std::printf("  wrote %s and %s\n", kCheckpointPath.string().c_str(),
              (kDataDir / "checkpoint_mode_summary.csv").string().c_str());
}

void run_resume_mode()
{
  if (!fs::exists(kCheckpointPath)) {
    std::printf("No checkpoint found at %s -- run --mode=checkpoint first.\n", kCheckpointPath.string().c_str());
    std::exit(1);
  }

  std::printf("=== restart demo: --mode=resume (res%zu_hllc, loading %s) ===\n", kResolution,
              kCheckpointPath.string().c_str());
  std::fflush(stdout);

  // Three separately-timed phases, not one bundled "load" number: Enzo's
  // review of this task pointed out that `ForwardStepBlocks`
  // construction (building both `CartesianGrid`s, allocating both
  // `BlockSet`s, filling the uniform IC) is itself a measurable fraction
  // of what a bundled construct+read_restart timer would have
  // attributed entirely to `read_restart`'s own I/O cost -- separating
  // them here reports each phase's actual share honestly.
  const auto wall_start = std::chrono::steady_clock::now();
  Blocks blocks(kResolution);
  const double wall_construct = seconds_since(wall_start);

  const auto wall_read_start = std::chrono::steady_clock::now();
  const Blocks::RestartInfo restart_info = blocks.read_restart(kCheckpointPath.string());
  const double wall_read = seconds_since(wall_read_start);
  std::printf("  loaded checkpoint: step=%zu, t=%.4f (construct=%.4fs, read_restart=%.4fs)\n", restart_info.step,
              static_cast<double>(restart_info.time), wall_construct, wall_read);
  std::fflush(stdout);

  const auto wall_run_start = std::chrono::steady_clock::now();
  const Scalar t_final = restart_info.time + blocks.run_n_steps(kDemoCheckpointLeadSteps, kCfl);
  const double wall_run = seconds_since(wall_run_start);
  const double wall_total = wall_construct + wall_read + wall_run;

  const double final_mass = blocks.total_mass();
  std::printf(
      "  resumed to step %zu (t=%.4f): total wall=%.2fs (construct=%.4fs, read_restart=%.4fs, %zu steps=%.2fs), "
      "final_mass=%.6f\n",
      restart_info.step + kDemoCheckpointLeadSteps, static_cast<double>(t_final), wall_total, wall_construct,
      wall_read, kDemoCheckpointLeadSteps, wall_run, final_mass);

  std::ofstream summary(kDataDir / "resume_mode_summary.csv");
  summary << "resolution,resumed_from_step,steps_run,wall_construct_s,wall_read_restart_s,wall_run_s,wall_total_s,"
             "final_mass\n";
  summary << kResolution << "," << restart_info.step << "," << kDemoCheckpointLeadSteps << "," << wall_construct
          << "," << wall_read << "," << wall_run << "," << wall_total << "," << final_mass << "\n";
  std::printf("  wrote %s\n", (kDataDir / "resume_mode_summary.csv").string().c_str());
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2 || (std::string(argv[1]) != "--mode=checkpoint" && std::string(argv[1]) != "--mode=resume")) {
    std::printf("Usage: %s --mode=checkpoint | --mode=resume\n", argv[0]);
    return 1;
  }

  // A mismatched checkpoint (e.g. --mode=resume run against a stale
  // checkpoint from a different resolution) throws std::runtime_error
  // from read_restart_checkpoint with a clear message -- caught here so
  // that reaches the user as a clean one-line error and non-zero exit
  // status, not a raw libc++abi/terminate crash from an uncaught
  // exception (Sally's review finding for this task).
  try {
    if (std::string(argv[1]) == "--mode=checkpoint") {
      run_checkpoint_mode();
    } else {
      run_resume_mode();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
