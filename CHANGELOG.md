# Changelog

All notable project-level changes may be summarized here.

Detailed AI-assisted task history belongs in `agent_history.md`.

## Unreleased

### Added

- Initial repository governance documents.
- Project architecture.
- Development roadmap.
- Verification requirements.
- Performance benchmark requirements.
- ADR framework.
- Initial Phase 0 agent task.
- Phase 0 execution foundation: CMake C++ project scaffold, configurable
  scalar/index types, host/device macros, `cfe::parallel_for` (serial CPU,
  threaded CPU, CUDA), fixed-size math containers/operations (`contract`,
  `weight`), contiguous compile-time-sized field storage (`AoSLayout`/
  `SoALayout`), unit tests, benchmark infrastructure, and a benchmark
  tutorial. No CFD physics. CUDA backend is implemented but unverified (no
  CUDA hardware available); see `docs/performance/0001-phase0-results.md`
  and `docs/adr/0001-execution-backend.md`.
