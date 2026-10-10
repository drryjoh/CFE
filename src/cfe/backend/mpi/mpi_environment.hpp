// MPI process bootstrap (task spec 0004, ARCHITECTURE.md #2's
// backend/mpi/ directory): a thin RAII wrapper around MPI_Init/
// MPI_Finalize, plus rank()/size() helpers, mirroring backend/cuda/'s
// role as reusable, grid-agnostic plumbing that grid/boundary/
// mpi_halo_boundary.hpp calls into rather than reimplementing itself.
//
// Only ever included from a translation unit built with CFE_ENABLE_MPI
// (same confinement rule AGENTS.md #9 applies to CUDA headers): there is
// no CPU-only fallback definition here, since nothing in the non-MPI
// build should ever reference this header at all.
#pragma once

#ifndef CFE_ENABLE_MPI
#error "mpi_environment.hpp requires CFE_ENABLE_MPI -- only include it from an MPI-gated translation unit."
#endif

#include <mpi.h>

namespace cfe {
namespace backend {
namespace mpi {

// Calls MPI_Init in its constructor and MPI_Finalize in its destructor.
// One instance per process, constructed at the top of main() before any
// other MPI call and destructed (by going out of scope) after the last
// one -- the standard MPI RAII-bootstrap idiom, so a thrown exception or
// an early return still finalizes cleanly.
class Environment
{
 public:
  Environment(int* argc, char*** argv) { MPI_Init(argc, argv); }

  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  ~Environment() { MPI_Finalize(); }
};

inline int rank(MPI_Comm comm = MPI_COMM_WORLD)
{
  int r = 0;
  MPI_Comm_rank(comm, &r);
  return r;
}

inline int size(MPI_Comm comm = MPI_COMM_WORLD)
{
  int s = 0;
  MPI_Comm_size(comm, &s);
  return s;
}

}  // namespace mpi
}  // namespace backend
}  // namespace cfe
