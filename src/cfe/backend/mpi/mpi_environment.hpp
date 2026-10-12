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

#include <stdexcept>
#include <string>

namespace cfe {
namespace backend {
namespace mpi {

// Calls MPI_Init_thread in its constructor and MPI_Finalize in its
// destructor. One instance per process, constructed at the top of
// main() before any other MPI call and destructed (by going out of
// scope) after the last one -- the standard MPI RAII-bootstrap idiom, so
// an exception thrown AFTER construction succeeds, or an early return,
// still finalizes cleanly via the destructor. A failure detected inside
// the constructor itself is a separate case the destructor cannot cover
// (a throwing constructor means the object never finishes construction,
// so its destructor never runs) -- see the explicit MPI_Finalize() call
// on the insufficient-thread-support path below.
//
// Requests MPI_THREAD_FUNNELED, not plain MPI_Init (whose thread-safety
// level is implementation-defined and, per the MPI standard, may be as
// low as MPI_THREAD_SINGLE): this project's CPU backend can be built
// threaded (`CFE_DEFAULT_BACKEND=threaded`, backend/parallel_for.hpp),
// in which case `MpiHaloBoundary::exchange`'s own pack/unpack
// `Backend::run` calls spawn worker threads for the per-cell kernel
// while this same function's `MPI_Sendrecv` calls are issued directly
// from the thread that is running `exchange()` itself -- i.e. exactly
// one thread ever calls into MPI, the same thread that called
// MPI_Init_thread here, and no MPI call is ever made concurrently with
// another from a different thread. MPI_THREAD_FUNNELED is the precise,
// minimal level that describes and permits that usage (not
// MPI_THREAD_SERIALIZED/MPI_THREAD_MULTIPLE, which would ask for more
// than this codebase ever needs). If the local MPI implementation
// cannot provide at least that level, failing loudly here is far
// preferable to a silent, intermittent data race deep inside a later
// halo exchange.
class Environment
{
 public:
  Environment(int* argc, char*** argv)
  {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
      // A constructor that throws never finishes construction, so
      // ~Environment() -- which is what would normally call
      // MPI_Finalize() -- never runs for this object. MPI_Init_thread
      // above already succeeded (just with an insufficient thread
      // level), so without this explicit call MPI would be left
      // permanently initialized-but-unfinalized on the way out, which
      // most MPI implementations treat as an error on process exit
      // (e.g. a hang or an abort message) regardless of whether this
      // exception is ever caught.
      MPI_Finalize();
      throw std::runtime_error(
          "cfe::backend::mpi::Environment: this MPI implementation only provides thread support level " +
          std::to_string(provided) + ", but MPI_THREAD_FUNNELED (" + std::to_string(MPI_THREAD_FUNNELED) +
          ") is required -- CFE's threaded CPU backend relies on being able to call MPI from the "
          "initializing thread while other threads run per-cell kernels.");
    }
  }

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
