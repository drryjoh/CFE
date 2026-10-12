// Small collective-reduction helpers (task spec 0006): the piece
// task 0004/0005's own "Do not implement" lists and ADR 0009 named but
// deliberately deferred -- a state-dependent-CFL field (BurgersField's
// `wave_speed()` is `max|u|` over the whole initial condition) is only
// safe to decompose across MPI ranks if every rank computes `dt` from
// the GLOBAL maximum, not its own local slice. Two ranks computing
// different `dt` for the same timestep would desynchronize the
// explicit time integration -- a real correctness break, not a
// cosmetic one.
//
// Only ever included from a translation unit built with CFE_ENABLE_MPI.
#pragma once

#ifndef CFE_ENABLE_MPI
#error "mpi_reduce.hpp requires CFE_ENABLE_MPI -- only include it from an MPI-gated translation unit."
#endif

#include <mpi.h>

#include "cfe/backend/mpi/mpi_datatype.hpp"

namespace cfe {
namespace backend {
namespace mpi {

// Returns the maximum of `local_value` across every rank in `comm` --
// every rank gets the same (global) result. Call this once, right
// after computing a rank-local reduction (e.g. the largest |state|
// over that rank's own cells), before using the result to size a
// shared timestep -- exactly the pattern every existing single-rank
// Burgers test/tutorial/benchmark already uses for its own `max_abs_u0`,
// just closed over every rank instead of implicitly assuming there is
// only one.
template <class Scalar>
Scalar allreduce_max(Scalar local_value, MPI_Comm comm = MPI_COMM_WORLD)
{
  Scalar global_value{};
  MPI_Allreduce(&local_value, &global_value, 1, mpi_datatype_for<Scalar>(), MPI_MAX, comm);
  return global_value;
}

}  // namespace mpi
}  // namespace backend
}  // namespace cfe
