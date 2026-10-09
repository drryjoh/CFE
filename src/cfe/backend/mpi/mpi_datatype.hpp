// Maps a C++ Scalar type to its matching MPI_Datatype (task spec 0004).
// Only `float`/`double` are specialized -- the only two types
// `cfe::scalar` (core/types.hpp) can ever be -- so an unsupported Scalar
// fails to compile (no specialization matches) rather than silently
// sending the wrong datatype at runtime.
//
// Only ever included from a translation unit built with CFE_ENABLE_MPI.
#pragma once

#ifndef CFE_ENABLE_MPI
#error "mpi_datatype.hpp requires CFE_ENABLE_MPI -- only include it from an MPI-gated translation unit."
#endif

#include <mpi.h>

namespace cfe {
namespace backend {
namespace mpi {

template <class Scalar>
MPI_Datatype mpi_datatype_for();

template <>
inline MPI_Datatype mpi_datatype_for<float>()
{
  return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_datatype_for<double>()
{
  return MPI_DOUBLE;
}

}  // namespace mpi
}  // namespace backend
}  // namespace cfe
