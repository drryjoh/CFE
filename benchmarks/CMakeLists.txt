# Phase 0 benchmark infrastructure (task spec item 11).
add_subdirectory(memory)
add_subdirectory(scalar_advection)
add_subdirectory(burgers)

if(CFE_ENABLE_MPI)
  add_subdirectory(mpi)
endif()
