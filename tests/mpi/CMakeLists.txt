# MPI correctness tests (task spec 0004): a separate executable from
# cfe_unit_tests, since these need mpirun/mpiexec to launch multiple
# processes, not a single-process ctest run. Only add_subdirectory'd
# when CFE_ENABLE_MPI is ON (see root CMakeLists.txt).

add_executable(cfe_test_mpi_halo_exchange test_mpi_halo_exchange.cpp)
target_link_libraries(cfe_test_mpi_halo_exchange PRIVATE cfe_core)

# MPIEXEC_EXECUTABLE/MPIEXEC_NUMPROC_FLAG/MPIEXEC_PREFLAGS/
# MPIEXEC_POSTFLAGS come from the root find_package(MPI COMPONENTS CXX)
# call -- not a hardcoded `mpirun -n`, so this also works unmodified
# under a cluster launcher wrapper (e.g. on PSC Bridges-2) if
# MPIEXEC_PREFLAGS needs scheduler-specific flags there.
foreach(np 2 4)
  add_test(NAME mpi_halo_exchange_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:cfe_test_mpi_halo_exchange> ${MPIEXEC_POSTFLAGS})
endforeach()

# Full 3D block-decomposition correctness test (task spec 0005): fixed
# at a 2x2x2 process grid, so np=8 exactly.
add_executable(cfe_test_mpi_halo_exchange_3d test_mpi_halo_exchange_3d.cpp)
target_link_libraries(cfe_test_mpi_halo_exchange_3d PRIVATE cfe_core)

add_test(NAME mpi_halo_exchange_3d_np8
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS}
                 $<TARGET_FILE:cfe_test_mpi_halo_exchange_3d> ${MPIEXEC_POSTFLAGS})
