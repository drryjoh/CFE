# MPI communication benchmark (task spec 0004). Only add_subdirectory'd
# when CFE_ENABLE_MPI is ON (see root CMakeLists.txt) -- run via
# mpirun/mpiexec, same as tests/mpi's correctness test.
add_executable(cfe_bench_mpi_halo_exchange bench_mpi_halo_exchange.cpp)
target_link_libraries(cfe_bench_mpi_halo_exchange PRIVATE cfe_core)
