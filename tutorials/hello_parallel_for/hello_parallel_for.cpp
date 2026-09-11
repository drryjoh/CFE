// The smallest possible demonstration of CMU-CFE's Phase 0 building blocks:
// fill a small field, run the same all-cell kernel used by
// benchmarks/memory/bench_field_update.cpp (q_new(i,k) = q(i,k) * q(i,k))
// on it via cfe::parallel_for, and print exactly what happened -- values
// before, values after, and how long it took. No CMake options to tune,
// no CSV to parse: just build and run this one file and read the log.
//
// The array here is deliberately tiny (small enough to eyeball the
// before/after numbers by hand) so this is not a performance measurement.
// For the real 48-combination timing sweep across precisions, component
// counts, backends, and memory layouts, see
// benchmarks/memory/bench_field_update.cpp and
// tutorials/phase0_benchmark/README.md instead.
#include <chrono>
#include <cstdio>

#include "cfe/cfe.hpp"

namespace {

template <class Field>
void print_field(const Field& field)
{
  for (std::size_t i = 0; i < field.n_cells(); ++i) {
    std::printf("  cell %zu:", i);
    for (std::size_t k = 0; k < field.n_components(); ++k) {
      std::printf(" %6.2f", field(i, k));
    }
    std::printf("\n");
  }
}

}  // namespace

int main()
{
  constexpr std::size_t n_cells = 8;
  constexpr std::size_t n_components = 3;

  // cfe::parallel_for always resolves to a CPU backend (serial or threaded)
  // -- see backend/parallel_for.hpp for why CUDA is deliberately not
  // aliased into it.
  std::printf("CMU-CFE Phase 0 hello-world: q_new(i,k) = q(i,k) * q(i,k)\n");
  std::printf("%zu cells x %zu components, backend = %s\n\n", n_cells, n_components,
#if defined(CFE_DEFAULT_BACKEND_THREADED)
              "threaded"
#else
              "serial"
#endif
  );

  cfe::Field<cfe::scalar, n_components> q(n_cells);
  cfe::Field<cfe::scalar, n_components> q_new(n_cells);

  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      q(i, k) = static_cast<cfe::scalar>(i + k + 1);
    }
  }

  std::printf("Before (q):\n");
  print_field(q);

  auto q_view = q.view();
  auto out_view = q_new.view();

  const auto t0 = std::chrono::steady_clock::now();
  cfe::parallel_for(n_cells, [=](std::size_t i) mutable {
    for (std::size_t k = 0; k < n_components; ++k) {
      out_view(i, k) = q_view(i, k) * q_view(i, k);
    }
  });
  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  std::printf("\nAfter (q_new = q * q):\n");
  print_field(q_new);

  std::printf("\nElapsed: %.2f microseconds for %zu scalar updates.\n", elapsed_us,
              n_cells * n_components);
  std::printf(
      "(This array is tiny on purpose so the numbers above are checkable by hand;\n"
      " it is not a performance measurement -- see cfe_bench_field_update for that.)\n");
  return 0;
}
