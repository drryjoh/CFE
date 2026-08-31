// Threaded CPU execution backend (task spec item 5).
//
// Deliberately the simplest thing that could work: split [0, n) into
// contiguous chunks, one per `std::thread`, and join. No thread pool, no
// work stealing -- if that ever becomes a bottleneck it should be replaced
// based on measurement, not anticipation (AGENTS.md #2).
#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace cfe {
namespace backend {
namespace threaded {

template <class Index, class Functor>
void parallel_for(Index n, Functor f, unsigned n_threads = 0) {
  if (n <= 0) return;

  if (n_threads == 0) {
    n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;
  }

  const Index count = n;
  const unsigned max_useful_threads =
      static_cast<unsigned>(std::min<Index>(count, static_cast<Index>(n_threads)));
  if (max_useful_threads <= 1) {
    for (Index i = 0; i < count; ++i) f(i);
    return;
  }

  const Index chunk = (count + max_useful_threads - 1) / max_useful_threads;

  std::vector<std::thread> workers;
  workers.reserve(max_useful_threads);
  for (unsigned t = 0; t < max_useful_threads; ++t) {
    const Index begin = static_cast<Index>(t) * chunk;
    const Index end = std::min(count, begin + chunk);
    if (begin >= end) break;
    workers.emplace_back([=]() mutable {
      for (Index i = begin; i < end; ++i) f(i);
    });
  }
  for (auto& w : workers) w.join();
}

} // namespace threaded
} // namespace backend
} // namespace cfe
