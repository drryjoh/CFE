// Serial CPU execution backend (task spec item 5).
#pragma once

#include <cstddef>

namespace cfe {
namespace backend {
namespace serial {

// Executes `f(i)` for i in [0, n) on the calling thread, in order.
template <class Index, class Functor>
void parallel_for(Index n, Functor f)
{
  for (Index i = 0; i < n; ++i) {
    f(i);
  }
}

}  // namespace serial
}  // namespace backend
}  // namespace cfe
