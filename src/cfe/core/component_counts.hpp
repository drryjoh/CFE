// Compile-time list of component counts Phase 0 must build and exercise
// (AGENTS.md #8, task spec item 9): 1, 5, 10, 20, 50, 100.
//
// `for_each_component_count` instantiates a callable once per count via a
// fold expression, so tests/benchmarks can write the sweep once instead of
// hand-duplicating six call sites.
#pragma once

#include <cstddef>
#include <utility>

namespace cfe {

using ComponentCounts = std::index_sequence<1, 5, 10, 20, 50, 100>;

template <class Functor, std::size_t... Ns>
void for_each_component_count_impl(Functor&& f, std::index_sequence<Ns...>) {
  (f(std::integral_constant<std::size_t, Ns>{}), ...);
}

// Calls `f(std::integral_constant<std::size_t, N>{})` for each required
// component count N. `f` should be a generic lambda, e.g.:
//
//   for_each_component_count([](auto n_components) {
//     constexpr std::size_t N = decltype(n_components)::value;
//     ...
//   });
template <class Functor>
void for_each_component_count(Functor&& f) {
  for_each_component_count_impl(std::forward<Functor>(f), ComponentCounts{});
}

} // namespace cfe
