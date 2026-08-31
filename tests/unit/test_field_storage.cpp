// Unit tests for cfe::Field / cfe::FieldView storage indexing (task spec
// items 8/9/13: "storage indexing is correct", "all required state sizes
// compile").
#include "cfe/core/component_counts.hpp"
#include "cfe/field/field.hpp"
#include "cfe/field/layout.hpp"
#include "test_framework.hpp"

CFE_TEST(test_aos_layout_packs_components_contiguously_per_cell) {
  // cell 1, component 2, of 4 cells x 3 components -> 1*3 + 2 = 5
  CFE_CHECK(cfe::AoSLayout::index(1, 2, 4, 3) == 5);
  CFE_CHECK(cfe::AoSLayout::index(0, 0, 4, 3) == 0);
  CFE_CHECK(cfe::AoSLayout::index(3, 2, 4, 3) == 11);
}

CFE_TEST(test_soa_layout_packs_each_component_contiguously_across_cells) {
  // cell 1, component 2, of 4 cells x 3 components -> 2*4 + 1 = 9
  CFE_CHECK(cfe::SoALayout::index(1, 2, 4, 3) == 9);
  CFE_CHECK(cfe::SoALayout::index(0, 0, 4, 3) == 0);
  CFE_CHECK(cfe::SoALayout::index(3, 2, 4, 3) == 11);
}

CFE_TEST(test_field_write_then_read_round_trips_every_cell_and_component) {
  constexpr std::size_t n_cells = 16;
  constexpr std::size_t n_components = 5;
  cfe::Field<double, n_components, cfe::AoSLayout> field(n_cells);

  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      field(i, k) = static_cast<double>(i * 100 + k);
    }
  }
  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      CFE_CHECK_NEAR(field(i, k), static_cast<double>(i * 100 + k), 1e-12);
    }
  }
}

CFE_TEST(test_field_view_shares_storage_with_owning_field) {
  cfe::Field<double, 3, cfe::AoSLayout> field(8);
  auto view = field.view();

  field(2, 1) = 42.0;
  CFE_CHECK_NEAR(view(2, 1), 42.0, 1e-12);

  view(4, 0) = -7.0;
  CFE_CHECK_NEAR(field(4, 0), -7.0, 1e-12);
}

CFE_TEST(test_aos_and_soa_fields_are_index_equivalent_despite_different_physical_layout) {
  constexpr std::size_t n_cells = 6;
  constexpr std::size_t n_components = 4;
  cfe::Field<double, n_components, cfe::AoSLayout> aos(n_cells);
  cfe::Field<double, n_components, cfe::SoALayout> soa(n_cells);

  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      const double value = static_cast<double>(i) * 10.0 + static_cast<double>(k);
      aos(i, k) = value;
      soa(i, k) = value;
    }
  }
  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      CFE_CHECK_NEAR(aos(i, k), soa(i, k), 1e-12);
    }
  }
}

CFE_TEST(test_field_storage_compiles_for_all_required_component_counts) {
  int checked = 0;
  cfe::for_each_component_count([&](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    cfe::Field<double, N> field(32);
    CFE_CHECK(field.n_cells() == 32);
    CFE_CHECK(field.size() == 32 * N);
    field(10, N - 1) = 3.14;
    CFE_CHECK_NEAR(field(10, N - 1), 3.14, 1e-9);
    ++checked;
  });
  CFE_CHECK(checked == 6);
}
