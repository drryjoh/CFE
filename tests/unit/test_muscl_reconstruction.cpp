// Unit tests for MusclMinmodReconstruction (task spec 0003 tests:
// "matches hand-computed reference values for monotone data, a local
// extremum (verifying the TVD clip fires), and genuinely linear data
// (verifying the limiter is non-dissipative there)").
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "test_framework.hpp"

CFE_TEST(test_minmod_picks_smaller_magnitude_when_both_arguments_agree_in_sign)
{
  CFE_CHECK_NEAR(cfe::fvm::minmod(3.0, 5.0), 3.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::minmod(5.0, 3.0), 3.0, 1e-12);
  // Same, negative sign: minmod(-3,-5) = -3 (smaller magnitude, sign kept).
  CFE_CHECK_NEAR(cfe::fvm::minmod(-3.0, -5.0), -3.0, 1e-12);
}

CFE_TEST(test_minmod_clips_to_zero_when_arguments_disagree_in_sign)
{
  // This is the TVD clip: at a local extremum (or a discontinuity), the
  // two one-sided differences have opposite sign, and the limiter must
  // produce exactly zero slope, not an average or anything else.
  CFE_CHECK_NEAR(cfe::fvm::minmod(3.0, -5.0), 0.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::minmod(-3.0, 5.0), 0.0, 1e-12);
}

CFE_TEST(test_minmod_clips_to_zero_when_either_argument_is_exactly_zero)
{
  CFE_CHECK_NEAR(cfe::fvm::minmod(0.0, 5.0), 0.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::minmod(5.0, 0.0), 0.0, 1e-12);
}

CFE_TEST(test_muscl_minmod_face_values_at_a_local_extremum_equal_the_cell_value)
{
  // state_{i-1}=1, state_i=5, state_{i+1}=2: cell i is a local maximum
  // (diffs are +4 then -3, opposite sign) -- the limiter must clip the
  // slope to zero, so BOTH face values collapse to exactly the cell's
  // own value, not a central-difference extrapolation past it (which is
  // exactly the oscillation CentralDifferenceReconstruction would
  // produce here).
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_right(1.0, 5.0, 2.0), 5.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_left(1.0, 5.0, 2.0), 5.0, 1e-12);
}

CFE_TEST(test_muscl_minmod_face_values_on_monotone_data_pick_the_smaller_one_sided_slope)
{
  // state_{i-1}=1, state_i=2, state_{i+1}=6: diffs are +1 (left-biased)
  // and +4 (right-biased), same sign -> minmod picks +1 (the smaller).
  // right face = 2 + 1/2 = 2.5, left face = 2 - 1/2 = 1.5.
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_right(1.0, 2.0, 6.0), 2.5, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_left(1.0, 2.0, 6.0), 1.5, 1e-12);
}

CFE_TEST(test_muscl_minmod_face_values_reduce_to_unlimited_slope_on_genuinely_linear_data)
{
  // state_{i-1}=0, state_i=2, state_{i+1}=4: a perfectly linear ramp
  // (both one-sided diffs are exactly +2) -- minmod(2,2)=2 exactly, so
  // the limiter does not clip at all here, and the reconstruction
  // reduces to the same unlimited linear extrapolation
  // CentralDifferenceReconstruction would give for this stencil
  // (state_self +/- half the shared slope): right = 2+1=3, left = 2-1=1.
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_right(0.0, 2.0, 4.0), 3.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_left(0.0, 2.0, 4.0), 1.0, 1e-12);
}

CFE_TEST(test_muscl_minmod_reduces_to_cell_average_for_a_uniform_field)
{
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_right(5.0, 5.0, 5.0), 5.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::muscl_minmod_value_left(5.0, 5.0, 5.0), 5.0, 1e-12);
}

CFE_TEST(test_muscl_minmod_reconstruction_functor_matches_free_functions)
{
  const cfe::fvm::MusclMinmodReconstruction reconstruction{};
  CFE_CHECK_NEAR(reconstruction.right(1.0, 2.0, 6.0), cfe::fvm::muscl_minmod_value_right(1.0, 2.0, 6.0),
                 1e-12);
  CFE_CHECK_NEAR(reconstruction.left(1.0, 2.0, 6.0), cfe::fvm::muscl_minmod_value_left(1.0, 2.0, 6.0),
                 1e-12);
}
