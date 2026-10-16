// HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver (Toro,
// Spruce, Speares, "Restoration of the contact surface in the HLL
// Riemann solver," Shock Waves 4, 1994; reproduced as Toro, "Riemann
// Solvers and Numerical Methods for Fluid Dynamics," 3rd ed., Ch. 10,
// eq. 10.37-10.43) for the compressible Euler equations.
//
// Unlike Rusanov (numerics/numerical_flux/rusanov.hpp), which collapses
// the Riemann fan to a SINGLE dissipative wave moving at the fastest
// speed in either direction, HLLC models the fan as FOUR regions
// separated by three waves (left nonlinear wave, the contact, right
// nonlinear wave) -- so the contact discontinuity is resolved as a
// genuine, undamped jump at its own speed `S_star`, rather than smeared
// by dissipation sized for the fastest wave in the problem. This is the
// documented weakness of Rusanov this flux specifically targets (see
// rusanov.hpp's header comment and docs/adr/0010-...md's "Future
// constraint"); see docs/adr/0012-hllc-numerical-flux.md for the
// evidence this actually holds (tests/verification/
// test_euler_isolated_contact_hllc.cpp, test_euler_sod_shock_tube_hllc.cpp).
//
//   F_HLLC(qL, qR) = F_L                      if S_L  >= 0
//                  = F*_L = F_L + S_L*(q*_L - qL)   if S_L <= 0 <= S_star
//                  = F*_R = F_R + S_R*(q*_R - qR)   if S_star <= 0 <= S_R
//                  = F_R                      if S_R  <= 0
//
// --- Wave-speed estimate: ANRS seed + bounded Newton-Raphson refinement -
//
// `S_L`/`S_R`/`S_star` all depend on an estimate of the star-region
// pressure `p*`. This went through TWO prior, rejected designs before
// landing here -- both caught by evidence, not by inspection, per
// AGENTS.md #25's acceptance sequence:
//
// 1. Plain PVRS (primitive-variable Riemann solver, Toro eq. 9.20, Sec.
//    9.5.1) used directly as `p*`: measured (test_hllc_flux.cpp's Sod
//    cross-check against the independent exact solver,
//    tests/verification/sod_shock_tube_exact.hpp) to give `p* = 0.55`
//    against the exact `p* = 0.30313` (82% error) -- Sod's 10:1 pressure
//    ratio is well outside PVRS's own "weak wave" validity range.
// 2. Toro's ANRS (Adaptive Noniterative Riemann Solver, Sec. 9.5.2) --
//    adaptively switches among PVRS/TRRS (two-rarefaction)/TSRS
//    (two-shock) closed-form estimates, still O(1), no iteration.
//    Measurably better for Sod (`p* = 0.315`, ~4% error) -- but testing
//    this flux end-to-end against the forward-facing-step benchmark
//    (tutorials/euler_2d_forward_step/, task 0011) surfaced a SEVERE
//    failure ANRS alone does not protect against: at the step's riser
//    face (a solid wall reflecting the Mach-3, u=3 inflow -- a
//    symmetric `u_L=+3`/`u_R=-3` head-on reflection, `gamma=1.4`,
//    `rho=1.4`, `p=1`), the TSRS branch's single-shot estimate gives
//    `p0 = 10.0` against the true `p* = 17.2` -- close enough in
//    RELATIVE terms (42% low) but on the WRONG SIDE of the threshold
//    (`p*/p_L > ~10.33`) needed for `q_L` to push `S_L = u_L - c_L*q_L`
//    negative. The result: `S_L` comes out POSITIVE (`+0.047`) for a
//    wall reflection that should produce a strong shock running
//    UPSTREAM into the oncoming supersonic flow -- `hllc_flux` silently
//    takes the `S_L >= 0` branch and returns `F(qL)` UNCHANGED, i.e. ZERO
//    numerical dissipation at the one place this problem most needs it.
//    Reproduced directly: the entire forward-step domain stayed frozen
//    at the exact uniform inflow state for the whole run (`t=0` to
//    `t=4`, every resolution) -- not a subtle accuracy gap, a complete,
//    silent failure to react to a genuine strong-shock interface. See
//    docs/adr/0012-hllc-numerical-flux.md for the full diagnostic
//    evidence (this is NOT a documented-acceptable simplification the
//    way componentwise reconstruction is in ADR 0010 -- a wrong-signed
//    wave speed is a correctness defect, not a lower-order approximation).
//
// Fix: use the SAME ANRS estimate only as a starting GUESS, then run a
// FIXED, UNCONDITIONAL number of Newton-Raphson iterations (Toro eq.
// 4.5-4.8's pressure function and its derivative -- the identical
// formulas tests/verification/sod_shock_tube_exact.hpp's independent
// exact solver already uses, reimplemented here in `src/` since
// production code must not depend on a `tests/` header) refining `p*`
// toward the EXACT two-shock/two-rarefaction solution. Fixed (not
// convergence-gated with an early `break`) specifically for warp-uniform
// GPU execution (a "warp" is a group of 32 CUDA threads executing in
// lockstep; one thread exiting a loop early does not let the warp move
// on -- it still waits for the slowest thread in the same warp, so a
// convergence-gated loop can cost just as much as the slowest cell's
// iteration count anyway, while looking "free" on CPU where threads
// truly run independently; see docs/performance/0007-phase3-euler-cuda-results.md
// for the fuller explanation of warps/occupancy this project uses
// elsewhere) -- every thread/cell does the same fixed amount of work
// regardless of how quickly its own Riemann problem happens to converge
// (Enzo review point); 5 iterations, seeded from ANRS (a far better seed
// than the exact solver's own crude `0.5*(p_L+p_R)`), converges to full
// `double` precision in <=4 iterations even for this file's own worst
// discovered case (`p0=10.0 -> 17.2` in 4 iterations, Newton's quadratic
// convergence) -- see docs/adr/0012-...md for the full iteration trace.
// `p` is floored at a tiny positive value after every iteration (not
// just the final result) -- a single bad step landing at a negative or
// zero trial pressure would otherwise feed a negative argument into the
// next iteration's own `sqrt`/`pow` evaluations.
//
//   p_pvrs = 0.5*(p_L+p_R) - 0.5*(u_R-u_L)*rho_bar*c_bar
//   p0 = ANRS(p_pvrs, p_L, p_R, ...)        -- Sec. 9.5.2, as before, SEED only
//   repeat 5 times:
//     f  = f_L(p) + f_R(p) + (u_R - u_L)              (Toro eq. 4.5-4.7)
//     f' = f_L'(p) + f_R'(p)
//     p  = max(tiny, p - f/f')
//   p* = max(0, p)
//
// This makes HLLC's wave speeds effectively exact-Riemann-solver-quality
// (not merely "a better noniterative estimate") at a small, FIXED extra
// cost per face (5 more sqrt/pow evaluations) -- a correctness-over-
// speed tradeoff AGENTS.md #2's priority order requires ("never trade
// correctness for speed"), and the forward-step re-run
// (tutorials/euler_2d_forward_step/README.md) is the end-to-end evidence
// this fix actually resolves the frozen-flow failure above.
//
// Residual gap, now measured with a fully Newton-converged `p*` (not an
// estimate): for classic Sod, `S_L=-1.183216` and `S_R=1.752156` now
// match the exact solver's own rarefaction-head and shock speeds EXACTLY
// (tests/verification/sod_shock_tube_exact.hpp's documented values) --
// yet `S_star=0.641832` still differs substantially from the exact
// `u*=0.927453`. This PROVES the gap is intrinsic to HLLC's star-speed
// formula itself (eq. 10.37 implicitly treats both nonlinear waves as if
// they were shocks via a Rankine-Hugoniot-style relation), not an
// artifact of wave-speed estimation accuracy: Sod's left wave is a
// genuinely STRONG rarefaction (u*/c_L ~ 0.78), and HLLC/HLL-family
// solvers fundamentally collapse an entire rarefaction fan to a single
// wave speed `S_L` regardless of how exactly that speed is computed.
// This is a documented, inherent property of this solver family (not a
// bug): see test_hllc_flux.cpp's double-shock-only test for a case where
// the formula's own assumptions (both nonlinear waves are genuine
// shocks) actually hold, where `S_star` does track `u*` tightly.
//
// Given `p*`:
//   q_K = 1                                               if p* <= p_K
//       = sqrt(1 + (gamma+1)/(2*gamma)*(p*/p_K - 1))       otherwise
//   S_L = u_L - c_L*q_L,  S_R = u_R + c_R*q_R
//   S_star = [p_R - p_L + rho_L*u_L*(S_L-u_L) - rho_R*u_R*(S_R-u_R)]
//            / [rho_L*(S_L-u_L) - rho_R*(S_R-u_R)]                 (10.37)
//
// --- Star-region conserved state, generalized over Dim -----------------
//
// For side K (L or R), with `n = Field::momentum_index(axis)` the ONE
// conserved component holding momentum along `axis`:
//
//   D*_K = rho_K * (S_K - u_K) / (S_K - S_star)
//   q*_K[density]          = D*_K
//   q*_K[momentum_index(axis)] = D*_K * S_star
//   q*_K[momentum_index(d)]    = D*_K * velocity_component(q_K, d)   for every other axis d
//   q*_K[energy]            = D*_K * [ E_K/rho_K + (S_star-u_K)*(S_star + p_K/(rho_K*(S_K-u_K))) ]
//
// Only the NORMAL momentum component is set to the star velocity
// `S_star`; every tangential momentum component is carried through at
// its ORIGINAL velocity, just density-rescaled by `D*_K/rho_K` -- a
// genuinely different formula from the normal component, not the same
// one applied componentwise (a documented pitfall: see
// test_hllc_flux.cpp's oblique 2D/3D test, which specifically exercises
// a nonzero tangential velocity to catch a "same formula for every
// momentum component" bug that a 1D-only test cannot).
//
// --- Consistency (self-check, see test_hllc_flux.cpp) -------------------
//
// For qL == qR (no jump), S_star reduces exactly to the shared u_axis,
// D*_K reduces exactly to rho_K, and q*_K reduces exactly to qK itself
// (the energy term's (S_star-u_K) factor vanishes) -- so every branch
// above reduces to the plain physical flux, exactly like Rusanov's own
// degenerate-case check (rusanov_flux(q,q,...) == physical_flux(q,...)).
//
// --- NOT a generic drop-in NumericalFlux (important scoping note) ------
//
// `RusanovFlux` (rusanov.hpp) only ever calls `field.wave_speed(...)`/
// `field.physical_flux(...)` -- a narrow enough contract that the SAME
// functor serves `BurgersField`/`ScalarAdvectionField` (N=1) and
// `EulerField` (N>1) unchanged. `HllcFlux` needs considerably more:
// `field.density/pressure/velocity_component/sound_speed`, `field.gamma`,
// and `Field::momentum_index/energy_index` -- i.e. a full ideal-gas
// equation of state, not just a single dissipation-speed scalar. This
// makes `HllcFlux` implicitly Euler-specific (or any future Field
// exposing the same ideal-gas Calculator surface), NOT a fully generic
// `NumericalFlux` the way `RusanovFlux`/`UpwindFlux` are -- attempting to
// instantiate it against `BurgersField`/`ScalarAdvectionField` simply
// fails to compile (those Fields have no `pressure`/`sound_speed`/
// `gamma`), which is the correct outcome, not a bug to fix.
#pragma once

#include <cmath>
#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/math/operations.hpp"

namespace cfe {

namespace detail {

// Shared floor constants (Snorlax review finding): these are
// safety-critical tuning constants -- they exist specifically to stop a
// real NaN regression (see hllc_star_state's and hllc_wave_speeds' own
// comments) -- so they are declared ONCE here and referenced by both
// functions, not independently redeclared at each call site where a
// future edit could silently desynchronize them.
template <class Scalar>
inline constexpr Scalar kTinyDensityFloor = Scalar(1e-6);
template <class Scalar>
inline constexpr Scalar kTinyPressureFloor = Scalar(1e-6);

// Floored (density, velocity-along-axis, pressure) for one conserved
// state -- the SINGLE formula both `hllc_wave_speeds` and
// `hllc_star_state` use to protect against the near-vacuum
// reconstruction artifact (see `kTinyDensityFloor`/`kTinyPressureFloor`
// above). Factored out after an Enzo review finding: an earlier version
// had `hllc_wave_speeds` compute `u_l`/`u_r` via
// `field.velocity_component(...)` and `p_l`/`p_r` via `field.pressure(...)`
// directly -- both of which divide by the RAW (possibly near-zero or
// slightly NEGATIVE) density internally, not the floored one computed
// alongside them. A small negative raw density make `kinetic_energy`
// (momentum^2 / negative rho) swing to a large NEGATIVE value, which
// then makes the computed pressure `(gamma-1)*(E - kinetic_energy)`
// swing to a large POSITIVE one -- finite and positive, so it silently
// PASSES the `p > kTinyPressureFloor` floor check even though it is
// wildly wrong, reintroducing the exact "complete, silent failure, no
// crash/NaN" class of bug Decision 2b's own fix was written to close,
// just at a different entry point. Computing velocity/pressure from the
// SAME already-floored density (never from a Field accessor that
// recomputes density itself) closes this for both call sites at once.
template <class Scalar, std::size_t N, class Field>
struct HllcFlooredPrimitives
{
  Scalar rho, u, p;
};

template <class Scalar, std::size_t N, class Field>
CFE_HOST_DEVICE
HllcFlooredPrimitives<Scalar, N, Field> hllc_floored_primitives(const State<Scalar, N>& q, Axis axis,
                                                                 const Field& field)
{
  constexpr std::size_t Dim = Field::dim;
  const Scalar rho_raw = field.density(q);
  const Scalar rho = rho_raw > kTinyDensityFloor<Scalar> ? rho_raw : kTinyDensityFloor<Scalar>;
  const Scalar u = field.momentum(q, axis) / rho;
  // Sum-of-squares via `contract` (math/operations.hpp, AGENTS.md #13)
  // rather than a hand-rolled accumulator -- PR #6 cleanup task, same
  // treatment as `EulerField::kinetic_energy`. Only the independent
  // (not fused into another per-component loop) sum-of-squares site in
  // this file was converted; `hllc_star_state`'s own momentum_squared_sum
  // below is built WHILE constructing `q_star`'s components in the same
  // loop and was deliberately left alone (converting it would need an
  // extra pass over an already-hardened, floor-safety-critical function
  // for no behavior change -- see this file's own review history).
  Vector<Scalar, Dim> momentum;
  for (Index d = 0; d < Dim; ++d) momentum[d] = q[Field::momentum_index(static_cast<Axis>(d))];
  const Scalar kinetic = Scalar(0.5) * contract(momentum, momentum) / rho;
  const Scalar p_raw = (field.gamma - Scalar(1)) * (field.total_energy(q) - kinetic);
  const Scalar p = p_raw > kTinyPressureFloor<Scalar> ? p_raw : kTinyPressureFloor<Scalar>;
  return HllcFlooredPrimitives<Scalar, N, Field>{rho, u, p};
}

// This side's star-region conserved state, given its own wave speed
// `S_K` and the shared contact speed `S_star` -- see this file's header
// comment for the formula and the tangential-vs-normal distinction.
template <class Scalar, std::size_t N, class Field>
CFE_HOST_DEVICE
State<Scalar, N> hllc_star_state(const State<Scalar, N>& q_k, Scalar s_k, Scalar s_star, Axis axis,
                                  const Field& field)
{
  constexpr std::size_t Dim = Field::dim;
  const Index normal_component = Field::momentum_index(axis);

  // Floors (Violet review finding, see this file's header comment on the
  // forward-step res40+ NaN): `density_star`'s formula is only
  // guaranteed non-negative if the wave ordering `S_L <= S_star <= S_R`
  // genuinely holds, which is a theorem given the EXACT two-shock/
  // two-rarefaction `p*` -- but a near-vacuum reconstructed input (a
  // real, observed MUSCL componentwise-reconstruction artifact at a
  // strong 2D multi-wave interaction, not a bug in this formula itself)
  // can perturb the Newton-refined `p*` just enough that this ordering
  // guarantee slips, producing a negative `density_star` that then
  // poisons every other component here, and ultimately the next RK
  // stage's own reconstruction (sqrt of a negative implied pressure).
  // Flooring both the input density/pressure used below AND the
  // resulting star density/implied pressure stops this at its two entry
  // points -- local to this one function, no change to the formula
  // itself for any well-separated (non-near-vacuum) input, where these
  // floors are no-ops. Uses the SAME `hllc_floored_primitives` helper
  // `hllc_wave_speeds` does, so the two can never silently drift out of
  // sync (Enzo/Snorlax review findings).
  const HllcFlooredPrimitives<Scalar, N, Field> prim = hllc_floored_primitives(q_k, axis, field);
  const Scalar rho_k = prim.rho;
  const Scalar u_k = prim.u;
  const Scalar p_k = prim.p;
  const Scalar e_k = field.total_energy(q_k);

  const Scalar density_star_raw = rho_k * (s_k - u_k) / (s_k - s_star);
  const Scalar density_star = density_star_raw > kTinyDensityFloor<Scalar> ? density_star_raw : kTinyDensityFloor<Scalar>;

  State<Scalar, N> q_star;
  q_star[0] = density_star;
  Scalar momentum_squared_sum = Scalar(0);
  for (Index d = 0; d < Dim; ++d) {
    const Index component = Field::momentum_index(static_cast<Axis>(d));
    if (component == normal_component) {
      q_star[component] = density_star * s_star;
    } else {
      q_star[component] = density_star * (q_k[component] / rho_k);
    }
    momentum_squared_sum += q_star[component] * q_star[component];
  }
  const Scalar energy_raw =
      density_star * (e_k / rho_k + (s_star - u_k) * (s_star + p_k / (rho_k * (s_k - u_k))));
  // Floor the IMPLIED pressure of the returned star state, not just its
  // raw energy value -- the same "a combination of independently-
  // reasonable components can imply a non-physical pressure" risk this
  // whole fix targets applies to the star state's own construction too.
  const Scalar kinetic_star = Scalar(0.5) * momentum_squared_sum / density_star;
  const Scalar min_energy_for_floor_pressure = kinetic_star + kTinyPressureFloor<Scalar> / (field.gamma - Scalar(1));
  q_star[Field::energy_index()] = energy_raw > min_energy_for_floor_pressure ? energy_raw : min_energy_for_floor_pressure;
  return q_star;
}

// Toro's shock-impedance coefficients (the `A_K`/`B_K` of eq. 9.42,
// equivalently the `a_k`/`b_k` inside eq. 4.6/4.7's shock branch) --
// factored out (Snorlax review finding) since three call sites
// (`hllc_pressure_function`, `hllc_pressure_function_derivative`, and
// the TSRS branch of `hllc_wave_speeds`) each independently needed this
// exact pair; computing it once per site removed three chances for the
// formula to silently drift apart.
template <class Scalar>
struct HllcShockImpedance
{
  Scalar a_k, b_k;
};

template <class Scalar>
CFE_HOST_DEVICE
HllcShockImpedance<Scalar> hllc_shock_impedance(Scalar gamma, Scalar rho_k, Scalar p_k)
{
  return HllcShockImpedance<Scalar>{Scalar(2) / ((gamma + Scalar(1)) * rho_k),
                                     (gamma - Scalar(1)) / (gamma + Scalar(1)) * p_k};
}

// Toro eq. 4.6/4.7: the exact pressure function for one side and its
// derivative, branching on shock (`p > p_k`) vs. rarefaction (`p <=
// p_k`) -- used ONLY to Newton-refine the ANRS seed below toward the
// exact two-shock/two-rarefaction `p*` (see this file's header comment
// for why the seed alone is not sufficient). Independently reimplemented
// here (not included from tests/verification/sod_shock_tube_exact.hpp,
// which has the same formulas) because production code under `src/`
// must not depend on a `tests/` header.
template <class Scalar>
CFE_HOST_DEVICE
Scalar hllc_pressure_function(Scalar gamma, Scalar p, Scalar rho_k, Scalar p_k, Scalar c_k)
{
  const HllcShockImpedance<Scalar> imp = hllc_shock_impedance(gamma, rho_k, p_k);
  if (p > p_k) {
    return (p - p_k) * sqrt(imp.a_k / (p + imp.b_k));
  }
  return (Scalar(2) * c_k / (gamma - Scalar(1))) *
         (pow(p / p_k, (gamma - Scalar(1)) / (Scalar(2) * gamma)) - Scalar(1));
}

template <class Scalar>
CFE_HOST_DEVICE
Scalar hllc_pressure_function_derivative(Scalar gamma, Scalar p, Scalar rho_k, Scalar p_k, Scalar c_k)
{
  const HllcShockImpedance<Scalar> imp = hllc_shock_impedance(gamma, rho_k, p_k);
  if (p > p_k) {
    return sqrt(imp.a_k / (imp.b_k + p)) * (Scalar(1) - (p - p_k) / (Scalar(2) * (imp.b_k + p)));
  }
  return (Scalar(1) / (rho_k * c_k)) * pow(p / p_k, -(gamma + Scalar(1)) / (Scalar(2) * gamma));
}

// The three HLLC wave speeds (`S_L`, `S_star`, `S_R`) for one Riemann
// problem -- see this file's header comment for the ANRS `p*` estimate
// and the eq. 10.37 `S_star` formula. Factored out (not inlined into
// `hllc_flux` below) specifically so test_hllc_flux.cpp can call the
// SAME formula the real flux uses for its exact-solver cross-check,
// rather than maintaining a second, duplicated copy of this arithmetic
// that could silently drift out of sync with it.
template <class Scalar>
struct HllcWaveSpeeds
{
  Scalar s_l, s_star, s_r;
};

template <class Scalar, std::size_t N, class Field>
CFE_HOST_DEVICE
HllcWaveSpeeds<Scalar> hllc_wave_speeds(const State<Scalar, N>& left_state,
                                                const State<Scalar, N>& right_state, Axis axis, const Field& field)
{
  // Floors (Violet review finding -- see hllc_star_state's own comment
  // for the full rationale): a MUSCL componentwise-reconstruction
  // artifact at a strong 2D multi-wave interaction can hand this
  // function a face state whose IMPLIED density/pressure is near-zero
  // or (after the artifact compounds one more stage) genuinely
  // negative. `hllc_floored_primitives` computes velocity/pressure from
  // the SAME floored density it returns (never via a `Field` accessor
  // that would recompute density itself, possibly from the raw,
  // unfloored value -- Enzo review finding: an earlier version of this
  // function used `field.velocity_component`/`field.pressure` directly
  // here, which did exactly that, letting a slightly-negative raw
  // density silently produce a wrong-but-finite-and-POSITIVE pressure
  // that passed the floor check undetected). No-op for any
  // well-separated, physically reasonable input -- only ever activates
  // in the near-vacuum/unphysical regime this fix specifically targets.
  const Scalar gamma = field.gamma;

  const HllcFlooredPrimitives<Scalar, N, Field> prim_l = hllc_floored_primitives(left_state, axis, field);
  const HllcFlooredPrimitives<Scalar, N, Field> prim_r = hllc_floored_primitives(right_state, axis, field);
  const Scalar rho_l = prim_l.rho;
  const Scalar rho_r = prim_r.rho;
  const Scalar u_l = prim_l.u;
  const Scalar u_r = prim_r.u;
  const Scalar p_l = prim_l.p;
  const Scalar p_r = prim_r.p;
  const Scalar c_l = sqrt(gamma * p_l / rho_l);
  const Scalar c_r = sqrt(gamma * p_r / rho_r);

  const Scalar rho_bar = Scalar(0.5) * (rho_l + rho_r);
  const Scalar c_bar = Scalar(0.5) * (c_l + c_r);
  const Scalar p_pvrs = Scalar(0.5) * (p_l + p_r) - Scalar(0.5) * (u_r - u_l) * rho_bar * c_bar;

  const Scalar p_min = p_l < p_r ? p_l : p_r;
  const Scalar p_max = p_l > p_r ? p_l : p_r;

  Scalar p0;
  if (p_max / p_min <= Scalar(2) && p_pvrs >= p_min && p_pvrs <= p_max) {
    // Weak wave: PVRS itself is adequate (Toro Sec. 9.5.2).
    p0 = p_pvrs;
  } else if (p_pvrs < p_min) {
    // Two-Rarefaction Riemann Solver (TRRS, Toro eq. 9.31) -- exact when
    // both nonlinear waves genuinely are rarefactions, which `p_pvrs`
    // landing below both pressures signals.
    const Scalar z = (gamma - Scalar(1)) / (Scalar(2) * gamma);
    const Scalar numerator = c_l + c_r - Scalar(0.5) * (gamma - Scalar(1)) * (u_r - u_l);
    const Scalar denominator = c_l / pow(p_l, z) + c_r / pow(p_r, z);
    // Guard `pow`'s base against going non-positive in a near-vacuum
    // edge case (an even stronger double rarefaction than
    // test_hllc_flux.cpp's own stress test) -- `numerator` can in
    // principle approach/cross zero there; clamped to a tiny positive
    // floor rather than left to risk a NaN from a negative base raised
    // to a non-integer power.
    const Scalar base = numerator > Scalar(1e-10) * denominator ? numerator / denominator : Scalar(1e-10);
    p0 = pow(base, Scalar(1) / z);
  } else {
    // Two-Shock Riemann Solver (TSRS, Toro eq. 9.41-9.43), `p_pvrs` as
    // the one-shot pressure fed into the shock impedance functions --
    // still non-iterative.
    const HllcShockImpedance<Scalar> imp_l = hllc_shock_impedance(gamma, rho_l, p_l);
    const HllcShockImpedance<Scalar> imp_r = hllc_shock_impedance(gamma, rho_r, p_r);
    const Scalar p_guess = p_pvrs > Scalar(0) ? p_pvrs : Scalar(0);
    const Scalar g_l = sqrt(imp_l.a_k / (p_guess + imp_l.b_k));
    const Scalar g_r = sqrt(imp_r.a_k / (p_guess + imp_r.b_k));
    p0 = (g_l * p_l + g_r * p_r - (u_r - u_l)) / (g_l + g_r);
  }

  // Newton-Raphson refinement, seeded from the ANRS estimate above --
  // see this file's header comment for why the seed alone is not
  // sufficient (the forward-step wall-reflection failure). A FIXED
  // iteration count (no convergence-gated early exit) so every
  // cell/thread does identical work regardless of how quickly its own
  // Riemann problem happens to converge -- GPU-warp-uniform by
  // construction, not merely by accident.
  constexpr int kNewtonIterations = 5;
  Scalar p = p0 > kTinyPressureFloor<Scalar> ? p0 : kTinyPressureFloor<Scalar>;
  for (int iteration = 0; iteration < kNewtonIterations; ++iteration) {
    const Scalar f = hllc_pressure_function(gamma, p, rho_l, p_l, c_l) +
                      hllc_pressure_function(gamma, p, rho_r, p_r, c_r) + (u_r - u_l);
    const Scalar f_prime = hllc_pressure_function_derivative(gamma, p, rho_l, p_l, c_l) +
                            hllc_pressure_function_derivative(gamma, p, rho_r, p_r, c_r);
    const Scalar p_new = p - f / f_prime;
    p = p_new > kTinyPressureFloor<Scalar> ? p_new : kTinyPressureFloor<Scalar>;
  }
  const Scalar p_star = p > Scalar(0) ? p : Scalar(0);

  const Scalar q_l =
      p_star <= p_l ? Scalar(1)
                    : sqrt(Scalar(1) + (gamma + Scalar(1)) / (Scalar(2) * gamma) * (p_star / p_l - Scalar(1)));
  const Scalar q_r =
      p_star <= p_r ? Scalar(1)
                    : sqrt(Scalar(1) + (gamma + Scalar(1)) / (Scalar(2) * gamma) * (p_star / p_r - Scalar(1)));

  const Scalar s_l = u_l - c_l * q_l;
  const Scalar s_r = u_r + c_r * q_r;
  const Scalar s_star = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) /
                         (rho_l * (s_l - u_l) - rho_r * (s_r - u_r));

  return HllcWaveSpeeds<Scalar>{s_l, s_star, s_r};
}

}  // namespace detail

// See this file's header comment for the full derivation/citations.
template <class Scalar, std::size_t N, class Field>
CFE_HOST_DEVICE
State<Scalar, N> hllc_flux(const State<Scalar, N>& left_state, const State<Scalar, N>& right_state, Axis axis,
                            const Field& field)
{
  const detail::HllcWaveSpeeds<Scalar> waves = detail::hllc_wave_speeds(left_state, right_state, axis, field);
  const Scalar s_l = waves.s_l;
  const Scalar s_star = waves.s_star;
  const Scalar s_r = waves.s_r;

  if (s_l >= Scalar(0)) {
    return field.physical_flux(left_state, axis);
  }
  if (s_star >= Scalar(0)) {
    const State<Scalar, N> q_star_l = detail::hllc_star_state(left_state, s_l, s_star, axis, field);
    const State<Scalar, N> flux_l = field.physical_flux(left_state, axis);
    State<Scalar, N> flux;
    for (Index c = 0; c < N; ++c) flux[c] = flux_l[c] + s_l * (q_star_l[c] - left_state[c]);
    return flux;
  }
  if (s_r >= Scalar(0)) {
    const State<Scalar, N> q_star_r = detail::hllc_star_state(right_state, s_r, s_star, axis, field);
    const State<Scalar, N> flux_r = field.physical_flux(right_state, axis);
    State<Scalar, N> flux;
    for (Index c = 0; c < N; ++c) flux[c] = flux_r[c] + s_r * (q_star_r[c] - right_state[c]);
    return flux;
  }
  return field.physical_flux(right_state, axis);
}

// Stateless functor wrapping hllc_flux into the same swappable shape
// `RusanovFlux` uses -- `FvmSolver`'s `NumericalFlux` template parameter
// accepts this with zero solver changes. See this file's header comment
// for the (narrower, Euler/EOS-specific) contract this requires from
// `Field` relative to `RusanovFlux`'s.
struct HllcFlux
{
  template <class Scalar, std::size_t N, class Field>
  CFE_HOST_DEVICE
  State<Scalar, N> operator()(const State<Scalar, N>& left_state, const State<Scalar, N>& right_state, Axis axis,
                               const Field& field) const
  {
    return hllc_flux(left_state, right_state, axis, field);
  }
};

}  // namespace cfe
