// Vendored from gcol33/SIMP (cc1de1d) by vendor_simp.sh -- do not edit.
// Upstream: SIMP inst/include/simp/scheme.h. Edit there and re-vendor.

#ifndef SIMP_SCHEME_H
#define SIMP_SCHEME_H

#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace simp {

// Elementary flow of a separable Hamiltonian H(q, p) = U(q) + K(p).
// A Kick advances momentum along the force (p += c * eps * force, with
// force = -dU/dq); a Drift advances position (q += c * eps * Minv * p).
enum class Op { Drift, Kick };

// A symplectic splitting integrator, stored as the sequence of elementary
// flows applied in one step. Leapfrog is kick-drift-kick. The kick
// coefficients sum to one and the drift coefficients sum to one, and the
// sequence is palindromic, which makes the step consistent and time
// reversible. Higher-order members are built from leapfrog by composition, so
// the coefficients are constructed, never tabulated.
struct Scheme {
  std::string name;
  int order;
  std::vector<std::pair<Op, double>> ops;

  int n_kicks() const {
    int k = 0;
    for (const auto& op : ops) if (op.first == Op::Kick) ++k;
    return k;
  }

  // Force evaluations per step along a trajectory. Consecutive steps share the
  // boundary kick (first-same-as-last), so a step costs one fewer gradient
  // than its kick count.
  int gradient_evals() const {
    int k = n_kicks();
    return k > 1 ? k - 1 : k;
  }
};

// Leapfrog (Stormer-Verlet), kick-drift-kick, order two.
inline Scheme leapfrog() {
  return Scheme{"leapfrog", 2,
                {{Op::Kick, 0.5}, {Op::Drift, 1.0}, {Op::Kick, 0.5}}};
}

// Two-stage order-two integrator (two gradients per step),
// kick-drift-kick-drift-kick with kick weights b, 1-2b, b and drift weights
// 1/2, 1/2. The free weight b tunes the energy error; the palindrome makes the
// step consistent and time reversible for any b. See minerror2() for the
// Gaussian-optimal fixed choice and adapt.h for a step-size-adapted choice.
inline Scheme two_stage(double b, const std::string& name = "two_stage") {
  return Scheme{name, 2,
                {{Op::Kick, b}, {Op::Drift, 0.5}, {Op::Kick, 1.0 - 2.0 * b},
                 {Op::Drift, 0.5}, {Op::Kick, b}}};
}

// Three-stage order-two integrator (three gradients per step),
// kick-drift-kick-drift-kick-drift-kick with kick weights b, 1/2-b, 1/2-b, b
// and drift weights a, 1-2a, a. The two free weights (b, a) leave enough slack
// to keep the energy error small across a band of step sizes rather than at a
// single one; see adapt.h.
inline Scheme three_stage(double b, double a,
                          const std::string& name = "three_stage") {
  return Scheme{name, 2,
                {{Op::Kick, b}, {Op::Drift, a}, {Op::Kick, 0.5 - b},
                 {Op::Drift, 1.0 - 2.0 * a}, {Op::Kick, 0.5 - b},
                 {Op::Drift, a}, {Op::Kick, b}}};
}

// Two-stage member with the free weight fixed at b = (3 - sqrt(5)) / 4, the
// value that minimizes the leading energy-error term on a harmonic oscillator.
// A Gaussian target has harmonic dynamics, and mass adaptation drives a target
// toward an isotropic Gaussian, so near the adapted optimum this integrator
// carries almost no energy error and admits large steps without the stability
// cliff of the high-order composition schemes.
inline Scheme minerror2() {
  return two_stage((3.0 - std::sqrt(5.0)) / 4.0, "minerror2");
}

// Scale a scheme so one step covers a fraction w of the step size.
inline Scheme scale(const Scheme& s, double w) {
  Scheme out = s;
  for (auto& op : out.ops) op.second *= w;
  return out;
}

// Apply A then B, merging the seam when both sides are the same kind (two
// consecutive kicks, or two consecutive drifts, add). The result is again a
// valid splitting.
inline Scheme concat(const Scheme& A, const Scheme& B) {
  Scheme out;
  out.ops = A.ops;
  if (!out.ops.empty() && !B.ops.empty() &&
      out.ops.back().first == B.ops.front().first) {
    out.ops.back().second += B.ops.front().second;
    out.ops.insert(out.ops.end(), B.ops.begin() + 1, B.ops.end());
  } else {
    out.ops.insert(out.ops.end(), B.ops.begin(), B.ops.end());
  }
  return out;
}

// Yoshida triple jump: three scaled copies of a symmetric scheme of even order
// n raise the order to n+2, with w1 = 1 / (2 - 2^(1/(n+1))) on the outer copies
// and w0 = 1 - 2 * w1 on the inner one. The weights are functions of n, not
// tabulated constants.
inline Scheme triple_jump(const Scheme& base, const std::string& name) {
  const double n = static_cast<double>(base.order);
  const double w1 = 1.0 / (2.0 - std::pow(2.0, 1.0 / (n + 1.0)));
  const double w0 = 1.0 - 2.0 * w1;
  Scheme out = concat(concat(scale(base, w1), scale(base, w0)), scale(base, w1));
  out.name = name;
  out.order = base.order + 2;
  return out;
}

// The built-in family: leapfrog plus the Yoshida members it generates.
inline std::vector<Scheme> registry() {
  Scheme lf = leapfrog();
  Scheme y4 = triple_jump(lf, "yoshida4");
  Scheme y6 = triple_jump(y4, "yoshida6");
  Scheme y8 = triple_jump(y6, "yoshida8");
  return {lf, minerror2(), y4, y6, y8};
}

inline Scheme scheme_by_name(const std::string& name) {
  for (const Scheme& s : registry()) {
    if (s.name == name) return s;
  }
  throw std::invalid_argument("unknown integrator: " + name);
}

}  // namespace simp

#endif  // SIMP_SCHEME_H
