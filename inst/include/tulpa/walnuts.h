// walnuts.h
// WALNUTS transition: NUTS with a locally adapted step size per macro step
// (Bou-Rabee, Carpenter, Kleppe, Liu, Marsden & Shaw 2025, arXiv 2506.18746).
//
// Each macro step of the orbit is integrated with the coarsest leapfrog
// subdivision -- min_micro_steps * 2^k micro steps of size step / 2^k -- whose
// joint log density moves by at most max_error, and is kept only if the
// subdivision is the one the reversed step would pick (no coarser count also
// meets the tolerance from the end state with the momentum flipped). The
// selection rule is a deterministic function of the state, so together with
// that check the orbit map is an involution and the kernel leaves the target
// invariant. This is the WALNUTS-D variant of section 4.1, ported from the
// reference implementation in flatironinstitute/walnutpie (walnuts.hpp,
// transition_w; MIT, see inst/COPYRIGHTS), generalized from a diagonal to any
// metric the caller's model supplies.
//
// Within a subtree states are selected by the Barker rule, at the top level by
// the biased-progressive Metropolis rule. A subtree whose leaf cannot be built
// (no subdivision within tolerance, or a non-reversible one) or that contains a
// sub-U-turn ends the orbit and is discarded, as in NUTS.
//
// The leapfrog is the plain Stormer-Verlet step: the error-controlled
// subdivision is defined on it.
//
// The transition is a template over the caller's model, so the engine's chain
// driver and a model package's own sampler run the same code. `Model` provides
//
//   void   gradient(const std::vector<double>& theta,
//                   std::vector<double>& grad, double* log_density);
//   double kinetic_energy(const double* rho) const;
//   void   inv_mass_times_p(const double* rho, double* out) const;
//   void   drift(double step, double* theta, const double* rho,
//                double* scratch) const;      // theta += step * M^-1 rho
//   template <class Rng> void sample_momentum(double* rho, Rng& rng) const;
//
// with `scratch` a length-n buffer the drift may use.

#ifndef TULPA_WALNUTS_H
#define TULPA_WALNUTS_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "walnuts_config.h"

namespace tulpa {

// One end (or the selected point) of an orbit segment.
struct WalnutsState {
  std::vector<double> theta;
  std::vector<double> rho;
  std::vector<double> grad;
  double logp_pos = 0.0;    // target log density at theta
  double logp_joint = 0.0;  // logp_pos - kinetic energy(rho)

  void resize(int n) { theta.resize(n); rho.resize(n); grad.resize(n); }
};

// A contiguous orbit segment: its two ends, the state selected from it, and
// the log of the summed joint densities over its states.
struct WalnutsSpan {
  WalnutsState bk;
  WalnutsState fw;
  WalnutsState select;  // rho unused
  double logp = 0.0;

  void resize(int n) { bk.resize(n); fw.resize(n); select.resize(n); }
};

// Per-chain buffers, sized once: one temporary span per tree level, the
// accumulated orbit, the segment being added, and the macro-step scratch.
struct WalnutsWorkspace {
  int n = 0;
  std::vector<WalnutsSpan> level_tmp;  // index = subtree depth
  WalnutsSpan accum;
  WalnutsSpan next;
  WalnutsState step_state;  // macro-step result
  WalnutsState rev_state;   // reversibility-check scratch
  std::vector<double> diff;
  std::vector<double> diff_sharp;
  std::vector<double> drift_scratch;

  // Per-transition tallies.
  int n_grad = 0;
  double accept_sum = 0.0;
  int accept_count = 0;
  bool divergent = false;

  void init(int n_params, int max_depth) {
    n = n_params;
    level_tmp.resize(static_cast<std::size_t>(max_depth) + 1);
    for (auto& s : level_tmp) s.resize(n);
    accum.resize(n);
    next.resize(n);
    step_state.resize(n);
    rev_state.resize(n);
    diff.resize(n);
    diff_sharp.resize(n);
    drift_scratch.resize(n);
  }
};

struct WalnutsTransitionResult {
  int depth = 0;           // doublings in the final orbit
  int n_grad = 0;          // gradient evaluations, reversibility checks included
  double mean_accept = 0;  // mean first-subdivision acceptance, for step-size adaptation
  bool divergent = false;  // some macro step met no subdivision within tolerance
  double H0 = 0.0;         // starting Hamiltonian
};

namespace walnuts_detail {

inline double log_sum_exp2(double x1, double x2) {
  if (std::isnan(x1) || std::isnan(x2)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double m = std::fmax(x1, x2);
  if (std::isinf(m)) return m;
  return m + std::log(std::exp(x1 - m) + std::exp(x2 - m));
}

template <class Model, class Rng>
struct Ctx {
  const WalnutsConfig& cfg;
  Model& model;
  WalnutsWorkspace& ws;
  Rng& rng;
  double step;
};

template <class C>
void eval_gradient(C& c, WalnutsState& s) {
  c.model.gradient(s.theta, s.grad, &s.logp_pos);
  c.ws.n_grad++;
}

// `num_steps` leapfrog steps of signed size `step` from s, in place; sets
// s.logp_pos / s.logp_joint at the end point.
template <class C>
void leapfrog(C& c, WalnutsState& s, double step, int num_steps) {
  const int n = c.ws.n;
  const double half = 0.5 * step;
  for (int k = 0; k < num_steps; k++) {
    for (int i = 0; i < n; i++) s.rho[i] += half * s.grad[i];
    c.model.drift(step, s.theta.data(), s.rho.data(),
                  c.ws.drift_scratch.data());
    eval_gradient(c, s);
    for (int i = 0; i < n; i++) s.rho[i] += half * s.grad[i];
  }
  s.logp_joint = s.logp_pos - c.model.kinetic_energy(s.rho.data());
}

// True when no coarser subdivision reaches the end state's tolerance from the
// end state with the momentum flipped, i.e. the reversed macro step would pick
// the same subdivision.
template <class C>
bool reversible(C& c, const WalnutsState& end, double step, int num_steps) {
  if (num_steps == 1) return true;
  WalnutsState& r = c.ws.rev_state;
  const int n = c.ws.n;
  while (num_steps >= 2 * c.cfg.min_micro_steps) {
    r.theta = end.theta;
    for (int i = 0; i < n; i++) r.rho[i] = -end.rho[i];
    r.grad = end.grad;
    num_steps /= 2;
    step *= 2.0;
    leapfrog(c, r, step, num_steps);
    if (std::fabs(r.logp_joint - end.logp_joint) <= c.cfg.max_error) {
      return false;
    }
  }
  return true;
}

// One macro step from the forward (D = +1) or backward (D = -1) end of `from`,
// written as a one-state span into `out`. False when no subdivision meets the
// tolerance (a divergence) or the one that does is not reversible.
template <class C>
bool build_leaf(C& c, int D, const WalnutsSpan& from, WalnutsSpan& out) {
  const WalnutsState& start = (D > 0) ? from.fw : from.bk;
  WalnutsState& s = c.ws.step_state;
  double step = (D > 0) ? c.step : -c.step;
  int num_steps = c.cfg.min_micro_steps;
  for (int halvings = 0; halvings < c.cfg.max_step_halvings;
       ++halvings, num_steps *= 2, step *= 0.5) {
    s.theta = start.theta;
    s.rho = start.rho;
    s.grad = start.grad;
    leapfrog(c, s, step, num_steps);
    const double err = std::fabs(start.logp_joint - s.logp_joint);
    if (num_steps == c.cfg.min_micro_steps) {
      const double a = std::exp(-err);
      c.ws.accept_sum += std::isfinite(a) ? a : 0.0;
      c.ws.accept_count++;
    }
    if (err <= c.cfg.max_error) {
      if (!reversible(c, s, step, num_steps)) return false;
      out.bk.theta = s.theta;  out.bk.rho = s.rho;  out.bk.grad = s.grad;
      out.bk.logp_pos = s.logp_pos;  out.bk.logp_joint = s.logp_joint;
      out.fw.theta = s.theta;  out.fw.rho = s.rho;  out.fw.grad = s.grad;
      out.fw.logp_pos = s.logp_pos;  out.fw.logp_joint = s.logp_joint;
      out.select.theta = s.theta;  out.select.grad = s.grad;
      out.select.logp_pos = s.logp_pos;
      out.logp = s.logp_joint;
      return true;
    }
  }
  c.ws.divergent = true;
  return false;
}

// U-turn between `first` and `second`, where `second` extends `first` in
// direction D: the endpoint momenta against the metric-scaled span.
template <class C>
bool uturn(C& c, int D, const WalnutsSpan& first, const WalnutsSpan& second) {
  const WalnutsSpan& span_bk = (D > 0) ? first : second;
  const WalnutsSpan& span_fw = (D > 0) ? second : first;
  const int n = c.ws.n;
  for (int i = 0; i < n; i++) {
    c.ws.diff[i] = span_fw.fw.theta[i] - span_bk.bk.theta[i];
  }
  c.model.inv_mass_times_p(c.ws.diff.data(), c.ws.diff_sharp.data());
  double dot_fw = 0.0, dot_bk = 0.0;
  for (int i = 0; i < n; i++) {
    dot_fw += span_fw.fw.rho[i] * c.ws.diff_sharp[i];
    dot_bk += span_bk.bk.rho[i] * c.ws.diff_sharp[i];
  }
  return dot_fw < 0.0 || dot_bk < 0.0;
}

// Fold `added` (extending `old` in direction D) into `old`. Barker selection
// inside a subtree, Metropolis (biased progressive) at the top level.
template <class C>
void combine(C& c, int D, bool metropolis, WalnutsSpan& old,
             const WalnutsSpan& added) {
  const double logp_total = log_sum_exp2(old.logp, added.logp);
  const double log_denominator = metropolis ? old.logp : logp_total;
  std::uniform_real_distribution<double> unif01(0.0, 1.0);
  if (std::log(unif01(c.rng)) < added.logp - log_denominator) {
    old.select.theta = added.select.theta;
    old.select.grad = added.select.grad;
    old.select.logp_pos = added.select.logp_pos;
  }
  WalnutsState& end = (D > 0) ? old.fw : old.bk;
  const WalnutsState& new_end = (D > 0) ? added.fw : added.bk;
  end.theta = new_end.theta;
  end.rho = new_end.rho;
  end.grad = new_end.grad;
  end.logp_pos = new_end.logp_pos;
  end.logp_joint = new_end.logp_joint;
  old.logp = logp_total;
}

// 2^depth states extending `from` in direction D, into `out`. False when a
// leaf fails or a sub-U-turn appears. `out` must not alias `from` or
// ws.level_tmp[depth]; the recursion uses level_tmp[depth] as the second half.
template <class C>
bool build_span(C& c, int D, int depth, const WalnutsSpan& from,
                WalnutsSpan& out) {
  if (depth == 0) return build_leaf(c, D, from, out);
  if (!build_span(c, D, depth - 1, from, out)) return false;
  WalnutsSpan& second = c.ws.level_tmp[depth];
  if (!build_span(c, D, depth - 1, out, second)) return false;
  if (uturn(c, D, out, second)) return false;
  combine(c, D, /*metropolis=*/false, out, second);
  return true;
}

}  // namespace walnuts_detail

// One WALNUTS transition from (q, grad, log_post), which are overwritten with
// the selected state. `step` is the macro step size; `ws` must have been
// initialised for q.size() parameters and at least `max_depth` levels.
template <class Model, class Rng>
WalnutsTransitionResult walnuts_transition(
    std::vector<double>& q, std::vector<double>& grad, double& log_post,
    double step, int max_depth, const WalnutsConfig& cfg, Model& model,
    WalnutsWorkspace& ws, Rng& rng) {
  walnuts_detail::Ctx<Model, Rng> c{cfg, model, ws, rng, step};
  ws.n_grad = 0;
  ws.accept_sum = 0.0;
  ws.accept_count = 0;
  ws.divergent = false;

  WalnutsSpan& acc = ws.accum;
  WalnutsState& init = acc.fw;
  init.theta = q;
  init.grad = grad;
  model.sample_momentum(init.rho.data(), rng);
  init.logp_pos = log_post;
  init.logp_joint = log_post - model.kinetic_energy(init.rho.data());
  acc.bk.theta = init.theta;  acc.bk.rho = init.rho;  acc.bk.grad = init.grad;
  acc.bk.logp_pos = init.logp_pos;  acc.bk.logp_joint = init.logp_joint;
  acc.select.theta = q;
  acc.select.grad = grad;
  acc.select.logp_pos = log_post;
  acc.logp = init.logp_joint;

  WalnutsTransitionResult res;
  res.H0 = -init.logp_joint;

  std::uniform_int_distribution<int> dir_dist(0, 1);
  int depth = 0;
  while (depth < max_depth) {
    const int D = dir_dist(rng) ? 1 : -1;
    if (!walnuts_detail::build_span(c, D, depth, acc, ws.next)) break;
    const bool made_uturn = walnuts_detail::uturn(c, D, acc, ws.next);
    walnuts_detail::combine(c, D, /*metropolis=*/true, acc, ws.next);
    ++depth;
    if (made_uturn) break;
  }

  q = acc.select.theta;
  grad = acc.select.grad;
  log_post = acc.select.logp_pos;

  res.depth = depth;
  res.n_grad = ws.n_grad;
  res.mean_accept = ws.accept_count > 0 ? ws.accept_sum / ws.accept_count : 0.0;
  res.divergent = ws.divergent;
  return res;
}

}  // namespace tulpa

#endif  // TULPA_WALNUTS_H
