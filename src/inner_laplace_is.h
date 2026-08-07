// inner_laplace_is.h
//
// Inner-Laplace importance k-hat (gcol33/tulpa#303): the likelihood-agnostic
// reliability number for the inner layer.
//
// WHAT IT SCORES. The inner Gaussian at a fixed theta IS an importance
// proposal, and the joint log density log p(y | x, theta) + log p(x | theta) at
// that same theta IS the target it approximates. The Pareto-smoothed shape of
// that importance ratio is therefore a direct answer to "is the inner Gaussian
// a usable stand-in for the exact conditional posterior", the structural
// counterpart of the outer k-hat that scores the hyperparameter-grid
// integration around a FIXED inner Laplace.
//
// WHY THE PROBED SUBSPACE, NOT THE FIELD. Importance sampling degrades with
// dimension on its own, so a k-hat over all n_x latent coordinates would be a
// statement about n_x rather than about the approximation, and would be
// uninterpretable. The sampling therefore runs on the same probed indices the
// cubic term gamma_3 scores (every arm's fixed-effect coefficients by default),
// along the same Gaussian-conditional-mean curve x(t) = mode + (t / sigma_i^2)
// v_i (inner_laplace_probe.h), with the remaining coordinates integrated out by
// the Gaussian conditional: ONE dimension per probed index. That keeps every
// sampling problem 1-D -- the regime PSIS is reliable in -- and makes the
// number directly comparable to the gamma_3 for the same index, since both read
// the same curve through the same joint density.
//
// On that curve the Gaussian proposal is exactly N(mode_i, sigma_i^2), because
// v_i' H v_i = e_i' Sigma e_i = sigma_i^2. Parameterising by the standardized
// offset z = t / sigma_i makes the proposal N(0, 1), so the importance log
// ratio is
//
//   lr(z) = log p_joint(x(z sigma_i)) + z^2 / 2
//
// up to a constant common to every draw (the Gaussian normalizer and the
// Jacobian sigma_i), which drops under the self-normalization PSIS applies.
// This file returns log p_joint at the draws; the PSIS fit and the k-hat itself
// are the shared R primitives (tulpa_psis / .nested_is_pareto_k, R/psis.R), so
// there is one Pareto fit in the engine, not two.
//
// NO LIKELIHOOD DERIVATIVE IS NEEDED. The whole computation reads the joint
// density through the Newton loop's own penalized-objective closure, so it is
// available wherever a mode was found -- including a fully coupled multi-arm
// likelihood, where the cubic term has no separable per-observation sum to read
// and declines. That is the point: this is the floor that keeps the inner layer
// answerable when gamma_3 cannot be computed.
//
// WHY A FIXED DRAW COUNT. The outer k-hat pays one full inner Laplace solve per
// importance draw, which is why its budget is a user knob (`control$k_samples`).
// An inner draw is one evaluation of the penalized objective at a latent point
// -- O(N) with no factorization -- so the budget carries no comparable cost
// decision, and fixing it keeps the sample-size dependent reliability boundary
// (.ps_khat_threshold in R/psis.R) constant, hence comparable across fits. The
// draws come from an engine-owned deterministic generator rather than R's RNG,
// so requesting the diagnostic cannot perturb a fit's posterior draws, and the
// reported k-hat is reproducible run to run instead of flapping with the seed.

#ifndef TULPA_INNER_LAPLACE_IS_H
#define TULPA_INNER_LAPLACE_IS_H

#include "inner_laplace_probe.h"
#include "laplace_cholesky.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

// Importance draws per probed index. Above the GPD-fit floor .PSIS_MIN_EVAL
// (25) by an order of magnitude, and small enough that the whole diagnostic
// costs a few Newton iterations' worth of objective evaluations.
constexpr int INNER_IS_DRAWS = 256;

// N(0, 1) draws for the inner importance sample: splitmix64 uniforms through
// the Box-Muller transform from a fixed seed. Deterministic by design -- see
// the file header on why the draws are engine-owned rather than taken from R's
// stream.
inline std::vector<double> inner_is_std_normal_draws() {
  std::vector<double> out;
  out.reserve(INNER_IS_DRAWS);
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  auto next_u01 = [&state]() -> double {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    // (z >> 11) is uniform on {0, ..., 2^53 - 1}; the half-shift keeps the
    // result strictly inside (0, 1) so log(u) below is always finite.
    return (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  };
  while (static_cast<int>(out.size()) < INNER_IS_DRAWS) {
    const double u1 = next_u01();
    const double u2 = next_u01();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 6.283185307179586 * u2;
    out.push_back(r * std::cos(a));
    if (static_cast<int>(out.size()) < INNER_IS_DRAWS) {
      out.push_back(r * std::sin(a));
    }
  }
  return out;
}

struct InnerISOutcome {
  std::vector<double> z;          // the standardized proposal draws (n_draws)
  std::vector<double> log_joint;  // n_draws * n_probe, probe-major (column-major
                                  // as an [n_draws x n_probe] matrix)
  std::vector<double> sigma;      // n_probe conditional SDs; NaN where the
                                  // probe column could not be solved
  // Why NOTHING was computable, when nothing was. Empty when at least one index
  // produced a curve. Vocabulary shared with the outer k-hat's decline reasons
  // (.K_DECLINE_REASONS, R/psis.R): "no_probe_indices" is reported to R as
  // "not_applicable", a failed probe column as "degenerate_proposal".
  std::string declined;
};

// Evaluate the joint log density along the Gaussian-conditional-mean curve at
// each probed index.
//
// `eval_log_joint(x) -> double` is the Newton loop's own penalized-objective
// closure (log-likelihood + log-prior at latent x), so this routine carries no
// likelihood knowledge whatsoever and works identically for the single-arm and
// joint (including coupled) loops. `x_buf` is caller-supplied scratch sized
// n_x; it must hold `mode` on entry and is restored to it on return. The
// factor is reused as-is, with no refactorization.
template <typename EvalLogJoint>
inline InnerISOutcome compute_inner_is_curve(
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    EvalLogJoint eval_log_joint,
    Rcpp::NumericVector& x_buf,
    const std::vector<int>& probe_idx
) {
  const double NaN = std::numeric_limits<double>::quiet_NaN();
  InnerISOutcome out;
  const std::size_t P = probe_idx.size();
  out.sigma.assign(P, NaN);
  if (P == 0) { out.declined = "no_probe_indices"; return out; }

  out.z = inner_is_std_normal_draws();
  const int S = static_cast<int>(out.z.size());
  out.log_joint.assign(P * static_cast<std::size_t>(S), NaN);

  std::vector<double> rhs(n_x, 0.0), v(n_x, 0.0), z_work;
  if (!use_sparse) z_work.assign(n_x, 0.0);

  bool any_scored = false;
  for (std::size_t idx = 0; idx < P; idx++) {
    double sigma_i = NaN;
    if (!inner_probe_column(n_x, probe_idx[idx], chol, sparse_solver, use_sparse,
                            rhs, v, z_work, sigma_i)) {
      continue;
    }
    out.sigma[idx] = sigma_i;
    // x(z) = mode + (z sigma_i / sigma_i^2) v_i = mode + (z / sigma_i) v_i.
    const double inv_sigma = 1.0 / sigma_i;
    for (int s = 0; s < S; s++) {
      const double step = out.z[s] * inv_sigma;
      for (int k = 0; k < n_x; k++) x_buf[k] = mode[k] + step * v[k];
      out.log_joint[idx * static_cast<std::size_t>(S) + s] =
          eval_log_joint(x_buf);
    }
    any_scored = true;
  }

  for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];  // restore
  if (!any_scored) out.declined = "degenerate_proposal";
  return out;
}

} // namespace tulpa

#endif // TULPA_INNER_LAPLACE_IS_H
