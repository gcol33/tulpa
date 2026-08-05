// inner_laplace_skew.h
//
// Inner-Laplace skewness diagnostic: the cubic (skewness) term of the
// simplified-Laplace correction, Rue, Martino & Chopin (2009) JRSSB 71(2)
// Sec 3.2.3, eq. (18)-(21), generalized from their augmented representation
// (x_j == eta_j, one latent coordinate per observation, so removing a row of
// the joint precision has a clean interpretation) to tulpa's general
// eta = compute_eta(x) linear-predictor representation, where one latent
// component can load onto several observations and vice versa.
//
// DERIVATION. For a fixed latent index i, standardize x_i^(s) = (x_i -
// mu_i)/sigma_i and follow the Gaussian-conditional-mean curve
// x_{-i}(x_i) = E_piG(x_{-i} | x_i), which is linear in x_i (standard
// Gaussian regression) and passes through the joint mode at x_i = mu_i.
// Taylor-expanding the JOINT log density (the numerator of eq. 12) along
// this curve: the linear term vanishes because the curve passes through the
// mode of the full joint density; the quadratic term reproduces the
// Gaussian approximation by construction; the cubic term is
//
//   (1/6) (x_i^(s))^3 * sum_j l_j'''(eta_j) * b_ij^3
//
// where b_ij = d(eta_j)/d(x_i) * sigma_i is the (standardized) response of
// eta_j to a unit move of x_i along the curve, and l_j''' is the third
// derivative of observation j's log-likelihood at the mode. This numerator
// term is the WHOLE cubic coefficient: the paper's denominator expansion
// (eq. 19-20) is carried to first order only and contributes solely to the
// (unimplemented here) location-shift term gamma_1, never to gamma_3 -- see
// the scope note below.
//
// Writing H for the converged Newton Hessian (posterior precision) and
// Sigma = H^{-1}: b_ij * sigma_i = d(eta_j)/d(x_i) * sigma_i^2 is exactly
// the j-th entry of u_i := eta(mode + Sigma e_i) - eta(mode), an AFFINE
// difference (so any additive offset baked into compute_eta cancels
// exactly). With v_i solving H v_i = e_i (so v_i = Sigma e_i, sigma_i^2 =
// v_i[i]) and u_i = compute_eta(mode + v_i) - compute_eta(mode):
//
//   gamma_3(i) = sigma_i^{-3} * sum_j l_j'''(eta_j) * u_{i,j}^3
//
// gamma_3(i) is the leading-order (Edgeworth-type) estimate of the skewness
// (third standardized cumulant) of the true conditional posterior
// pi(x_i | theta, y) relative to the Gaussian (inner Laplace)
// approximation: for a nearly-Gaussian log-density with cumulants
// (0, 1, kappa_3, ...), log f(z) = -z^2/2 + (kappa_3/6) (z^3 - 3z) + ...,
// so the coefficient of z^3/6 IS the third cumulant to leading order
// (Barndorff-Nielsen & Cox 1989, saddlepoint / Edgeworth expansions).
//
// VERIFICATION AGAINST THE PAPER. In the paper's own augmented
// representation (x_j == eta_j literally), u_{i,j} = v_i[j] = Sigma_ij, so
// b_ij = Sigma_ij / sigma_i. Their a_ij is defined by
// (E[x_j|x_i] - mu_j)/sigma_j = a_ij (x_i - mu_i)/sigma_i, which gives
// a_ij = Sigma_ij / (sigma_i sigma_j), i.e. sigma_j a_ij = Sigma_ij/sigma_i =
// b_ij: this formula reduces EXACTLY to eq. (21)'s gamma_3 = sum_j d_j^(3)
// {sigma_j a_ij}^3 in that special case.
//
// SCOPE. Only gamma_3 (skewness) is computed, not the paper's gamma_1
// (location-shift, eq. 21's first line) or a quartic (kurtosis) term. gamma_1
// needs the denominator log-determinant's response to a likelihood-curvature
// perturbation that is DIAGONAL only in the paper's augmented representation
// (removing row/col i from a matrix that is prior-precision-plus-diagonal);
// in tulpa's general eta = A(theta) x representation the same perturbation is
// a rank-deficient, non-diagonal Hessian change (H_{-i,-i}(x_i) =
// A_{-i}' diag(w(eta(x_i))) A_{-i} + P_{-i,-i}), whose log-determinant
// response needs the conditional covariance of eta given x_i (not just the
// column Sigma e_i this file already computes) -- a genuinely separate,
// larger derivation, not attempted here rather than shipped as an unverified
// guess. A closed-form quartic (kurtosis) term is not part of the paper's
// simplified-Laplace method either: Sec 3.2.3's own discussion of symmetric
// heavy-tailed cases (Student-t) routes to a different, more expensive
// numerical procedure (the spline-corrected Gaussian, eq. 17) rather than a
// closed-form fourth-order term, so one is not fabricated here under the
// paper's name.

#ifndef TULPA_INNER_LAPLACE_SKEW_H
#define TULPA_INNER_LAPLACE_SKEW_H

#include "laplace_cholesky.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace tulpa {

struct InnerSkewOutcome {
  std::vector<double> gamma3;      // one entry per requested index, NaN = not computable there
  int n_nonfinite_dropped = 0;     // (i, j) contributions skipped for a non-finite l'''_j
};

// curvature3_fn(j, eta_j) -> l_j'''(eta_j) at the mode, or NaN if this
// observation's likelihood has no registered third derivative.
// compute_eta_fn(x, eta_out): the SAME closure convention laplace_newton_ll
// already uses (in-place write into eta_out, sized n_eta).
//
// x_buf / eta_buf0 / eta_buf1 are caller-supplied scratch (sized n_x, n_eta,
// n_eta respectively); x_buf must hold `mode` on entry and is restored to it
// on return. Reuses the live factor (chol, dense fallback; or sparse_solver
// when use_sparse) without refactorizing -- the same pattern the
// inv_block_layout diagonal-block extraction in laplace_newton.h uses.
template <typename ComputeEtaFn>
inline InnerSkewOutcome compute_inner_skew_gamma3(
    int n_x, int n_eta,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    ComputeEtaFn compute_eta_fn,
    Rcpp::NumericVector& x_buf,
    Rcpp::NumericVector& eta_buf0,
    Rcpp::NumericVector& eta_buf1,
    const std::function<double(int, double)>& curvature3_fn,
    const std::vector<int>& probe_idx
) {
  InnerSkewOutcome out;
  out.gamma3.assign(probe_idx.size(), std::numeric_limits<double>::quiet_NaN());
  // No oracle at all (e.g. a coupled multi-process spec build_spec_curvature3_fn
  // declines) -- every index is "not computable", not "zero skew". Without this
  // early return the per-index loop below would see every l3[j] as NaN, drop
  // every contribution, and divide 0/sigma_i^3 = 0 into gamma3 -- a silently
  // wrong "perfectly Gaussian" reading instead of NaN.
  if (!curvature3_fn || probe_idx.empty()) return out;

  std::vector<double> rhs(n_x, 0.0), v(n_x, 0.0), z_work;
  if (!use_sparse) z_work.assign(n_x, 0.0);

  // eta at the mode -- x_buf already holds `mode` on entry.
  compute_eta_fn(x_buf, eta_buf0);
  std::vector<double> l3(n_eta, std::numeric_limits<double>::quiet_NaN());
  for (int j = 0; j < n_eta; j++) l3[j] = curvature3_fn(j, eta_buf0[j]);

  for (std::size_t idx = 0; idx < probe_idx.size(); idx++) {
    int i = probe_idx[idx];
    if (i < 0 || i >= n_x) continue;

    std::fill(rhs.begin(), rhs.end(), 0.0);
    rhs[i] = 1.0;
    bool solved;
    if (use_sparse) {
      sparse_solver.solve(rhs.data(), v.data(), n_x);
      solved = true;
      for (int k = 0; k < n_x; k++) if (!std::isfinite(v[k])) { solved = false; break; }
    } else {
      solved = chol_substitute_raw(chol.L.data(), n_x, rhs.data(), v.data(), z_work.data());
    }
    if (!solved) continue;

    double sigma2_i = v[i];
    if (!(sigma2_i > 0.0) || !std::isfinite(sigma2_i)) continue;  // unidentified / degenerate
    double sigma_i = std::sqrt(sigma2_i);

    for (int k = 0; k < n_x; k++) x_buf[k] = mode[k] + v[k];
    compute_eta_fn(x_buf, eta_buf1);

    double acc = 0.0;
    bool any_finite = false;
    for (int j = 0; j < n_eta; j++) {
      double u = eta_buf1[j] - eta_buf0[j];
      if (u == 0.0) continue;
      double l3j = l3[j];
      if (!std::isfinite(l3j)) { out.n_nonfinite_dropped++; continue; }
      acc += l3j * u * u * u;
      any_finite = true;
    }
    // Leave gamma3[idx] at its NaN default when NOTHING finite contributed --
    // otherwise an index whose every observation lacks a third derivative would
    // silently read as acc/sigma_i^3 == 0 ("perfectly Gaussian") rather than
    // "not computable".
    if (any_finite) out.gamma3[idx] = acc / (sigma_i * sigma2_i);  // sigma_i^-3
  }

  for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];  // restore
  return out;
}

// Joint-arm generalization of compute_inner_skew_gamma3 above, for
// laplace_newton_joint.h / laplace_newton_joint_sparse.h's multi-arm Newton
// loops. The derivation in the file header assumes a log-likelihood that is a
// single separable sum `sum_j l_j(eta_j)`; a SEPARABLE joint fit (every arm's
// per-observation contributions summed, no CellCouplingSpec term -- the only
// coupling tulpa's own production src/ ever registers, see
// laplace_spec_curvature3.h's scope note) is exactly that sum with j ranging
// over the union of (arm, observation) pairs instead of a single arm's rows,
// so the formula and its correctness proof carry over unchanged -- this is
// the same sum, not a new derivation. A genuinely COUPLED arm (its per-obs
// sum replaced by a CellCouplingSpec's evaluate_cell() term, `skip_arm[k] ==
// true`) is a different, non-separable log-density this formula does not
// cover; `curvature3_fns[k]` must be the empty std::function for such arms so
// their observations are dropped rather than silently scored against the
// wrong (unused) per-obs likelihood -- see build_joint_curvature3_fns in
// laplace_newton_joint.h, which enforces exactly this.
//
// eta_buf0 / eta_buf1 are the per-arm scratch (NewtonScratchJoint's `etas` /
// `etas_tmp`), one Rcpp::NumericVector per arm sized to that arm's N.
// curvature3_fns has one entry per arm (parallel to eta_buf0); an empty entry
// declines that whole arm (every one of its observations contributes NaN,
// i.e. gets dropped from the sum -- same per-entry semantics as a single
// non-finite l_j''' in the single-arm version above).
template <typename ComputeEtaJointFn>
inline InnerSkewOutcome compute_inner_skew_gamma3_joint(
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    ComputeEtaJointFn compute_eta_joint_fn,
    Rcpp::NumericVector& x_buf,
    std::vector<Rcpp::NumericVector>& eta_buf0,
    std::vector<Rcpp::NumericVector>& eta_buf1,
    const std::vector<std::function<double(int, double)>>& curvature3_fns,
    const std::vector<int>& probe_idx
) {
  InnerSkewOutcome out;
  out.gamma3.assign(probe_idx.size(), std::numeric_limits<double>::quiet_NaN());
  const int n_arms = static_cast<int>(eta_buf0.size());
  bool any_oracle = false;
  for (const auto& f : curvature3_fns) if (f) { any_oracle = true; break; }
  if (!any_oracle || probe_idx.empty()) return out;

  std::vector<double> rhs(n_x, 0.0), v(n_x, 0.0), z_work;
  if (!use_sparse) z_work.assign(n_x, 0.0);

  // eta at the mode, per arm -- x_buf already holds `mode` on entry.
  compute_eta_joint_fn(x_buf, eta_buf0);
  std::vector<std::vector<double>> l3(n_arms);
  for (int k = 0; k < n_arms; k++) {
    const int Nk = eta_buf0[k].size();
    l3[k].assign(Nk, std::numeric_limits<double>::quiet_NaN());
    if (k < static_cast<int>(curvature3_fns.size()) && curvature3_fns[k]) {
      for (int j = 0; j < Nk; j++) l3[k][j] = curvature3_fns[k](j, eta_buf0[k][j]);
    }
  }

  for (std::size_t idx = 0; idx < probe_idx.size(); idx++) {
    int i = probe_idx[idx];
    if (i < 0 || i >= n_x) continue;

    std::fill(rhs.begin(), rhs.end(), 0.0);
    rhs[i] = 1.0;
    bool solved;
    if (use_sparse) {
      sparse_solver.solve(rhs.data(), v.data(), n_x);
      solved = true;
      for (int k = 0; k < n_x; k++) if (!std::isfinite(v[k])) { solved = false; break; }
    } else {
      solved = chol_substitute_raw(chol.L.data(), n_x, rhs.data(), v.data(), z_work.data());
    }
    if (!solved) continue;

    double sigma2_i = v[i];
    if (!(sigma2_i > 0.0) || !std::isfinite(sigma2_i)) continue;  // unidentified / degenerate
    double sigma_i = std::sqrt(sigma2_i);

    for (int k = 0; k < n_x; k++) x_buf[k] = mode[k] + v[k];
    compute_eta_joint_fn(x_buf, eta_buf1);

    double acc = 0.0;
    bool any_finite = false;
    for (int k = 0; k < n_arms; k++) {
      const int Nk = eta_buf0[k].size();
      for (int j = 0; j < Nk; j++) {
        double u = eta_buf1[k][j] - eta_buf0[k][j];
        if (u == 0.0) continue;
        double l3kj = l3[k][j];
        if (!std::isfinite(l3kj)) { out.n_nonfinite_dropped++; continue; }
        acc += l3kj * u * u * u;
        any_finite = true;
      }
    }
    if (any_finite) out.gamma3[idx] = acc / (sigma_i * sigma2_i);  // sigma_i^-3
  }

  for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];  // restore
  return out;
}

} // namespace tulpa

#endif // TULPA_INNER_LAPLACE_SKEW_H
