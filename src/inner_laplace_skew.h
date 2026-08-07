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
// UNITS WITH SEVERAL LINEAR PREDICTORS. The step from the joint log density to
// `sum_j l_j'''(eta_j) u_j^3` above assumes the data log-likelihood is a
// separable sum of one-eta terms. A unit that reads several linear predictors at
// once -- a zero-inflation mixture's (count, zi) pair, a CellCouplingSpec cell's
// arms -- has no such per-eta third derivative, but the expansion itself is
// unchanged: it is still the cubic Taylor coefficient along the same curve, and
// only the contraction widens, to
//
//   gamma_3(i) = sigma_i^{-3} sum_units sum_{a,b,c} T^{abc} u^a u^b u^c
//
// with T^{abc} the unit's third derivative in its linear predictors. The
// separable case is K = 1 coordinates per unit, where the triple sum has one
// term, T^{111} = l''', and the formula reduces to the line above term for term.
// curvature3_contract.h derives and implements the wider contraction; the two
// oracle shapes reach this file through Curvature3Oracle (per-observation) and
// CellCubic3Fn (per coupled cell).
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

#include "curvature3_contract.h"
#include "inner_laplace_probe.h"
#include "laplace_cholesky.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

// Cubic contraction over every coupled cell of a CellCouplingSpec fit, given the
// per-arm eta at the mode and at the probed point mode + v_i (so the direction is
// their difference). Built by build_cell_curvature3_tensor (cell_curvature3.h);
// declared here because this is where a joint fit stores it.
using CellCubic3Fn = std::function<double(const std::vector<Rcpp::NumericVector>&,
                                          const std::vector<Rcpp::NumericVector>&)>;

struct InnerSkewOutcome {
  std::vector<double> gamma3;      // one entry per requested index, NaN = not computable there
  int n_nonfinite_dropped = 0;     // (i, j) contributions skipped for a non-finite l'''_j
  // Why NOTHING was computable, when nothing was (gcol33/tulpa#296). Empty
  // when at least one index scored. A NaN says only "not computable"; the
  // reasons behind it are not interchangeable -- a coupled multi-process
  // likelihood may ship no way to reach a third derivative at all, while a
  // numerically failed finite difference is specific to one fit. The oracle
  // carries its own reason (Curvature3Oracle::declined) and it is reported
  // verbatim here rather than re-derived downstream.
  std::string declined;
  // Joint only: arms with no third-derivative oracle at all (0-based), so a
  // partially scored joint fit names which arms it left out.
  std::vector<int> arms_declined;
};

// Did any probed index come back with a finite gamma_3?
inline bool inner_skew_any_scored(const std::vector<double>& gamma3) {
  for (double g : gamma3) if (std::isfinite(g)) return true;
  return false;
}

// Third-derivative oracles for a joint fit, carried WITH the reason any arm has
// none (gcol33/tulpa#296). Built by build_joint_curvature3_fns
// (laplace_newton_joint.h); the reason travels with the oracles rather than
// being re-derived downstream, so a decline can never lose its explanation on
// the way to the fit object.
//
// `arms` holds one per-observation oracle per arm, for the arms whose
// contribution IS a separable per-observation sum. `cell_cubic`, when set,
// covers the arms a CellCouplingSpec took over: their per-obs sum is excluded
// from the joint log-lik (`skip_arm`), and the cell tensor contraction replaces
// it here (gcol33/tulpa#301). The two are disjoint by construction -- an arm is
// either summed per observation or routed through the cell branch -- so they add
// rather than double-count.
struct JointCurvature3Oracles {
  std::vector<Curvature3Oracle> arms;
  CellCubic3Fn        cell_cubic;
  std::vector<int>    arms_declined;   // 0-based arms with no oracle
  std::string         declined;        // fit-level reason when NOTHING has one
  bool any() const {
    for (const auto& o : arms) if (o.any()) return true;
    return static_cast<bool>(cell_cubic);
  }
};

// The probe scan every gamma_3 variant runs: walk the requested latent indices,
// solve H v_i = e_i against the LIVE factor, evaluate eta at mode + v_i, and hand
// the caller's accumulator the two eta buffers. Only the accumulation over the
// data log-likelihood differs between the single-arm, multi-process and joint
// variants, so everything else lives here once.
//
// `fill_eta0` / `fill_eta1` write eta at the current `x_buf` into the caller's own
// buffers (`fill_eta0` also precomputes anything that depends only on the mode).
// `accumulate(dropped, any_finite)` returns the un-normalised cubic sum for the
// current index; it must leave `any_finite` false when nothing finite reached it,
// so the index stays NaN rather than reading acc/sigma_i^3 == 0 ("perfectly
// Gaussian") -- the silently-wrong 0 gcol33/tulpa#272 fixed.
//
// x_buf must hold `mode` on entry and is restored to it on return. Reuses the
// live factor (chol, dense fallback; or sparse_solver when use_sparse) without
// refactorizing -- the same pattern the inv_block_layout diagonal-block
// extraction in laplace_newton.h uses.
template <typename FillEta0Fn, typename FillEta1Fn, typename AccumFn>
inline InnerSkewOutcome inner_skew_probe_scan(
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    Rcpp::NumericVector& x_buf,
    const std::vector<int>& probe_idx,
    FillEta0Fn fill_eta0,
    FillEta1Fn fill_eta1,
    AccumFn accumulate
) {
  InnerSkewOutcome out;
  out.gamma3.assign(probe_idx.size(), std::numeric_limits<double>::quiet_NaN());

  std::vector<double> rhs(n_x, 0.0), v(n_x, 0.0), z_work;
  if (!use_sparse) z_work.assign(n_x, 0.0);

  // eta at the mode -- x_buf already holds `mode` on entry.
  fill_eta0();

  for (std::size_t idx = 0; idx < probe_idx.size(); idx++) {
    int i = probe_idx[idx];
    if (i < 0 || i >= n_x) continue;

    double sigma_i = 0.0;
    if (!inner_probe_column(n_x, i, chol, sparse_solver, use_sparse,
                            rhs, v, z_work, sigma_i)) {
      continue;
    }
    const double sigma2_i = v[i];

    for (int k = 0; k < n_x; k++) x_buf[k] = mode[k] + v[k];
    fill_eta1();

    bool any_finite = false;
    const double acc = accumulate(out.n_nonfinite_dropped, any_finite);
    if (any_finite) out.gamma3[idx] = acc / (sigma_i * sigma2_i);  // sigma_i^-3
  }

  for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];  // restore
  if (!inner_skew_any_scored(out.gamma3)) out.declined = "no_finite_contribution";
  return out;
}

// Single-arm gamma_3. `oracle` is the third-derivative oracle for this fit's
// likelihood: `scalar` (l_j'''(eta_j) at the mode, or NaN where the likelihood
// has no registered third derivative) when the unit carries one eta, `unit` (the
// per-observation tensor contraction, curvature3_contract.h) when it carries
// `n_coords` of them laid out observation-major in eta as [i * n_coords + k] --
// the layout compute_eta_spec writes.
//
// compute_eta_fn(x, eta_out): the SAME closure convention laplace_newton_ll
// already uses (in-place write into eta_out, sized n_eta). x_buf / eta_buf0 /
// eta_buf1 are caller-supplied scratch (sized n_x, n_eta, n_eta).
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
    const Curvature3Oracle& oracle,
    const std::vector<int>& probe_idx
) {
  // No oracle at all -- every index is "not computable", not "zero skew".
  // Without this early return the per-index loop below would see every l3[j] as
  // NaN, drop every contribution, and divide 0/sigma_i^3 = 0 into gamma3.
  if (probe_idx.empty()) {
    InnerSkewOutcome out;
    out.gamma3.assign(probe_idx.size(), std::numeric_limits<double>::quiet_NaN());
    out.declined = "no_probe_indices";
    return out;
  }
  if (!oracle.any()) {
    InnerSkewOutcome out;
    out.gamma3.assign(probe_idx.size(), std::numeric_limits<double>::quiet_NaN());
    out.declined = oracle.declined.empty() ? std::string("no_oracle")
                                           : oracle.declined;
    return out;
  }

  std::vector<double> l3;
  const int nc = (oracle.n_coords > 0) ? oracle.n_coords : 1;
  const int n_units = (nc > 0) ? (n_eta / nc) : 0;
  std::vector<double> u_unit(oracle.unit ? nc : 0, 0.0);

  auto fill_eta0 = [&]() {
    compute_eta_fn(x_buf, eta_buf0);
    if (oracle.scalar) {
      l3.assign(n_eta, std::numeric_limits<double>::quiet_NaN());
      for (int j = 0; j < n_eta; j++) l3[j] = oracle.scalar(j, eta_buf0[j]);
    }
  };
  auto fill_eta1 = [&]() { compute_eta_fn(x_buf, eta_buf1); };

  auto accumulate = [&](int& dropped, bool& any_finite) -> double {
    double acc = 0.0;
    if (oracle.scalar) {
      for (int j = 0; j < n_eta; j++) {
        double u = eta_buf1[j] - eta_buf0[j];
        if (u == 0.0) continue;
        double l3j = l3[j];
        if (!std::isfinite(l3j)) { dropped++; continue; }
        acc += l3j * u * u * u;
        any_finite = true;
      }
      return acc;
    }
    const double* e0 = eta_buf0.begin();
    const double* e1 = eta_buf1.begin();
    for (int i = 0; i < n_units; i++) {
      const std::ptrdiff_t off = (std::ptrdiff_t)i * nc;
      bool moved = false;
      for (int k = 0; k < nc; k++) {
        u_unit[k] = e1[off + k] - e0[off + k];
        if (u_unit[k] != 0.0) moved = true;
      }
      if (!moved) continue;
      const double c = oracle.unit(i, e0 + off, u_unit.data());
      if (!std::isfinite(c)) { dropped++; continue; }
      acc += c;
      any_finite = true;
    }
    return acc;
  };

  return inner_skew_probe_scan(n_x, mode, chol, sparse_solver, use_sparse,
                               x_buf, probe_idx, fill_eta0, fill_eta1,
                               accumulate);
}

// Joint-arm generalization of compute_inner_skew_gamma3 above, for
// laplace_newton_joint.h / laplace_newton_joint_sparse.h's multi-arm Newton
// loops. A SEPARABLE joint fit (every arm's per-observation contributions
// summed) is the file header's own sum with j ranging over the union of
// (arm, observation) pairs instead of a single arm's rows, so the formula and
// its correctness proof carry over unchanged -- the same sum, not a new
// derivation.
//
// A genuinely COUPLED arm has its per-obs sum replaced by a CellCouplingSpec's
// evaluate_cell() term. Those arms are excluded from the per-observation sum
// (`skip_arm[k] == true` there, an empty `oracles.arms[k]` here, so their rows
// can never be scored against the wrong unused per-obs likelihood) and enter
// through `oracles.cell_cubic` instead: the cell tensor contraction of
// cell_curvature3.h, added once per probed index over all cells. The two terms
// partition the arms, so they add rather than double-count. Without a cell
// oracle a coupled arm simply contributes nothing, which is the pre-#301
// behaviour and still what a spec whose CellDerivs Hessian cannot be read gets.
//
// eta_buf0 / eta_buf1 are the per-arm scratch (NewtonScratchJoint's `etas` /
// `etas_tmp`), one Rcpp::NumericVector per arm sized to that arm's N.
// `oracles.arms` has one entry per arm (parallel to eta_buf0), and
// `oracles.declined` / `oracles.arms_declined` carry WHY an arm has none, so a
// partially or fully declined fit says which arms were left out and for what
// reason (gcol33/tulpa#296).
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
    const JointCurvature3Oracles& oracles,
    const std::vector<int>& probe_idx
) {
  const int n_arms = static_cast<int>(eta_buf0.size());
  if (probe_idx.empty() || !oracles.any()) {
    InnerSkewOutcome out;
    out.gamma3.assign(probe_idx.size(), std::numeric_limits<double>::quiet_NaN());
    out.arms_declined = oracles.arms_declined;
    if (probe_idx.empty()) {
      out.declined = "no_probe_indices";
    } else {
      out.declined = oracles.declined.empty() ? std::string("no_oracle")
                                              : oracles.declined;
    }
    return out;
  }

  std::vector<std::vector<double>> l3(n_arms);
  std::vector<std::vector<double>> u_unit(n_arms);

  auto fill_eta0 = [&]() {
    compute_eta_joint_fn(x_buf, eta_buf0);
    for (int k = 0; k < n_arms; k++) {
      const int Nk = eta_buf0[k].size();
      if (k >= static_cast<int>(oracles.arms.size())) continue;
      const Curvature3Oracle& o = oracles.arms[k];
      if (o.scalar) {
        l3[k].assign(Nk, std::numeric_limits<double>::quiet_NaN());
        for (int j = 0; j < Nk; j++) l3[k][j] = o.scalar(j, eta_buf0[k][j]);
      } else if (o.unit) {
        u_unit[k].assign(o.n_coords > 0 ? o.n_coords : 1, 0.0);
      }
    }
  };
  auto fill_eta1 = [&]() { compute_eta_joint_fn(x_buf, eta_buf1); };

  auto accumulate = [&](int& dropped, bool& any_finite) -> double {
    double acc = 0.0;
    for (int k = 0; k < n_arms; k++) {
      if (k >= static_cast<int>(oracles.arms.size())) continue;
      const Curvature3Oracle& o = oracles.arms[k];
      const int Nk = eta_buf0[k].size();
      if (o.scalar) {
        for (int j = 0; j < Nk; j++) {
          double u = eta_buf1[k][j] - eta_buf0[k][j];
          if (u == 0.0) continue;
          double l3kj = l3[k][j];
          if (!std::isfinite(l3kj)) { dropped++; continue; }
          acc += l3kj * u * u * u;
          any_finite = true;
        }
      } else if (o.unit) {
        const int nc = (o.n_coords > 0) ? o.n_coords : 1;
        const double* e0 = eta_buf0[k].begin();
        const double* e1 = eta_buf1[k].begin();
        for (int i = 0; i < Nk / nc; i++) {
          const std::ptrdiff_t off = (std::ptrdiff_t)i * nc;
          bool moved = false;
          for (int t = 0; t < nc; t++) {
            u_unit[k][t] = e1[off + t] - e0[off + t];
            if (u_unit[k][t] != 0.0) moved = true;
          }
          if (!moved) continue;
          const double c = o.unit(i, e0 + off, u_unit[k].data());
          if (!std::isfinite(c)) { dropped++; continue; }
          acc += c;
          any_finite = true;
        }
      }
    }
    if (oracles.cell_cubic) {
      const double c = oracles.cell_cubic(eta_buf0, eta_buf1);
      if (std::isfinite(c)) {
        acc += c;
        any_finite = true;
      } else {
        dropped++;
      }
    }
    return acc;
  };

  InnerSkewOutcome out =
      inner_skew_probe_scan(n_x, mode, chol, sparse_solver, use_sparse,
                            x_buf, probe_idx, fill_eta0, fill_eta1, accumulate);
  out.arms_declined = oracles.arms_declined;
  return out;
}

} // namespace tulpa

#endif // TULPA_INNER_LAPLACE_SKEW_H
