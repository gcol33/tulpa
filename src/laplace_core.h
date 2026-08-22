// laplace_core.h
// Core Laplace approximation engine for tulpa
// Implements nested Laplace approximation for latent Gaussian models

#ifndef TULPA_LAPLACE_CORE_H
#define TULPA_LAPLACE_CORE_H

#include <Rcpp.h>
#include <limits>
#include <string>
#include <vector>
#include "laplace_likelihoods.h"

namespace tulpa {

// ---------------------------------------------------------------------
// Laplace approximation core
// ---------------------------------------------------------------------

// Result structure for Laplace mode finding.
//
// `mode` is `std::vector<double>` rather than `Rcpp::NumericVector` so the
// solver can populate it inside an OpenMP parallel region — Rf_allocVector
// is not thread-safe (run_nested_laplace_grid relies on this).
struct LaplaceResult {
  std::vector<double> mode;     // Mode of latent field x*(theta)
  double log_det_Q;             // Log determinant of posterior precision
  double log_marginal;          // Log p(y | theta) approximation
  int n_iter;                   // Newton iterations used
  bool converged;               // Convergence flag

  // Achieved residual: max_j |d(log p(y|x,theta) + log p(x|theta))/dx_j| at the
  // reported mode. The solve's convergence flag says the STOPPING RULE was met;
  // this says how stationary the point it stopped at actually is, which is a
  // different question and the one every downstream quantity that differentiates
  // through the mode depends on. The Laplace log-marginal only feels a mode
  // error quadratically, but its theta-gradient feels it linearly (log|H| is not
  // stationary in x), so a residual that is negligible for the fit itself can
  // still cost the outer gradient several digits -- see .laplace_exact_core(),
  // which declines rather than differentiate through a mode that did not settle.
  // Every driver re-scatters grad at the returned mode for the log-determinant,
  // so this is read off that pass at no extra cost.
  double score_max = 0.0;

  // Whether the Hessian at the returned point is the positive-definite matrix
  // the Laplace expansion needs. False where the final factorization had to
  // condition it -- a diagonal load or an eigenvalue clamp -- or could not
  // factor it at all: the reported log-determinant is then of a DIFFERENT matrix
  // than the scatter built, and the inverse of that matrix is not a posterior
  // covariance, which is why the stored precision and the marginal covariance
  // blocks are withheld there. True on every solve whose first factorization
  // succeeded, which is every solve at a PD Hessian.
  bool hessian_pd_at_mode = true;

  // The final factorization had to condition the Hessian to succeed -- a
  // cumulative diagonal load (the LM escalation ladder) or an eigenvalue clamp.
  // Distinct from hessian_pd_at_mode, which on the sum-to-zero path reads the
  // DIRECT factor of the pinned matrix instead: escalation is the normal case
  // there, because the rank-1 pins are deliberately left off the stored H, so a
  // conditioned solve on that path is not a solve at a non-PD point. What this
  // records is the mechanical fact, so a caller can tell an export taken from a
  // conditioned factorization apart from one that never needed conditioning.
  // The exported precision and the fixed-effect block are snapshots taken
  // BEFORE the escalation either way.
  bool pd_conditioned = false;

  // The sum-to-zero log-determinant fell back. On that path the reported
  // -0.5 log|B| is read from a direct factor of B = H + sum_k coef_k 1_k 1_k',
  // the pinned matrix; where that factor cannot be formed both readers keep the
  // PD-enforced log-determinant instead, which is a real determinant of
  // H + lambda I after the LM escalation ladder and not of B. The escalation is
  // the normal case on this path rather than the exception, so the two are not
  // close, and the value weights the outer hyperparameter grid: a cell that fell
  // back is reweighted against its neighbours. True where that happened, so the
  // count is reportable instead of the substitution being silent.
  bool s2z_log_det_fallback = false;

  // The solve never started: the penalized objective was non-finite at the
  // supplied latent start and the feasibility sweep (make_start_feasible) found
  // no interior point. This is distinct from converged = false, which means the
  // solve ran and did not reach the tolerance. It is reported as a flag rather
  // than thrown because the solver runs inside OpenMP parallel regions, where an
  // Rcpp::stop escaping the structured block is std::terminate; the R side turns
  // it into an error. Reachable only for a likelihood whose domain excludes the
  // start, i.e. the eta > 0 links.
  bool start_infeasible = false;

  // Q at the mode in CSC lower-triangle. Populated only when the Newton
  // solver is called with store_Q = true (default false). Q_csc_n == 0
  // means "not stored". When stored, Q_csc_p has length Q_csc_n + 1 and
  // Q_csc_i / Q_csc_x both have length Q_csc_p[Q_csc_n] (the nnz of the
  // lower triangle including the diagonal). Format matches the dgCMatrix
  // / cholmod_sparse stype = -1 convention.
  std::vector<int>    Q_csc_p;
  std::vector<int>    Q_csc_i;
  std::vector<double> Q_csc_x;
  int                 Q_csc_n = 0;

  // Marginal posterior-covariance blocks of the latent field: each block is a
  // diagonal block of H^{-1} (the FULL inverse, so cross-coupling with the
  // fixed effects and other blocks is marginalized out). Populated only when
  // the solver is called with a non-empty inv-block layout (default empty).
  // `re_cov_block_sizes[b]` is the side length m_b of block b; `re_cov_flat`
  // is the column-major concatenation of the blocks (block b occupies
  // m_b * m_b entries). Empty `re_cov_block_sizes` means "not computed".
  // Used by the EM M-step for a full random-effect covariance, which needs
  // Cov(u_g | y, Sigma) per group.
  std::vector<double> re_cov_flat;
  std::vector<int>    re_cov_block_sizes;

  // Inner-Laplace skewness diagnostic (Rue, Martino & Chopin 2009, JRSSB
  // 71(2) Sec 3.2.3, eq. 21's cubic term, generalized to tulpa's eta =
  // compute_eta(x) representation -- see src/inner_laplace_skew.h for the
  // derivation). `inner_skew[k]` is gamma_3 for latent index
  // `inner_skew_idx[k]`, i.e. the leading-order skewness estimate of the true
  // conditional posterior pi(x_i | theta, y) under the Gaussian (inner
  // Laplace) approximation; NaN means "not computable" (an unidentified
  // component, or a likelihood contribution without a registered
  // third-derivative). Populated only when the solver is called with
  // compute_skew = true (default false). `inner_skew_dropped` counts
  // (i, observation) contributions skipped for a non-finite third
  // derivative -- a nonzero count does not invalidate the other entries, but
  // is surfaced so the caller can say how many were skipped.
  // WHY nothing was computable, when nothing was. NaN alone
  // says "not computable" without separating a structural impossibility (a
  // coupled multi-process likelihood has no single per-observation term this
  // formula scores, so it can NEVER be scored) from a transient one (a
  // finite-difference that failed numerically) -- and a fit reporting the
  // former was read as the caller having switched the diagnostic off. Empty
  // when at least one index scored. Vocabulary: "no_probe_indices",
  // "coupled_arm" (a joint fit whose scorable arms all declined),
  // "curvature3_unavailable" (no registered third derivative / no
  // eta_weights_fn), "no_finite_contribution" (an oracle existed but nothing
  // finite reached any probed index), plus the four reasons a driver settles
  // before either diagnostic runs, which decline the importance curve too
  // (inner_probe_decline, inner_laplace_skew.h): "not_converged" (the solve
  // stopped short of a mode, so there is no point to expand about and no
  // Gaussian at a mode to score), and on the sparse joint path
  // "pd_eigen_clamp", "s2z_rank1_factor" and "factor_unavailable", each naming
  // a way the live CHOLMOD factor holds a different matrix than the one the
  // solve stepped with.
  // `inner_skew_arms_declined` lists the joint arms with no oracle at all
  // (0-based here, 1-based on the R side), so a PARTIALLY scored joint fit
  // names which arms were left out.
  // The companion LOCATION term: `inner_skew_gamma1[k]` is
  // Rue, Martino & Chopin's gamma^(1) at the same latent index, the first-order
  // coefficient of eq. (12)'s DENOMINATOR along the same conditional-mean curve.
  // Same layout, same NaN-means-not-computable rule. Empty when the pass did not
  // run at all, with `inner_skew_gamma1_declined` carrying the reason
  // ("eta_var_budget" for a field past INNER_ETA_VAR_MAX_SOLVES,
  // "eta_var_solve_failed", "multi_eta_unit" for a widened or coupled unit).
  std::vector<double> inner_skew;
  std::vector<double> inner_skew_gamma1;
  std::string         inner_skew_gamma1_declined;
  std::vector<int>    inner_skew_idx;
  int                 inner_skew_dropped = 0;
  std::string         inner_skew_declined;
  std::vector<int>    inner_skew_arms_declined;

  // Inner-Laplace importance k-hat raw material (see
  // src/inner_laplace_is.h). Populated alongside the skewness diagnostic, from
  // the same probe indices and the same conditional-curve solve, but needing no
  // likelihood derivative at all -- so it is available where the cubic term
  // declines. `inner_is_z` are the standardized proposal draws (one shared set),
  // `inner_is_log_joint` the joint log density along each probed index's
  // conditional-mean curve at those draws, laid out column-major as
  // [n_draws x n_probe] against `inner_skew_idx`, and `inner_is_sigma` the
  // per-index conditional SD (NaN where the probe column could not be solved).
  // The Pareto fit that turns these into a k-hat is the shared R primitive.
  std::vector<double> inner_is_z;
  std::vector<double> inner_is_log_joint;
  std::vector<double> inner_is_sigma;
  std::string         inner_is_declined;

  // Subspace debias (see src/subspace_debias.h). Populated
  // only when the solver is called with a non-empty index set: the Metropolis
  // draws of x_S - mode_S along the Gaussian-conditional-mean surface through
  // the mode, laid out column-major as [n_kept x q] against `debias_idx`
  // (0-based here, 1-based on the R side), together with the inner Laplace's
  // own marginal covariance of x_S (`debias_sigma_ss`, q x q column-major) so
  // the caller can rebuild the Gaussian conditional for the coordinates it did
  // NOT sample. `debias_declined` says why nothing was sampled, when nothing
  // was; an empty index set never reaches the sampler at all, so the fit is
  // then the plain Laplace fit with no random number consumed. Vocabulary:
  // "no_probe_indices", "degenerate_proposal", "objective_not_finite", plus the
  // gate reasons the driver settles first -- "not_converged" and, on the sparse
  // joint path, "pd_eigen_clamp" / "s2z_rank1_factor" / "factor_unavailable".
  // A gated decline echoes `debias_idx` back with `debias_n_kept = 0`, so a
  // requested correction that produced nothing is not mistaken for one that was
  // never requested.
  std::vector<int>    debias_idx;
  std::vector<double> debias_draws;
  std::vector<double> debias_sigma_ss;
  int                 debias_n_kept = 0;
  double              debias_accept = 0.0;
  double              debias_scale = 0.0;
  std::string         debias_declined;

  // Corrected integrated Laplace (see src/inner_cila.h).
  // Populated only when the solver is called with an active request: the M
  // unnormalized log importance ratios of draws from this cell's inner
  // Gaussian, the leading latent prefix of each draw (`cila_fixed`,
  // column-major [M x cila_n_fixed], presented under whatever fold the loop
  // presents its mode under), and the corrected cell log-marginal
  // logmeanexp(cila_log_w). `cila_requested` separates "the correction was
  // asked for and produced nothing" from "it was never asked for", so
  // `cila_declined` is read as a reason rather than as silence.
  bool                cila_requested = false;
  std::vector<double> cila_log_w;
  std::vector<double> cila_fixed;
  double              cila_log_marginal =
                          std::numeric_limits<double>::quiet_NaN();
  int                 cila_n_points = 0;
  int                 cila_n_fixed = 0;
  int                 cila_variant = 0;
  std::string         cila_declined;
  std::string         cila_fallback;
};

// Emit the subspace-debias fields onto a result list, when the solver ran the
// correction. One attach point, mirroring attach_inner_is_fields below.
inline void attach_debias_fields(Rcpp::List& out, const LaplaceResult& res) {
  if (res.debias_idx.empty()) return;
  const int q = static_cast<int>(res.debias_idx.size());
  Rcpp::IntegerVector idx_r(q);
  for (int b = 0; b < q; b++) idx_r[b] = res.debias_idx[b] + 1;
  out["debias_idx"] = idx_r;
  out["debias_accept"] = res.debias_accept;
  out["debias_scale"] = res.debias_scale;
  out["debias_declined"] = res.debias_declined;
  const int S = res.debias_n_kept;
  if (S > 0 && res.debias_draws.size() ==
                   static_cast<std::size_t>(S) * static_cast<std::size_t>(q)) {
    Rcpp::NumericMatrix dr(S, q);
    for (std::size_t e = 0; e < res.debias_draws.size(); e++) dr[e] = res.debias_draws[e];
    out["debias_draws"] = dr;
  }
  if (res.debias_sigma_ss.size() ==
          static_cast<std::size_t>(q) * static_cast<std::size_t>(q)) {
    Rcpp::NumericMatrix ss(q, q);
    for (int e = 0; e < q * q; e++) ss[e] = res.debias_sigma_ss[e];
    out["debias_sigma_ss"] = ss;
  }
}

// Emit the corrected-integrated-Laplace fields onto a result list, when the
// correction was requested. Written even when it declined, so the reason
// reaches the fit instead of the fields simply being absent.
inline void attach_cila_fields(Rcpp::List& out, const LaplaceResult& res) {
  if (!res.cila_requested) return;
  out["cila_log_marginal"] = res.cila_log_marginal;
  out["cila_n_points"]     = res.cila_n_points;
  out["cila_variant"]      = res.cila_variant;
  out["cila_declined"]     = res.cila_declined;
  out["cila_fallback"]     = res.cila_fallback;
  if (!res.cila_log_w.empty()) out["cila_log_w"] = res.cila_log_w;
  const int M = res.cila_n_points;
  const int p = res.cila_n_fixed;
  if (M > 0 && p > 0 &&
      res.cila_fixed.size() == static_cast<std::size_t>(M) *
                               static_cast<std::size_t>(p)) {
    Rcpp::NumericMatrix fx(M, p);
    for (std::size_t e = 0; e < res.cila_fixed.size(); e++) fx[e] = res.cila_fixed[e];
    out["cila_fixed"] = fx;
  }
}

// Emit the inner-Laplace importance-curve fields onto a result list, when the
// solver computed them. One attach point for the single-fit contract
// (laplace_result_to_list) and the grid driver's per-cell emitters
// (nested_laplace_grid.h), so the R side reads one set of names everywhere.
inline void attach_inner_is_fields(Rcpp::List& out, const LaplaceResult& res) {
  if (res.inner_is_sigma.empty()) return;
  const int P = static_cast<int>(res.inner_is_sigma.size());
  const int S = static_cast<int>(res.inner_is_z.size());
  out["inner_is_z"]     = res.inner_is_z;
  out["inner_is_sigma"] = res.inner_is_sigma;
  out["inner_is_declined"] = res.inner_is_declined;
  if (S > 0 && static_cast<int>(res.inner_is_log_joint.size()) == S * P) {
    Rcpp::NumericMatrix lj(S, P);
    for (int e = 0; e < S * P; e++) lj[e] = res.inner_is_log_joint[e];
    out["inner_is_log_joint"] = lj;
  }
}

// Convert LaplaceResult to Rcpp::List. Single source of truth used by every
// laplace_core* R export. Rcpp wraps std::vector<double> implicitly into a
// fresh NumericVector at the boundary.
inline Rcpp::List laplace_result_to_list(const LaplaceResult& result) {
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged,
    Rcpp::Named("score_max") = result.score_max,
    Rcpp::Named("start_infeasible") = result.start_infeasible,
    Rcpp::Named("hessian_pd_at_mode") = result.hessian_pd_at_mode,
    Rcpp::Named("pd_conditioned") = result.pd_conditioned
  );

  // Marginal posterior-covariance blocks, when the solver was asked to extract
  // them. Emitted as an R list of m_b x m_b numeric matrices (one per block),
  // in the order the layout requested. Absent otherwise so every other caller
  // is unaffected.
  if (!result.re_cov_block_sizes.empty()) {
    int n_blocks = static_cast<int>(result.re_cov_block_sizes.size());
    Rcpp::List cov_blocks(n_blocks);
    std::size_t off = 0;
    for (int b = 0; b < n_blocks; b++) {
      int m = result.re_cov_block_sizes[b];
      Rcpp::NumericMatrix M(m, m);
      // re_cov_flat is column-major per block, matching NumericMatrix storage.
      for (int e = 0; e < m * m; e++) M[e] = result.re_cov_flat[off + e];
      off += static_cast<std::size_t>(m) * m;
      cov_blocks[b] = M;
    }
    out["cov_blocks"] = cov_blocks;
  }

  // The posterior precision at the mode, when the solver was asked to keep it.
  // Lower triangle in CSC (stype = -1), so the R side symmetrizes. Emitted as
  // the raw triplet of vectors rather than a dgCMatrix because this header has
  // no Matrix dependency; .laplace_joint_hessian() assembles it.
  if (result.Q_csc_n > 0) {
    out["H_joint_p"] = result.Q_csc_p;
    out["H_joint_i"] = result.Q_csc_i;
    out["H_joint_x"] = result.Q_csc_x;
    out["H_joint_n"] = result.Q_csc_n;
  }

  // Inner-Laplace skewness diagnostic, when the solver was asked to compute
  // it (compute_skew = true). `inner_skew_idx` is 1-based on the R side.
  if (!result.inner_skew_idx.empty()) {
    Rcpp::IntegerVector idx_r(result.inner_skew_idx.size());
    for (std::size_t k = 0; k < result.inner_skew_idx.size(); k++) {
      idx_r[k] = result.inner_skew_idx[k] + 1;
    }
    out["inner_skew"] = result.inner_skew;
    out["inner_skew_gamma1"] = result.inner_skew_gamma1;
    out["inner_skew_gamma1_declined"] = result.inner_skew_gamma1_declined;
    out["inner_skew_idx"] = idx_r;
    out["inner_skew_dropped"] = result.inner_skew_dropped;
    out["inner_skew_declined"] = result.inner_skew_declined;
    if (!result.inner_skew_arms_declined.empty()) {
      Rcpp::IntegerVector arms_r(result.inner_skew_arms_declined.size());
      for (std::size_t k = 0; k < result.inner_skew_arms_declined.size(); k++) {
        arms_r[k] = result.inner_skew_arms_declined[k] + 1;
      }
      out["inner_skew_arms_declined"] = arms_r;
    }
  }
  attach_inner_is_fields(out, result);
  attach_debias_fields(out, result);
  attach_cila_fields(out, result);

  return out;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CORE_H
