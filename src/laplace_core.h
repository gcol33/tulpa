// laplace_core.h
// Core Laplace approximation engine for tulpa
// Implements nested Laplace approximation for latent Gaussian models

#ifndef TULPA_LAPLACE_CORE_H
#define TULPA_LAPLACE_CORE_H

#include <Rcpp.h>
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
};

// Convert LaplaceResult to Rcpp::List. Single source of truth used by every
// laplace_core* R export. Rcpp wraps std::vector<double> implicitly into a
// fresh NumericVector at the boundary.
inline Rcpp::List laplace_result_to_list(const LaplaceResult& result) {
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
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

  return out;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CORE_H
