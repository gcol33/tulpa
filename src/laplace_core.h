// laplace_core.h
// Core Laplace approximation engine for ratiod
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

// Result structure for Laplace mode finding
struct LaplaceResult {
  Rcpp::NumericVector mode;     // Mode of latent field x*(theta)
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
};

// Convert LaplaceResult to Rcpp::List. Single source of truth used by every
// laplace_core* R export.
inline Rcpp::List laplace_result_to_list(const LaplaceResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CORE_H
