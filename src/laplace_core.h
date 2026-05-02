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
};

} // namespace tulpa

#endif // TULPA_LAPLACE_CORE_H
