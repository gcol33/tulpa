// vi_convergence_probe.cpp
// Replays a given ELBO sequence, with the stochastic gradient at each
// iteration, through the VI stopping rule (ConvergenceChecker,
// src/vi_optimizer.h) and reports where it stops.
//
// The rule decides where every VI fit actually ends -- the iteration budget is
// rarely what binds. Its defining properties are statements about the rule,
// not about any one model, so they are tested on sequences rather than through
// a fit: it reads ELBO DIFFERENCES only, so shifting a whole run by a constant
// cannot move the stop (gcol33/tulpa#821), and it stops only where the
// gradient is stationary, so a plateau in a noisy ELBO does not end a run whose
// scales are still contracting (gcol33/tulpa#917).

#include <Rcpp.h>
#include <RcppEigen.h>
#include "vi_optimizer.h"

// [[Rcpp::export]]
Rcpp::List cpp_vi_convergence_replay(const Rcpp::NumericVector& elbo,
                                     const Rcpp::NumericMatrix& grad,
                                     double tol_grad = 1e-4,
                                     double tol_rel_elbo = 0.01,
                                     int patience = 50) {
  if (grad.nrow() != elbo.size()) {
    Rcpp::stop("grad needs one row per ELBO value.");
  }
  tulpa::vi::ConvergenceChecker checker(tol_grad, tol_rel_elbo, patience);
  Eigen::VectorXd g(grad.ncol());
  for (R_xlen_t i = 0; i < elbo.size(); ++i) {
    for (int j = 0; j < grad.ncol(); ++j) g(j) = grad(i, j);
    const std::string reason = checker.check(elbo[i], g);
    if (!reason.empty()) {
      return Rcpp::List::create(
          Rcpp::Named("iteration") = static_cast<int>(i) + 1,
          Rcpp::Named("reason")    = reason,
          Rcpp::Named("stopped")   = true);
    }
  }
  return Rcpp::List::create(
      Rcpp::Named("iteration") = static_cast<int>(elbo.size()),
      Rcpp::Named("reason")    = "max_iter",
      Rcpp::Named("stopped")   = false);
}
