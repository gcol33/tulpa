// vi_convergence_probe.cpp
// Replays a given ELBO sequence through the VI stopping rule
// (ConvergenceChecker, src/vi_optimizer.h) and reports where it stops.
//
// The rule decides where every VI fit actually ends -- the iteration budget is
// rarely what binds -- and its defining property is that it reads ELBO
// DIFFERENCES only, so shifting a whole run by a constant cannot move the stop
// (gcol33/tulpa#821). That property is a statement about the rule, not about
// any one model, so it is tested on a sequence rather than through a fit.

#include <Rcpp.h>
#include "vi_optimizer.h"

// [[Rcpp::export]]
Rcpp::List cpp_vi_convergence_replay(const Rcpp::NumericVector& elbo,
                                     double grad_norm = 1.0,
                                     double tol_grad = 1e-4,
                                     double tol_rel_elbo = 0.01,
                                     int patience = 50) {
  tulpa::vi::ConvergenceChecker checker(tol_grad, tol_rel_elbo, patience);
  for (R_xlen_t i = 0; i < elbo.size(); ++i) {
    const std::string reason = checker.check(elbo[i], grad_norm);
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
