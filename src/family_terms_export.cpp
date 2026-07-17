// family_terms_export.cpp
// Per-observation (loglik, grad, curvature) probes for the family kernels that
// are maintained in parallel (A9).
//
// The same per-family math lives in several kernels because each serves a
// different backend: laplace_family_link.h (the Laplace/Newton dispatch),
// glmm_oracle.h (the compiled GLMM oracle behind agq_fit / re_aghq / the Gibbs
// sweep), and laplace_likelihoods.cpp (the explicit triplets). They are only
// consistent by convention -- and the conventions genuinely differ:
//
//   * `phi` is the residual SD in laplace_family_link.h and the residual
//     VARIANCE in glmm_oracle.h. The R layer bridges this at each boundary
//     (fit_laplace.R sqrt()s it; agq.R squares sigma_eps). Nothing in C++
//     enforces it, and both readings are finite and well-behaved -- exactly how
//     the 0.0.73 bug (laplace and mala silently fitting different models at
//     phi != 1) went unnoticed.
//   * glmm_oracle.h returns the signed d2/deta2; the Laplace side returns the
//     negated Hessian.
//
// These exports give each kernel a per-observation callable surface at a shared
// (y, eta, phi) so test-family-cross-path.R can pin the conventions against
// each other. They exist for testing only.

#include <Rcpp.h>
#include <string>

#include "laplace_family_link.h"
#include "glmm_oracle.h"
#include "laplace_likelihoods.h"

// Terms from the Laplace/Newton family dispatch (laplace_family_link.h).
// `phi` follows that kernel's convention: residual SD for gaussian/lognormal.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_terms(double y, int n_trials, double eta,
                                     std::string family, double phi,
                                     double phi2 = NA_REAL) {
  const tulpa::GradHess gh =
      tulpa::grad_hess_for_family(y, n_trials, eta, family, phi, phi2);
  return Rcpp::NumericVector::create(
      Rcpp::_["log_lik"]   = tulpa::log_lik_for_family(y, n_trials, eta, family,
                                                       phi, phi2),
      Rcpp::_["grad"]      = gh.grad,
      Rcpp::_["neg_hess"]  = gh.neg_hess);
}

// Terms from the compiled GLMM oracle (glmm_oracle.h). `phi` follows THAT
// kernel's convention: residual VARIANCE for gaussian. `d2` is returned as the
// negated Hessian so it is directly comparable with cpp_family_terms.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_glmm_elt_terms(std::string family, double eta, double y,
                                       double n_trials, double phi) {
  const tulpa::GLMMElt e =
      tulpa::glmm_elt(tulpa::glmm_family_from_string(family), eta, y, n_trials,
                      phi);
  return Rcpp::NumericVector::create(
      Rcpp::_["log_lik"]  = e.l,
      Rcpp::_["grad"]     = e.d1,
      Rcpp::_["neg_hess"] = -e.d2);
}

// Terms from the explicit gaussian triplet (laplace_likelihoods.cpp). `phi` is
// the residual SD. The other triplets already have probes
// (cpp_test_laplace_binomial / _poisson / _negbin); gaussian had none, which is
// why its phi convention was the one nobody could see.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_laplace_gaussian(double y, double eta, double phi) {
  return Rcpp::NumericVector::create(
      Rcpp::_["log_lik"]  = tulpa::log_lik_gaussian(y, eta, phi),
      Rcpp::_["grad"]     = tulpa::grad_log_lik_gaussian(y, eta, phi),
      Rcpp::_["neg_hess"] = tulpa::neg_hess_log_lik_gaussian(y, eta, phi));
}
