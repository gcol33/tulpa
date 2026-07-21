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
#include "laplace_family_curvature.h"
#include "glmm_oracle.h"
#include "laplace_likelihoods.h"
#include "builtin_family_ll_ad.h"

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

// Score plus OBSERVED curvature from the same dispatch. This is what the
// zero-inflation mixture differentiates through at y = 0, and it differs from
// cpp_family_terms wherever the Newton working weight is an expected form, so
// it needs its own probe to pin against .family_obs_weight() in R.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_terms(double y, int n_trials, double eta,
                                         std::string family, double phi,
                                         double phi2 = NA_REAL) {
  const tulpa::GradHess gh =
      tulpa::obs_grad_hess_for_family(y, n_trials, eta, family, phi, phi2);
  return Rcpp::NumericVector::create(
      Rcpp::_["grad"]     = gh.grad,
      Rcpp::_["neg_hess"] = gh.neg_hess);
}

// d(neg_hess)/d eta from laplace_family_curvature.h, with the gate that says
// whether it is exact for this family. The exact Laplace gradient differentiates
// log|H|, and H carries the weight cpp_family_terms reports as `neg_hess`, so
// this is the derivative of THAT weight -- for the expected-form families it is
// deliberately not the third derivative of the log density.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_curvature_deta(double y, int n_trials, double eta,
                                              std::string family, double phi,
                                              double phi2 = NA_REAL) {
  return Rcpp::NumericVector::create(
      Rcpp::_["dw_deta"] = tulpa::curvature_deta_for_family(y, n_trials, eta,
                                                            family, phi, phi2),
      Rcpp::_["exact"]   = tulpa::has_curvature_derivative(family) ? 1.0 : 0.0);
}

// Whether curvature_deta_for_family() is exact for this family, so the exact
// Laplace gradient can refuse rather than optimize a fiction.
// [[Rcpp::export]]
bool cpp_family_has_curvature_derivative(std::string family) {
  return tulpa::has_curvature_derivative(family);
}

// Vectorized over observations. The exact Laplace gradient needs dw/deta at
// every observation of a fit, so the per-element probe above would cost one
// .Call per row; this keeps it to one. `n_trials` is recycled when length 1.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_curvature_deta_vec(Rcpp::NumericVector y,
                                                  Rcpp::IntegerVector n_trials,
                                                  Rcpp::NumericVector eta,
                                                  std::string family, double phi,
                                                  double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n) {
    Rcpp::stop("cpp_family_curvature_deta_vec: y (%d) and eta (%d) differ in length.",
               (int)y.size(), (int)n);
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_family_curvature_deta_vec: n_trials must be length 1 or %d (got %d).",
               (int)n, (int)n_trials.size());
  }
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; i++) {
    out[i] = tulpa::curvature_deta_for_family(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], family, phi, phi2);
  }
  return out;
}

// The AD-templated density (builtin_family_ll_ad.h), which is what the sampler
// backends differentiate, evaluated at fwd::Dual so both its value and its
// derivative come back. The double path's value comes from a separate
// implementation (log_lik_for_family), so agreement here is a genuine
// cross-check of two independent expressions of the same density rather than a
// tautology -- and the derivative pins the AD plumbing each branch relies on.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_ad_terms(double y, int n_trials, double eta,
                                        std::string family, double phi,
                                        double phi2 = NA_REAL) {
  tulpa::BuiltinFamilyResponse r;
  r.y        = &y;
  r.n_trials = &n_trials;
  r.N        = 1;
  r.family   = family;
  r.phi      = phi;
  r.phi2     = phi2;

  const fwd::Dual e(eta, 1.0);
  const fwd::Dual ll = tulpa::builtin_family_base_ll_ad<fwd::Dual>(
      y, n_trials, &r, e);
  return Rcpp::NumericVector::create(
      Rcpp::_["log_lik"] = ll.val,
      Rcpp::_["grad"]    = ll.grad);
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
