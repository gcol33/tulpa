// Test probes over src/family_terms_export.cpp, compiled against the engine by load_cpp_fixture().

#include <RcppEigen.h>
#include <Rcpp.h>
#include "tulpa_stop.h"
#include "probe_lengths.h"
#include <string>
#include "laplace_family_link.h"
#include "laplace_family_curvature.h"
#include "laplace_family_zi_curvature.h"
#include "laplace_family_zi_phi.h"
#include "glmm_oracle.h"
#include "laplace_likelihoods.h"
#include "builtin_family_ll_ad.h"
#include "density_probe.h"

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

// d2(neg_hess)/d eta2 from laplace_family_curvature.h, with the gate that says
// whether it is exact for this family. The closed-form Laplace theta-Hessian
// differentiates the exact gradient once more; that second pass needs the eta
// second derivative of the same weight cpp_family_curvature_deta reports.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_curvature_deta2(double y, int n_trials, double eta,
                                               std::string family, double phi,
                                               double phi2 = NA_REAL) {
  return Rcpp::NumericVector::create(
      Rcpp::_["d2w_deta2"] = tulpa::curvature_deta2_for_family(y, n_trials, eta,
                                                               family, phi, phi2),
      Rcpp::_["exact"]     = tulpa::has_curvature_2nd_derivative(family) ? 1.0 : 0.0);
}

// The same second eta-derivative for the OBSERVED curvature W_obs = -l''(eta),
// which is the working weight plus the family's own delta. Where the delta's
// second derivative is unregistered the sum would be the working-curvature
// answer wearing the observed one's name, so the entry returns NaN; `exact`
// reports the same decision as a flag, making the gate readable from R rather
// than only inferable from a NaN.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_curvature_deta2(double y, int n_trials,
                                                   double eta,
                                                   std::string family,
                                                   double phi,
                                                   double phi2 = NA_REAL) {
  const bool ok = tulpa::has_curvature_2nd_derivative(family) &&
                  tulpa::has_obs_curvature_delta_2nd_derivative(family);
  return Rcpp::NumericVector::create(
      Rcpp::_["d2w_obs_deta2"] = tulpa::obs_curvature_deta2_for_family(
          y, n_trials, eta, family, phi, phi2),
      Rcpp::_["exact"]         = ok ? 1.0 : 0.0);
}

// Whether obs_curvature_deta2_for_family() is exact for this family.
// [[Rcpp::export]]
bool cpp_family_has_obs_curvature_2nd_derivative(std::string family) {
  return tulpa::has_curvature_2nd_derivative(family) &&
         tulpa::has_obs_curvature_delta_2nd_derivative(family);
}

// The supported set exactly as the kernel's refusal message reports it, so a
// test can hold the message to the gate rather than to a remembered list.
// [[Rcpp::export]]
std::string cpp_family_compiled_zi_supported_families() {
  return tulpa::zi::compiled_zi_supported_families();
}

// The mixture's 2 x 2 curvature block itself, one row per observation, columns
// [W_ee, W_ez, W_zz]. Exported so the derivative above can be checked against a
// finite difference of the very weight the kernel builds H from.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_curvature(Rcpp::NumericVector y,
                                             Rcpp::IntegerVector n_trials,
                                             Rcpp::NumericVector eta,
                                             Rcpp::NumericVector logit_zi,
                                             std::string family, double phi,
                                             double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  check_probe_lengths("cpp_zi_mixture_curvature", n, y, n_trials, &logit_zi);
  const bool recycle_nt = (n_trials.size() == 1);
  Rcpp::NumericMatrix out(n, 3);
  Rcpp::colnames(out) = Rcpp::CharacterVector::create("W_ee", "W_ez", "W_zz");
  double grad[2], nh[4];
  for (R_xlen_t i = 0; i < n; i++) {
    tulpa::zi::mixture_eta_weights_double(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], logit_zi[i],
        family, phi, phi2, grad, nh);
    out(i, 0) = nh[0]; out(i, 1) = nh[1]; out(i, 2) = nh[3];
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
  r.prepare();

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

// The link ladder itself, vectorized over eta. The mean and its first four
// eta-derivatives all reach a gradient, and each is a place a difference of
// two nearly-equal quantities can cancel: cloglog's mu is 1 - exp(-exp(eta)),
// which loses a digit per decade of |eta| in the lower tail and rounds to
// exactly 0 -- outside the support of every consumer of the link -- once
// exp(eta) falls below the double spacing at 1. Reading these against R's own
// link functions on RELATIVE tolerance is what sees that; an absolute one
// passes on a value that has no correct digits left.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_link_ladder(Rcpp::NumericVector eta, std::string link) {
  const int n = eta.size();
  Rcpp::NumericMatrix out(n, 5);
  Rcpp::colnames(out) = Rcpp::CharacterVector::create(
      "linkinv", "mu_eta", "mu_eta2", "mu_eta3", "mu_eta4");
  for (int i = 0; i < n; i++) {
    out(i, 0) = tulpa::linkinv(eta[i], link);
    out(i, 1) = tulpa::mu_eta(eta[i], link);
    out(i, 2) = tulpa::mu_eta2(eta[i], link);
    out(i, 3) = tulpa::mu_eta3(eta[i], link);
    out(i, 4) = tulpa::mu_eta4(eta[i], link);
  }
  return out;
}
