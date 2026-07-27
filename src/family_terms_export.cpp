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
#include "laplace_family_zi_curvature.h"
#include "laplace_family_zi_phi.h"
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

// Whether curvature_deta2_for_family() is exact for this family, so the exact
// Laplace theta-Hessian can refuse rather than optimize a fiction.
// [[Rcpp::export]]
bool cpp_family_has_curvature_2nd_derivative(std::string family) {
  return tulpa::has_curvature_2nd_derivative(family);
}

// Vectorized observed-minus-working curvature W_obs - w at every observation.
// The exact mode Jacobian dx_hat/dtheta needs the TRUE posterior Hessian
// A' diag(W_obs) A + P, which is the working-weight H_joint plus
// A' diag(W_obs - w) A. Returning the difference lets the caller correct
// H_joint in place rather than rebuild the precision. It is identically zero for
// every family whose working weight already is the observed curvature, and for
// the families that lack an exact observed form obs_grad_hess_for_family
// delegates to the working weight, so this returns zero there too -- the caller
// must gate on has_exact_mode_jacobian, not on the difference being nonzero.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_curvature_delta_vec(Rcpp::NumericVector y,
                                                       Rcpp::IntegerVector n_trials,
                                                       Rcpp::NumericVector eta,
                                                       std::string family, double phi,
                                                       double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n) {
    Rcpp::stop("cpp_family_obs_curvature_delta_vec: y (%d) and eta (%d) differ in length.",
               (int)y.size(), (int)n);
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_family_obs_curvature_delta_vec: n_trials must be length 1 or %d (got %d).",
               (int)n, (int)n_trials.size());
  }
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; i++) {
    const int nt = recycle_nt ? n_trials[0] : n_trials[i];
    const double w_obs =
        tulpa::obs_grad_hess_for_family(y[i], nt, eta[i], family, phi, phi2).neg_hess;
    const double w =
        tulpa::grad_hess_for_family(y[i], nt, eta[i], family, phi, phi2).neg_hess;
    out[i] = w_obs - w;
  }
  return out;
}

// Vectorized d(W_obs - w)/deta, the eta-derivative of the vector above. The
// closed outer Hessian differentiates quantities formed on the TRUE-curvature
// inverse, which needs dH_true/dtheta and so this derivative; without it the
// assembly would differentiate through the working-weight inverse instead, an
// inconsistency that only shows where the two weights differ.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_curvature_delta_deta_vec(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericVector eta, std::string family, double phi,
    double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n) {
    Rcpp::stop("cpp_family_obs_curvature_delta_deta_vec: y (%d) and eta (%d) "
               "differ in length.", (int)y.size(), (int)n);
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_family_obs_curvature_delta_deta_vec: n_trials must be "
               "length 1 or %d (got %d).", (int)n, (int)n_trials.size());
  }
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; i++) {
    out[i] = tulpa::obs_curvature_delta_deta_for_family(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], family, phi, phi2);
  }
  return out;
}

// Whether the derivative above is exact for this family. False for
// neg_binomial_1, whose observed curvature would need a tetragamma.
// [[Rcpp::export]]
bool cpp_family_has_obs_curvature_delta_derivative(std::string family) {
  return tulpa::has_obs_curvature_delta_derivative(family);
}

// Whether the exact analytic mode Jacobian can be formed for this family, so the
// marginal correction can trust the closed J or fall back to differencing.
// [[Rcpp::export]]
bool cpp_family_has_exact_mode_jacobian(std::string family) {
  return tulpa::has_exact_mode_jacobian(family);
}

// Vectorized over observations, matching cpp_family_curvature_deta_vec: the
// closed-form theta-Hessian needs d2w/deta2 at every observation of a fit.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_curvature_deta2_vec(Rcpp::NumericVector y,
                                                   Rcpp::IntegerVector n_trials,
                                                   Rcpp::NumericVector eta,
                                                   std::string family, double phi,
                                                   double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n) {
    Rcpp::stop("cpp_family_curvature_deta2_vec: y (%d) and eta (%d) differ in length.",
               (int)y.size(), (int)n);
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_family_curvature_deta2_vec: n_trials must be length 1 or %d (got %d).",
               (int)n, (int)n_trials.size());
  }
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; i++) {
    out[i] = tulpa::curvature_deta2_for_family(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], family, phi, phi2);
  }
  return out;
}

// The six partials of the zero-inflation mixture's 2 x 2 curvature block, one
// row per observation, columns [dWee_deta, dWee_dz, dWez_deta, dWez_dz,
// dWzz_deta, dWzz_dz]. The exact outer gradient contracts them against the
// three linear-predictor (co)variances to form its dW/dtheta channel.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_curvature_deriv(Rcpp::NumericVector y,
                                                   Rcpp::IntegerVector n_trials,
                                                   Rcpp::NumericVector eta,
                                                   Rcpp::NumericVector logit_zi,
                                                   std::string family, double phi,
                                                   double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n || logit_zi.size() != n) {
    Rcpp::stop("cpp_zi_mixture_curvature_deriv: y (%d), eta (%d) and logit_zi "
               "(%d) must have the same length.",
               (int)y.size(), (int)n, (int)logit_zi.size());
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_zi_mixture_curvature_deriv: n_trials must be length 1 or %d "
               "(got %d).", (int)n, (int)n_trials.size());
  }
  Rcpp::NumericMatrix out(n, 6);
  Rcpp::colnames(out) = Rcpp::CharacterVector::create(
      "dWee_deta", "dWee_dz", "dWez_deta", "dWez_dz", "dWzz_deta", "dWzz_dz");
  for (R_xlen_t i = 0; i < n; i++) {
    const tulpa::zi::MixtureCurvatureDeriv d = tulpa::zi::mixture_curvature_deriv(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], logit_zi[i],
        family, phi, phi2);
    out(i, 0) = d.dWee_deta; out(i, 1) = d.dWee_dz;
    out(i, 2) = d.dWez_deta; out(i, 3) = d.dWez_dz;
    out(i, 4) = d.dWzz_deta; out(i, 5) = d.dWzz_dz;
  }
  return out;
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

// Whether the six partials above are exact for this family, so the outer
// optimization can take the analytic gradient or fall back cleanly.
// [[Rcpp::export]]
bool cpp_family_has_zi_curvature_derivative(std::string family) {
  return tulpa::zi::has_zi_curvature_derivative(family);
}

// The SECOND derivatives of the mixture's curvature block, one row per
// observation. Because the block is -Hess(log density), each is a fourth
// derivative of one scalar, so five columns cover all nine second partials --
// named for how many of the four derivatives are in eta. The closed outer
// Hessian contracts them against the same predictor (co)variances the gradient
// uses. Under a hurdle the three mixed columns are identically zero.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_curvature_deriv2(Rcpp::NumericVector y,
                                                    Rcpp::IntegerVector n_trials,
                                                    Rcpp::NumericVector eta,
                                                    Rcpp::NumericVector logit_zi,
                                                    std::string family, double phi,
                                                    double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n || logit_zi.size() != n) {
    Rcpp::stop("cpp_zi_mixture_curvature_deriv2: y (%d), eta (%d) and logit_zi "
               "(%d) must have the same length.",
               (int)y.size(), (int)n, (int)logit_zi.size());
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_zi_mixture_curvature_deriv2: n_trials must be length 1 or %d "
               "(got %d).", (int)n, (int)n_trials.size());
  }
  Rcpp::NumericMatrix out(n, 5);
  Rcpp::colnames(out) = Rcpp::CharacterVector::create(
      "d4_e4", "d4_e3z", "d4_e2z2", "d4_ez3", "d4_z4");
  for (R_xlen_t i = 0; i < n; i++) {
    const tulpa::zi::MixtureCurvatureDeriv2 d = tulpa::zi::mixture_curvature_deriv2(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], logit_zi[i],
        family, phi, phi2);
    out(i, 0) = d.d4_e4;   out(i, 1) = d.d4_e3z; out(i, 2) = d.d4_e2z2;
    out(i, 3) = d.d4_ez3;  out(i, 4) = d.d4_z4;
  }
  return out;
}

// Whether the closed-form outer Hessian's curvature input is exact under a
// mixture. Narrower than cpp_family_has_zi_curvature_derivative(): the gradient
// differentiates the 2 x 2 block once, the Hessian twice, and the second one
// additionally needs the base family's second eta-derivative for the coupled
// y = 0 branch.
// [[Rcpp::export]]
bool cpp_family_has_zi_curvature_2nd_derivative(std::string family) {
  return tulpa::zi::has_zi_curvature_2nd_derivative(family);
}

// Dispersion derivatives of the mixture's y = 0 branch under GENUINE zero
// inflation (see laplace_family_zi_phi.h). Zero at y != 0 and for a hurdle,
// where the base family's own phi registry is already the mixture's derivative;
// the R assembly adds the two over their disjoint rows.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_phi_deriv(Rcpp::NumericVector y,
                                             Rcpp::IntegerVector n_trials,
                                             Rcpp::NumericVector eta,
                                             Rcpp::NumericVector logit_zi,
                                             std::string family, double phi,
                                             double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n || logit_zi.size() != n) {
    Rcpp::stop("cpp_zi_mixture_phi_deriv: y (%d), eta (%d) and logit_zi "
               "(%d) must have the same length.",
               (int)y.size(), (int)n, (int)logit_zi.size());
  }
  const bool recycle_nt = (n_trials.size() == 1);
  if (!recycle_nt && n_trials.size() != n) {
    Rcpp::stop("cpp_zi_mixture_phi_deriv: n_trials must be length 1 or %d "
               "(got %d).", (int)n, (int)n_trials.size());
  }
  Rcpp::NumericMatrix out(n, 16);
  Rcpp::colnames(out) = Rcpp::CharacterVector::create(
      "dl_dp", "dsc_e_dp", "dsc_z_dp", "dWee_dp", "dWez_dp", "dWzz_dp",
      "dWee_dp_de", "dWee_dp_dz", "dWez_dp_dz", "dWzz_dp_dz",
      "dl_dp2", "dsc_e_dp2", "dsc_z_dp2", "dWee_dp2", "dWez_dp2", "dWzz_dp2");
  for (R_xlen_t i = 0; i < n; i++) {
    const tulpa::zi::MixturePhiDeriv d = tulpa::zi::mixture_phi_deriv(
        y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i], logit_zi[i],
        family, phi, phi2);
    out(i, 0)  = d.dl_dp;      out(i, 1)  = d.dsc_e_dp;
    out(i, 2)  = d.dsc_z_dp;   out(i, 3)  = d.dWee_dp;
    out(i, 4)  = d.dWez_dp;    out(i, 5)  = d.dWzz_dp;
    out(i, 6)  = d.dWee_dp_de; out(i, 7)  = d.dWee_dp_dz;
    out(i, 8)  = d.dWez_dp_dz; out(i, 9)  = d.dWzz_dp_dz;
    out(i, 10) = d.dl_dp2;     out(i, 11) = d.dsc_e_dp2;
    out(i, 12) = d.dsc_z_dp2;  out(i, 13) = d.dWee_dp2;
    out(i, 14) = d.dWez_dp2;   out(i, 15) = d.dWzz_dp2;
  }
  return out;
}

// Whether the dispersion coordinate is exact alongside a zero-inflation
// process for this family.
// [[Rcpp::export]]
bool cpp_family_has_zi_phi_deriv(std::string family) {
  return tulpa::zi::has_zi_phi_deriv(family);
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
