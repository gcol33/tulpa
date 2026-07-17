// pc_prior_equiv_export.cpp
// Probes for the d = 2 PC range prior and the bounded-parameter map

//
// The range density existed three times: the SPDE hyper-prior in
// tulpa_priors_spde.h, the R nested path (pc_prior_log_density in
// fit_spde_nested.R), and an unwired log_prior_phi_pc in hmc_gp_log_lik.h that
// set rate = -log(alpha)/U where its own derivation gives -log(alpha)*U. The GP
// and SVC paths used none of them: both placed a Uniform on phi behind a hard
// -INFINITY wall, which gives NUTS no gradient to recover from and railed phi
// to the prior mean. Every path now routes through pc_prior.h.
//
// These exports let test-pc-prior.R check the shared helper against the R
// implementation, which is the independently written twin, and check the
// bounded map's Jacobian against a numerical derivative of the map it is
// supposed to differentiate.

#include <Rcpp.h>

#include "pc_prior.h"
#include "autodiff_utils.h"
#include "hmc_sampler.h"

// The half-Cauchy scale prior on the coordinate the SVC block samples it on.
// Exported for the same reason as the range density: the NNGP SVC path is
// reachable only from a consumer package's fit, so the density is asserted
// through the shared helper the sampler itself calls.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_log_prior_log_sigma2_half_cauchy(
    Rcpp::NumericVector log_sigma2, double scale) {
  Rcpp::NumericVector out(log_sigma2.size());
  for (R_xlen_t i = 0; i < log_sigma2.size(); i++) {
    out[i] = tulpa::log_prior_log_sigma2_half_cauchy<double>(log_sigma2[i],
                                                             scale);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_log_prior_range_pc(Rcpp::NumericVector range,
                                                double U, double alpha) {
  Rcpp::NumericVector out(range.size());
  for (R_xlen_t i = 0; i < range.size(); i++) {
    const double r = range[i];
    out[i] = tulpa::log_prior_range_pc<double>(r, U, alpha);
  }
  return out;
}

// Same density, entered from log(range) -- the form the sampled log-scale
// coordinates use.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_log_prior_range_pc_at_log(
    Rcpp::NumericVector log_range, double U, double alpha) {
  Rcpp::NumericVector out(log_range.size());
  for (R_xlen_t i = 0; i < log_range.size(); i++) {
    const double lr = log_range[i];
    out[i] = tulpa::log_prior_range_pc_at_log<double>(lr, U, alpha);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_bounded_from_logit(Rcpp::NumericVector u,
                                                double lower, double upper) {
  Rcpp::NumericVector out(u.size());
  for (R_xlen_t i = 0; i < u.size(); i++) {
    const double ui = u[i];
    out[i] = tulpa::math::bounded_from_logit<double>(ui, lower, upper);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_log_jacobian_bounded(Rcpp::NumericVector phi,
                                                  double lower, double upper) {
  Rcpp::NumericVector out(phi.size());
  for (R_xlen_t i = 0; i < phi.size(); i++) {
    const double p = phi[i];
    out[i] = tulpa::math::log_jacobian_bounded<double>(p, lower, upper);
  }
  return out;
}

// The gate compute_param_layout applies to every NNGP block carrying a sampled
// range, exposed so the refusal itself is covered rather than inferred.
// [[Rcpp::export]]
void cpp_test_gp_layout_anchor_check(double U, double alpha) {
  tulpa_hmc::require_range_prior_anchors(U, alpha, "gp");
}
