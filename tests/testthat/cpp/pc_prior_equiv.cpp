// pc_prior_equiv.cpp
// Probes for the d = 2 PC range prior and the bounded-parameter map

//
// Every path that needs the range density routes through pc_prior.h: the SPDE
// hyper-prior in tulpa_priors_spde.h, the GP and SVC sampler paths, and the R
// nested path (pc_prior_log_density in fit_spde_nested.R). The rate is
// -log(alpha) * U. The sampler paths place the prior on phi through the
// bounded map rather than behind a hard -INFINITY wall, so NUTS keeps a
// gradient everywhere it can step.
//
// These exports let test-pc-prior.R check the shared helper against the R
// implementation, which is the independently written twin, and check the
// bounded map's Jacobian against a numerical derivative of the map it is
// supposed to differentiate.

#include <Rcpp.h>

#include "pc_prior.h"
#include "autodiff_utils.h"
#include "hmc_sampler.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_log_prior_log_sigma2_pc(
    Rcpp::NumericVector log_sigma2, double U, double alpha) {
  Rcpp::NumericVector out(log_sigma2.size());
  for (R_xlen_t i = 0; i < log_sigma2.size(); i++) {
    out[i] = tulpa::log_prior_log_sigma2_pc<double>(log_sigma2[i], U, alpha);
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
Rcpp::NumericVector cpp_test_log_jacobian_bounded(Rcpp::NumericVector u,
                                                  double lower, double upper) {
  Rcpp::NumericVector out(u.size());
  for (R_xlen_t i = 0; i < u.size(); i++) {
    out[i] = tulpa::math::log_jacobian_bounded<double>(u[i], lower, upper);
  }
  return out;
}

// The gate compute_param_layout applies to every NNGP block carrying a sampled
// range, exposed so the refusal itself is covered rather than inferred.
// [[Rcpp::export]]
void cpp_test_gp_layout_anchor_check(double U, double alpha) {
  tulpa_hmc::require_range_prior_anchors(U, alpha, "gp");
}
