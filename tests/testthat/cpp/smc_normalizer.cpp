// The normalized log prior the SMC evidence adds back: compute_log_prior plus
// compute_log_prior_normalizer, at rows of a parameter matrix, for the
// built-in-family GLM with the latent blocks the sampler front door builds. A
// test compares it with the prior density written out independently in R. A
// theta with no rows returns the layout's parameter names alone.

#include <Rcpp.h>
#include <string>
#include <vector>

#include "laplace_spec_fit.h"      // as_offset_vec
#include "sampler_model_data.h"    // build_sampler_model_inputs / sampler_param_names

// [[Rcpp::export]]
Rcpp::List cpp_test_log_prior_normalized(
    Rcpp::NumericMatrix theta, Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials, Rcpp::NumericMatrix X, std::string family,
    double phi, double sigma_beta, double sigma_re_scale,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue) {
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(R_NilValue, y.size());
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, NA_REAL, sigma_beta, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec, tvc_spec);
    const int D = in.layout.total_params;
    if (theta.nrow() > 0 && theta.ncol() != D) {
        Rcpp::stop("theta has %d columns but the layout has %d parameters",
                   theta.ncol(), D);
    }
    std::vector<std::string> declined;
    const double norm =
        tulpa_hmc::compute_log_prior_normalizer(in.data, in.layout, &declined);
    Rcpp::NumericVector lp(theta.nrow());
    std::vector<double> th(D);
    for (int r = 0; r < theta.nrow(); r++) {
        for (int j = 0; j < D; j++) th[j] = theta(r, j);
        lp[r] = tulpa_hmc::compute_log_prior(th, in.data, in.layout) + norm;
    }
    return Rcpp::List::create(
        Rcpp::Named("log_prior") = lp,
        Rcpp::Named("normalizer") = norm,
        Rcpp::Named("declined") = Rcpp::wrap(declined),
        Rcpp::Named("names") = tulpa::sampler_param_names(in.data, in.layout,
                                                          R_NilValue));
}
