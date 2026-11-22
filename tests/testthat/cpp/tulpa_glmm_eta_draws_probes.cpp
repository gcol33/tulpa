// Test probes over src/tulpa_glmm_eta_draws.cpp, compiled against the engine by load_cpp_fixture().

#include <RcppEigen.h>
#include <Rcpp.h>
#include "tulpa_stop.h"
#include "fixed_effect_prior.h"
#include <string>
#include <vector>
#include "hmc_sampler.h"           // ModelData / ParamLayout
#include "laplace_spec_fit.h"      // as_offset_vec
#include "log_post_impl.h"         // GenericLogPostState, generic_eta_at
#include "sampler_model_data.h"    // build_sampler_model_inputs
#include "sampler_log_prob.h"      // sampler_log_prob_rows
#include "hmc_gp.h"                // make_msgp_nc_view_{local,regional}
#include "nngp_nc_term_apply.h"    // apply_nngp_nc_term_double

// The log posterior at each draw row as the gradient pass reports it
// (`compute_gradient`'s value output, the target every gradient-driven kernel
// accepts and rejects on) next to the value-only `compute_log_post`, on the
// model built from the arguments the sampler received. The two are one target
// and have to agree at every position.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_sampler_log_post_paths(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5
) {
    const double sigma_beta_value = tulpa::read_sigma_beta(sigma_beta);
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset(N, 0.0);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, NA_REAL, sigma_beta_value, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, R_NilValue,
        R_NilValue, R_NilValue);
    const int D = in.layout.total_params;
    if (draws.ncol() != D) {
        tulpa::stop("cpp_test_sampler_log_post_paths: the draws carry %d "
                   "columns but the model lays out %d parameters.",
                   {draws.ncol(), D});
    }
    const int S = draws.nrow();
    Rcpp::NumericMatrix out(S, 2);
    std::vector<double> q(D), grad(D);
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) q[j] = draws(s, j);
        double lp_grad = 0.0;
        tulpa_hmc::compute_gradient(q, in.data, in.layout, grad, &lp_grad);
        out(s, 0) = lp_grad;
        out(s, 1) = tulpa_hmc::compute_log_post(q, in.data, in.layout);
    }
    return out;
}

// The per-draw log posterior of a ModelData sampler fit, from the same helper
// every non-NUTS backend reports its `log_prob` through, on the model built from
// the arguments the sampler received. Evaluated on a NUTS fit's draws it is
// what that fit's own recorded log_prob has to reproduce.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_tulpa_glmm_log_prob_draws(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi_spec = R_NilValue
) {
    const double sigma_beta_value =
        tulpa::read_sigma_beta(sigma_beta);
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta_value, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);
    return tulpa::sampler_log_prob_rows(draws, in.data, in.layout);
}
