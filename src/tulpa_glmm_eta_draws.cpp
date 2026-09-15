// tulpa_glmm_eta_draws.cpp
// ----------------------------------------------------------------------------
// The in-sample linear predictor at each posterior draw of a ModelData sampler
// fit, read through the same assembly the sampler's observation loop reads.
//
// A draw row is the full parameter vector `compute_param_layout(data)` lays out
// -- fixed effects, random effects, every latent field, the hyperparameters --
// so eta at observation i is `generic_eta_at()` at that vector: the fixed part
// with its offset, the random effects, the areal / temporal / continuous fields
// and any varying coefficients, after each block's non-centered reconstruction
// and centring. Rebuilding the same sum in R from the draw columns is a second
// assembly, and one that forgets a component still returns a finite eta.
//
// The model is built from exactly the arguments the sampler received, so the
// layout the draws were laid out on and the layout read here are one layout;
// a column count that disagrees is an error rather than a misread.
// ----------------------------------------------------------------------------

#include <Rcpp.h>
#include <string>
#include <vector>

#include "hmc_sampler.h"           // ModelData / ParamLayout
#include "laplace_spec_fit.h"      // as_offset_vec
#include "log_post_impl.h"         // GenericLogPostState, generic_eta_at
#include "sampler_model_data.h"    // build_sampler_model_inputs
#include "sampler_log_prob.h"      // sampler_log_prob_rows

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_tulpa_glmm_eta_draws(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    double sigma_beta = 10.0,
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
    const int N = y.size();

    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);

    const tulpa_hmc::ModelData& data = in.data;
    const tulpa_hmc::ParamLayout& layout = in.layout;
    const int D = layout.total_params;
    if (draws.ncol() != D) {
        Rcpp::stop("cpp_tulpa_glmm_eta_draws: the draws carry %d columns but the "
                   "model lays out %d parameters.", draws.ncol(), D);
    }
    if (data.n_processes != 1) {
        Rcpp::stop("cpp_tulpa_glmm_eta_draws: a built-in family fit carries one "
                   "linear predictor; this model declares %d.",
                   data.n_processes);
    }

    const int S = draws.nrow();
    Rcpp::NumericMatrix out(S, N);
    std::vector<double> params(D);
    double eta = 0.0;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) params[j] = draws(s, j);
        tulpa::GenericLogPostState<double> state;
        tulpa::initialize_generic_state(params, data, layout, state);
        tulpa::precompute_generic_fixed_eta(data, state);
        for (int i = 0; i < N; i++) {
            tulpa::generic_eta_at(i, data, layout, state, &eta);
            out(s, i) = eta;
        }
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
    double sigma_beta = 10.0,
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
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);
    return tulpa::sampler_log_prob_rows(draws, in.data, in.layout);
}
