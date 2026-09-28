// test_smc_cabi.cpp
// Drives the `tulpa_smc_fit` C entry the way a model package does -- through
// R_GetCCallable and the flat SMCShimResult -- on a built-in-family GLM, so the
// shim's argument order, its buffer copies and the tempered population it
// returns are checked against the in-process driver the front door runs.

#include <Rcpp.h>
#include <string>
#include <vector>

#include "laplace_spec_fit.h"      // as_offset_vec
#include "sampler_model_data.h"    // build_sampler_model_inputs
#include "tulpa/smc_api.h"         // SmcFitFn, SMCShimResult, get_smc_fit_fn

// [[Rcpp::export]]
Rcpp::List cpp_test_smc_cabi(Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
                             Rcpp::NumericMatrix X, std::string family,
                             double phi, double sigma_beta, int n_particles,
                             int seed, double bridge_end) {
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(R_NilValue, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, NA_REAL, sigma_beta, offset, 2.5,
        R_NilValue, R_NilValue, R_NilValue);
    const int D = in.layout.total_params;
    std::vector<double> init(D, 0.0);
    tulpa::init_bounded_support_params(init, in.data, in.layout);

    tulpa::SmcFitFn fit = tulpa::get_smc_fit_fn();
    tulpa::SMCShimResult r;
    fit(&in.data, &in.layout, init.data(), D, n_particles, 5, 0.5, sigma_beta,
        nullptr, nullptr, (unsigned int)seed, 0, bridge_end, &r);
    if (!r.success) {
        std::string msg(r.error_msg);
        r.free_buffers();
        Rcpp::stop("tulpa_smc_fit failed: %s", msg);
    }

    auto rows = [](const double* buf, int n, int p) {
        Rcpp::NumericMatrix m(n, p);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++) m(i, j) = buf[(size_t)i * p + j];
        return m;
    };
    Rcpp::NumericMatrix draws = rows(r.particles, r.n_particles, r.n_params);
    Rcpp::RObject tempered = R_NilValue;
    if (r.n_tempered > 0) tempered = rows(r.tempered_particles, r.n_tempered, r.n_params);
    const double tempered_beta = r.tempered_beta;
    const bool tempered_null = (r.tempered_particles == nullptr);
    r.free_buffers();
    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("tempered_draws") = tempered,
        Rcpp::Named("tempered_beta") = tempered_beta,
        Rcpp::Named("tempered_null") = tempered_null);
}
