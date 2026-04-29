// test_log_post_split.cpp
// Rcpp wrappers exposing compute_log_post / compute_log_prior /
// compute_log_lik_only to R for unit testing.
//
// Internal-only: not roxygen-documented, not user-facing. Drives
// tests/testthat/test-log-post-split.R, which verifies that
// compute_log_post == compute_log_prior + compute_log_lik_only
// to within numerical tolerance (gcol33/tulpa#6 prereq).

#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <vector>

#include "hmc_sampler.h"
#include "model_data_rcpp.h"

using namespace Rcpp;
using namespace tulpa_hmc;

namespace {

// Build a ModelData + ParamLayout for the simple subset (binomial /
// poisson_gamma / negbin_negbin with optional RE / ICAR / BYM2 / RW / AR1)
// shared with the ESS / SGHMC / SGLD samplers. Test fixtures only.
struct TestSetup {
    tulpa::ModelData data;
    tulpa::ParamLayout layout;
};

TestSetup build_simple_setup(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    const std::string& model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params
) {
    TestSetup s;
    const int N = y_num.size();
    const int p_num = X_num.ncol();
    const int p_denom = X_denom.ncol();
    populate_model_data_simple(
        s.data, N, p_num, p_denom,
        y_num, y_denom, y_denom_cont,
        X_num, X_denom, model_type_str,
        re_params, spatial_params, temporal_params,
        prior_params, zi_params);
    s.layout = tulpa_hmc::compute_param_layout(s.data);
    return s;
}

} // namespace

// [[Rcpp::export]]
double cpp_compute_log_post_test(
    Rcpp::NumericVector params,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params
) {
    TestSetup s = build_simple_setup(
        y_num, y_denom, y_denom_cont, X_num, X_denom, model_type_str,
        re_params, spatial_params, temporal_params, prior_params, zi_params);
    if (params.size() != s.layout.total_params) {
        stop("params length (%d) does not match layout.total_params (%d)",
             (int)params.size(), s.layout.total_params);
    }
    std::vector<double> p = Rcpp::as<std::vector<double>>(params);
    return tulpa_hmc::compute_log_post(p, s.data, s.layout);
}

// [[Rcpp::export]]
double cpp_compute_log_prior_test(
    Rcpp::NumericVector params,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params
) {
    TestSetup s = build_simple_setup(
        y_num, y_denom, y_denom_cont, X_num, X_denom, model_type_str,
        re_params, spatial_params, temporal_params, prior_params, zi_params);
    if (params.size() != s.layout.total_params) {
        stop("params length (%d) does not match layout.total_params (%d)",
             (int)params.size(), s.layout.total_params);
    }
    std::vector<double> p = Rcpp::as<std::vector<double>>(params);
    return tulpa_hmc::compute_log_prior(p, s.data, s.layout);
}

// [[Rcpp::export]]
double cpp_compute_log_lik_only_test(
    Rcpp::NumericVector params,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params
) {
    TestSetup s = build_simple_setup(
        y_num, y_denom, y_denom_cont, X_num, X_denom, model_type_str,
        re_params, spatial_params, temporal_params, prior_params, zi_params);
    if (params.size() != s.layout.total_params) {
        stop("params length (%d) does not match layout.total_params (%d)",
             (int)params.size(), s.layout.total_params);
    }
    std::vector<double> p = Rcpp::as<std::vector<double>>(params);
    return tulpa_hmc::compute_log_lik_only(p, s.data, s.layout);
}

// [[Rcpp::export]]
int cpp_log_post_split_n_params(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params
) {
    TestSetup s = build_simple_setup(
        y_num, y_denom, y_denom_cont, X_num, X_denom, model_type_str,
        re_params, spatial_params, temporal_params, prior_params, zi_params);
    return s.layout.total_params;
}
