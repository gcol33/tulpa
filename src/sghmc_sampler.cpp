// sghmc_sampler.cpp
// Rcpp interface for Stochastic Gradient HMC and SGLD
//
// These methods enable Tier 1 (Exact) inference on large datasets
// by using minibatch gradients with appropriate noise corrections.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "sghmc_sampler.h"
#include "model_data_rcpp.h"

using namespace Rcpp;
using namespace tulpa_hmc;
using namespace tulpa_sghmc;

// [[Rcpp::export]]
Rcpp::List cpp_sghmc_fit(
    Rcpp::NumericVector q_init,
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
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double alpha,
    int L,
    unsigned int seed,
    bool verbose
) {
    using namespace tulpa_hmc;

    // =========================================================================
    // Set up model data (shared with ESS/SGLD)
    // =========================================================================

    int N = y_num.size();
    int p_num = X_num.ncol();
    int p_denom = X_denom.ncol();

    ModelData data;
    populate_model_data_simple(data, N, p_num, p_denom,
        y_num, y_denom, y_denom_cont, X_num, X_denom,
        model_type_str, re_params, spatial_params, temporal_params,
        prior_params, zi_params);

    // =========================================================================
    // Compute parameter layout
    // =========================================================================
    ParamLayout layout = compute_param_layout(data);
    int n_params = layout.total_params;

    // =========================================================================
    // Initialize parameters
    // =========================================================================
    std::vector<double> init_params(n_params, 0.0);
    if (q_init.size() == n_params) {
        init_params = Rcpp::as<std::vector<double>>(q_init);
    }

    // =========================================================================
    // Configure SGHMC
    // =========================================================================
    SGHMCConfig config;
    config.n_iter = n_iter;
    config.n_warmup = n_warmup;
    config.n_thin = 1;
    config.batch_size = std::min(batch_size, N);  // Can't exceed N
    config.epsilon = epsilon;
    config.alpha = alpha;
    config.L = L;
    config.verbose = verbose;
    config.print_every = 100;
    config.seed = seed;
    config.adapt_epsilon = true;

    // =========================================================================
    // Run SGHMC sampler
    // =========================================================================
    SGHMCResult result = run_sghmc_sampler(init_params, data, layout, config);

    if (!result.success) {
        stop("SGHMC sampler failed: " + result.error_msg);
    }

    // =========================================================================
    // Build return list
    // =========================================================================
    int n_save = result.samples.rows();

    NumericMatrix samples_out(n_save, n_params);
    for (int i = 0; i < n_save; i++) {
        for (int j = 0; j < n_params; j++) {
            samples_out(i, j) = result.samples(i, j);
        }
    }

    // Build parameter names (shared with ESS/SGLD)
    CharacterVector param_names = build_simple_param_names(p_num, p_denom, data, layout);
    colnames(samples_out) = param_names;

    return List::create(
        Named("samples") = samples_out,
        Named("log_lik") = wrap(result.log_lik),
        Named("epsilon_history") = wrap(result.epsilon_history),
        Named("n_params") = n_params,
        Named("param_names") = param_names,
        Named("batch_size") = config.batch_size,
        Named("final_epsilon") = result.epsilon_history.back()
    );
}


// [[Rcpp::export]]
Rcpp::List cpp_sgld_fit(
    Rcpp::NumericVector q_init,
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
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double schedule_a,
    double schedule_b,
    double schedule_gamma,
    bool use_schedule,
    unsigned int seed,
    bool verbose
) {
    using namespace tulpa_hmc;

    // =========================================================================
    // Set up model data (shared with ESS/SGHMC)
    // =========================================================================

    int N = y_num.size();
    int p_num = X_num.ncol();
    int p_denom = X_denom.ncol();

    ModelData data;
    populate_model_data_simple(data, N, p_num, p_denom,
        y_num, y_denom, y_denom_cont, X_num, X_denom,
        model_type_str, re_params, spatial_params, temporal_params,
        prior_params, zi_params);

    // =========================================================================
    // Compute parameter layout
    // =========================================================================
    ParamLayout layout = compute_param_layout(data);
    int n_params = layout.total_params;

    // =========================================================================
    // Initialize parameters
    // =========================================================================
    std::vector<double> init_params(n_params, 0.0);
    if (q_init.size() == n_params) {
        init_params = Rcpp::as<std::vector<double>>(q_init);
    }

    // =========================================================================
    // Configure SGLD
    // =========================================================================
    SGLDConfig config;
    config.n_iter = n_iter;
    config.n_warmup = n_warmup;
    config.n_thin = 1;
    config.batch_size = std::min(batch_size, N);
    config.epsilon = epsilon;
    config.verbose = verbose;
    config.print_every = 100;
    config.seed = seed;
    config.schedule_a = schedule_a;
    config.schedule_b = schedule_b;
    config.schedule_gamma = schedule_gamma;
    config.use_schedule = use_schedule;

    // =========================================================================
    // Run SGLD sampler
    // =========================================================================
    SGLDResult result = run_sgld_sampler(init_params, data, layout, config);

    if (!result.success) {
        stop("SGLD sampler failed: " + result.error_msg);
    }

    // =========================================================================
    // Build return list
    // =========================================================================
    int n_save = result.samples.rows();

    NumericMatrix samples_out(n_save, n_params);
    for (int i = 0; i < n_save; i++) {
        for (int j = 0; j < n_params; j++) {
            samples_out(i, j) = result.samples(i, j);
        }
    }

    // Build parameter names (shared with ESS/SGHMC)
    CharacterVector param_names = build_simple_param_names(p_num, p_denom, data, layout);
    colnames(samples_out) = param_names;

    return List::create(
        Named("samples") = samples_out,
        Named("log_lik") = wrap(result.log_lik),
        Named("epsilon_history") = wrap(result.epsilon_history),
        Named("n_params") = n_params,
        Named("param_names") = param_names,
        Named("batch_size") = config.batch_size
    );
}
