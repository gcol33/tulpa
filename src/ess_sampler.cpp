// ess_sampler.cpp
// Rcpp interface for Elliptical Slice Sampling
//
// NOTE: ESS is most efficient for simple models with random effects.
// For models with GP/HSGP spatial effects, HMC is recommended.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "ess_sampler.h"
#include "log_post_impl.h"
#include "model_data_rcpp.h"

using namespace Rcpp;
using namespace tulpa_hmc;
using namespace tulpa_ess;

// Implementation of log_post for ESS
namespace tulpa_ess {

double compute_log_post_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
    // Use the templated implementation from log_post_impl.h
    return tulpa::compute_log_post_impl<double>(params, data, layout);
}

} // namespace tulpa_ess

// [[Rcpp::export]]
Rcpp::List cpp_ess_fit(
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
    unsigned int seed,
    bool verbose
) {
    using namespace tulpa_hmc;

    // =========================================================================
    // Set up model data (shared with SGHMC/SGLD)
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
    // Configure ESS
    // =========================================================================
    ESSConfig config;
    config.n_iter = n_iter;
    config.n_warmup = n_warmup;
    config.n_thin = 1;
    config.verbose = verbose;
    config.print_every = 100;
    config.seed = seed;

    // =========================================================================
    // Run ESS sampler
    // =========================================================================
    ESSResult result = run_ess_sampler(init_params, data, layout, config);

    if (!result.success) {
        stop("ESS sampler failed: " + result.error_msg);
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

    // Build parameter names (shared with SGHMC/SGLD)
    CharacterVector param_names = build_simple_param_names(p_num, p_denom, data, layout);
    colnames(samples_out) = param_names;

    return List::create(
        Named("samples") = samples_out,
        Named("log_lik") = wrap(result.log_lik),
        Named("n_slice_evals") = result.n_slice_evals,
        Named("avg_slice_evals") = result.avg_slice_evals,
        Named("n_params") = n_params,
        Named("param_names") = param_names
    );
}

// [[Rcpp::export]]
int cpp_ess_get_n_params(
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List zi_params
) {
    // Simplified parameter counting for ESS
    int n_params = 0;

    // Fixed effects
    n_params += X_num.ncol();
    n_params += X_denom.ncol();

    // Random effects
    int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
    if (n_re_groups > 0) {
        n_params += 1;  // log_sigma_re
        n_params += n_re_groups;  // re values
    }

    // Overdispersion (depends on model type)
    if (model_type_str == "negbin_negbin") {
        n_params += 2;  // log_phi_num, log_phi_denom
    } else if (model_type_str == "poisson_gamma") {
        n_params += 1;  // log_phi (shape for gamma)
    }

    // Spatial
    std::string spatial_type = Rcpp::as<std::string>(spatial_params["type"]);
    if (spatial_type != "none") {
        int n_units = Rcpp::as<int>(spatial_params["n_units"]);
        n_params += 1;  // log_tau_spatial
        n_params += n_units;  // phi_spatial
        if (spatial_type == "bym2") {
            n_params += 2;  // log_sigma_bym2, logit_rho_bym2
            n_params += n_units;  // theta_bym2
        } else if (spatial_type == "car_proper") {
            n_params += 1;  // logit_rho_car
        }
    }

    // Temporal
    std::string temporal_type = Rcpp::as<std::string>(temporal_params["type"]);
    if (temporal_type != "none") {
        int n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
        n_params += 1;  // log_tau_temporal
        n_params += n_temporal_params;  // phi_temporal
        if (temporal_type == "ar1") {
            n_params += 1;  // logit_rho_ar1
        }
    }

    // ZI
    std::string zi_type = Rcpp::as<std::string>(zi_params["type"]);
    if (zi_type != "none") {
        NumericMatrix X_zi = Rcpp::as<NumericMatrix>(zi_params["X"]);
        n_params += X_zi.ncol();  // beta_zi
    }

    return n_params;
}
