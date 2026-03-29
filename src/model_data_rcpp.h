// model_data_rcpp.h
// Shared helpers for populating ModelData from Rcpp arguments
// Used by ESS, SGHMC, and SGLD samplers (which support the same "simple" subset)
//
// The main HMC sampler has its own more extensive setup that includes
// GP, HSGP, SVC, multiscale temporal, latent factors, etc.

#ifndef TULPA_MODEL_DATA_RCPP_H
#define TULPA_MODEL_DATA_RCPP_H

#include <Rcpp.h>
#include <string>
#include <vector>
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "hmc_zi.h"

namespace tulpa_hmc {

using tulpa::ModelData;
using tulpa::ParamLayout;

// Populate ModelData from Rcpp arguments for simple samplers (ESS/SGHMC/SGLD).
// These samplers support: fixed effects, random effects, ICAR/BYM2 spatial,
// RW/AR temporal, and zero-inflation. They do NOT support GP, HSGP, SVC,
// multiscale temporal, latent factors, or spatiotemporal interactions.
inline void populate_model_data_simple(
    ModelData& data,
    int N, int p_num, int p_denom,
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
    data.N = N;
    data.legacy.p_num = p_num;
    data.legacy.p_denom = p_denom;

    // Copy response data
    data.legacy.y_num = Rcpp::as<std::vector<int>>(y_num);
    data.legacy.y_denom = Rcpp::as<std::vector<int>>(y_denom);
    data.legacy.y_denom_cont = Rcpp::as<std::vector<double>>(y_denom_cont);

    // Copy design matrices (row-major for cache efficiency)
    data.legacy.X_num_flat.resize(N * p_num);
    data.legacy.X_denom_flat.resize(N * p_denom);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p_num; j++) {
            data.legacy.X_num_flat[i * p_num + j] = X_num(i, j);
        }
        for (int j = 0; j < p_denom; j++) {
            data.legacy.X_denom_flat[i * p_denom + j] = X_denom(i, j);
        }
    }

    // Model type
    if (model_type_str == "binomial") {
        data.legacy.model_type = tulpa::ModelType::BINOMIAL;
    } else if (model_type_str == "negbin_negbin") {
        data.legacy.model_type = tulpa::ModelType::NEGBIN_NEGBIN;
    } else if (model_type_str == "poisson_gamma") {
        data.legacy.model_type = tulpa::ModelType::POISSON_GAMMA;
    } else {
        data.legacy.model_type = tulpa::ModelType::NEGBIN_NEGBIN;  // Default
    }

    // Random effects
    data.re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
    data.n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
    data.n_re_terms = 0;
    data.total_re_groups = data.n_re_groups;
    data.has_re_slopes = false;
    data.has_re_correlated_slopes = false;
    data.total_re_params = data.n_re_groups;
    data.total_sigma_params = (data.n_re_groups > 0) ? 1 : 0;
    data.total_chol_params = 0;

    // Prior parameters
    data.sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
    data.sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
    data.phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
    data.phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);

    // Spatial structure
    std::string spatial_type_str = Rcpp::as<std::string>(spatial_params["type"]);
    if (spatial_type_str == "icar" || spatial_type_str == "bym2") {
        if (spatial_type_str == "icar") {
            data.spatial_type = tulpa::SpatialType::ICAR;
        } else {
            data.spatial_type = tulpa::SpatialType::BYM2;
            data.bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);
        }
        data.spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
        data.n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
        data.adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
        data.adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
        data.n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
        data.tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
        data.tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);
    } else {
        data.spatial_type = tulpa::SpatialType::NONE;
        data.n_spatial_units = 0;
    }

    // Temporal structure
    std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
    if (temporal_type_str == "rw1") {
        data.temporal_type = tulpa::TemporalType::RW1;
    } else if (temporal_type_str == "rw2") {
        data.temporal_type = tulpa::TemporalType::RW2;
    } else if (temporal_type_str == "ar1") {
        data.temporal_type = tulpa::TemporalType::AR1;
    } else {
        data.temporal_type = tulpa::TemporalType::NONE;
    }

    if (data.temporal_type != tulpa::TemporalType::NONE) {
        data.temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
        data.temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
        data.n_times = Rcpp::as<int>(temporal_params["n_times"]);
        data.n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
        data.n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
        data.temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
        data.temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
        data.tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
        data.tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);
    } else {
        data.n_times = 0;
        data.n_temporal_groups = 0;
        data.n_temporal_params = 0;
        data.temporal_cyclic = false;
        data.temporal_shared = true;
    }

    // Zero-inflation
    std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);
    data.zi_type = tulpa_zi::parse_zi_type(zi_type_str);

    if (data.zi_type != tulpa::ZIType::NONE) {
        Rcpp::NumericMatrix X_zi = Rcpp::as<Rcpp::NumericMatrix>(zi_params["X"]);
        data.p_zi = X_zi.ncol();
        data.X_zi_flat.resize(N * data.p_zi);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < data.p_zi; j++) {
                data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
            }
        }
        data.zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);
    } else {
        data.p_zi = 0;
        data.zi_prior_sd = 10.0;
    }

    // Features not supported by simple samplers
    data.p_oi = 0;
    data.oi_prior_sd = 10.0;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;
    data.has_svc = false;
    data.has_rsr = false;
    data.svc_data.n_svc = 0;
    data.svc_data.n_obs = N;
    data.svc_data.nn = 0;
    data.svc_sigma2_prior_scale = 1.0;
    data.svc_phi_prior_lower = 0.1;
    data.svc_phi_prior_upper = 10.0;
    data.has_multiscale_temporal = false;
    data.has_latent = false;
    data.latent_n_factors = 0;
    data.has_spatiotemporal = false;
    data.n_threads = 1;
}

// Build parameter names for simple sampler output.
// Shared by ESS, SGHMC, and SGLD.
inline Rcpp::CharacterVector build_simple_param_names(
    int p_num, int p_denom,
    const ModelData& data,
    const ParamLayout& layout
) {
    int n_params = layout.total_params;
    Rcpp::CharacterVector param_names(n_params);
    int idx = 0;

    // Beta numerator
    for (int j = 0; j < p_num; j++) {
        param_names[idx++] = "beta_num[" + std::to_string(j + 1) + "]";
    }

    // Beta denominator
    for (int j = 0; j < p_denom; j++) {
        param_names[idx++] = "beta_denom[" + std::to_string(j + 1) + "]";
    }

    // Random effects
    if (layout.has_re) {
        param_names[idx++] = "log_sigma_re";
        for (int g = 0; g < data.n_re_groups; g++) {
            param_names[idx++] = "re[" + std::to_string(g + 1) + "]";
        }
    }

    // Overdispersion
    if (layout.legacy.has_phi_num) {
        param_names[idx++] = "log_phi_num";
    }
    if (layout.legacy.has_phi_denom) {
        param_names[idx++] = "log_phi_denom";
    }

    // Spatial
    if (layout.has_spatial) {
        param_names[idx++] = "log_tau_spatial";
        for (int s = 0; s < data.n_spatial_units; s++) {
            param_names[idx++] = "phi_spatial[" + std::to_string(s + 1) + "]";
        }
        if (layout.is_bym2) {
            param_names[idx++] = "log_sigma_bym2";
            param_names[idx++] = "logit_rho_bym2";
            for (int s = 0; s < data.n_spatial_units; s++) {
                param_names[idx++] = "theta_bym2[" + std::to_string(s + 1) + "]";
            }
        }
    }

    // Temporal
    if (layout.has_temporal) {
        param_names[idx++] = "log_tau_temporal";
        for (int t = 0; t < data.n_temporal_params; t++) {
            param_names[idx++] = "phi_temporal[" + std::to_string(t + 1) + "]";
        }
        if (layout.is_ar1) {
            param_names[idx++] = "logit_rho_ar1";
        }
    }

    // ZI
    if (layout.has_zi) {
        for (int j = 0; j < data.p_zi; j++) {
            param_names[idx++] = "beta_zi[" + std::to_string(j + 1) + "]";
        }
    }

    return param_names;
}

} // namespace tulpa_hmc

#endif // TULPA_MODEL_DATA_RCPP_H
