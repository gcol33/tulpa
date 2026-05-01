// hmc_observation_likelihood.h
// Observation-level linear predictor assembly and likelihood evaluation.

#ifndef TULPA_HMC_OBSERVATION_LIKELIHOOD_H
#define TULPA_HMC_OBSERVATION_LIKELIHOOD_H

#include <vector>
#include "hmc_likelihood.h"
#include "linalg_fast.h"

namespace tulpa_hmc {

struct ObservationLikelihoodContext {
  const std::vector<double>& params;
  const tulpa::ModelData& data;
  const tulpa::ParamLayout& layout;

  const double* beta_num = nullptr;
  const double* beta_denom = nullptr;
  const double* re = nullptr;
  const double* re_nc_flat = nullptr;

  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  double sigma_s_bym2 = 1.0;
  double sigma_u_bym2 = 1.0;

  const double* phi_temporal = nullptr;
  const double* gp_w = nullptr;
  const double* gp_local = nullptr;
  const double* gp_regional = nullptr;
  const std::vector<double>* msgp_hsgp_f_local = nullptr;
  const std::vector<double>* msgp_hsgp_f_regional = nullptr;
  const std::vector<double>* hsgp_f = nullptr;

  const double* trend = nullptr;
  const double* seasonal = nullptr;
  const double* short_term = nullptr;

  const std::vector<double>* latent_sigma = nullptr;
  const std::vector<double>* latent_factors = nullptr;
  const double* st_delta = nullptr;
  const std::vector<double>* tvc_eta = nullptr;
  const double* svc_eta = nullptr;

  const double* beta_zi = nullptr;
  const double* beta_oi = nullptr;
  double phi_num = 1.0;
  double phi_denom = 1.0;
};

inline double compute_observation_log_lik(
    const ObservationLikelihoodContext& ctx,
    int i
) {
  const auto& data = ctx.data;
  const auto& layout = ctx.layout;
  const auto& params = ctx.params;

  double eta_num = tulpa_linalg::dot_product(
      &data.legacy.X_num_flat[i * data.legacy.p_num],
      ctx.beta_num, data.legacy.p_num);
  double eta_denom = tulpa_linalg::dot_product(
      &data.legacy.X_denom_flat[i * data.legacy.p_denom],
      ctx.beta_denom, data.legacy.p_denom);

  if (layout.has_re) {
    int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

    if (layout.has_re_slopes) {
      for (int t = 0; t < n_terms; t++) {
        int group_idx = data.re_group_multi_flat[i * n_terms + t];
        if (group_idx <= 0) continue;

        int g = group_idx - 1;
        int n_coefs = layout.re_n_coefs_multi[t];
        int re_base = layout.re_start_multi[t] + g * n_coefs;
        bool is_corr_t = layout.re_correlated_multi[t] && n_coefs > 1;
        bool use_nc = is_corr_t || (data.re_parameterization == 1);
        const double* re_vals = use_nc ? ctx.re_nc_flat : params.data();
        double re_contrib = re_vals[re_base];

        int n_slopes = n_coefs - 1;
        if (n_slopes > 0 && !data.re_slope_matrices[t].empty()) {
          for (int s = 0; s < n_slopes; s++) {
            double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
            re_contrib += re_vals[re_base + 1 + s] * x_slope;
          }
        }

        eta_num += re_contrib;
        eta_denom += re_contrib;
      }
    } else if (n_terms > 1) {
      for (int t = 0; t < n_terms; t++) {
        int group_idx = data.re_group_multi_flat[i * n_terms + t];
        if (group_idx <= 0) continue;

        int g = group_idx - 1;
        int re_idx = layout.re_start_multi[t] + g;
        double re_val = (data.re_parameterization == 1)
            ? ctx.re_nc_flat[re_idx]
            : params[re_idx];
        eta_num += re_val;
        eta_denom += re_val;
      }
    } else if (data.re_group[i] > 0) {
      int g = data.re_group[i] - 1;
      double re_val = (data.re_parameterization == 1)
          ? ctx.re_nc_flat[layout.re_start + g]
          : ctx.re[g];
      eta_num += re_val;
      eta_denom += re_val;
    }
  }

  if (layout.has_spatial && ctx.phi_spatial != nullptr &&
      !data.spatial_group.empty() && data.spatial_group[i] > 0) {
    int s = data.spatial_group[i] - 1;
    double spatial_effect = ctx.phi_spatial[s];
    if (layout.is_bym2) {
      double scaled_phi = ctx.phi_spatial[s] * data.bym2_scale_factor;
      spatial_effect = ctx.sigma_s_bym2 * scaled_phi +
                       ctx.sigma_u_bym2 * ctx.theta_bym2[s];
    }
    eta_num += spatial_effect;
    eta_denom += spatial_effect;
  }

  if (layout.has_temporal && !data.temporal_time_idx.empty() &&
      data.temporal_time_idx[i] > 0) {
    int t = data.temporal_time_idx[i] - 1;
    int g = data.temporal_group_idx[i] - 1;
    double temporal_effect = ctx.phi_temporal[g * data.n_times + t];
    if (data.temporal_shared) {
      eta_num += temporal_effect;
      eta_denom += temporal_effect;
    } else {
      eta_num += temporal_effect;
    }
  }

  if (layout.is_gp && data.has_gp && ctx.gp_w != nullptr) {
    double gp_effect = ctx.gp_w[data.gp_data.obs_to_loc[i]];
    if (data.gp_data.shared) {
      eta_num += gp_effect;
      eta_denom += gp_effect;
    } else {
      eta_num += gp_effect;
    }
  }

  if (layout.is_multiscale_gp && data.has_multiscale_gp) {
    double ms_spatial_effect = 0.0;
    if (data.msgp_is_hsgp) {
      ms_spatial_effect = (*ctx.msgp_hsgp_f_local)[i] +
                          (*ctx.msgp_hsgp_f_regional)[i];
    } else {
      int loc_i = data.multiscale_gp_data.obs_to_loc[i];
      ms_spatial_effect = ctx.gp_local[loc_i] + ctx.gp_regional[loc_i];
    }
    if (data.multiscale_gp_data.shared) {
      eta_num += ms_spatial_effect;
      eta_denom += ms_spatial_effect;
    } else {
      eta_num += ms_spatial_effect;
    }
  }

  if (layout.is_hsgp && data.has_hsgp && ctx.hsgp_f != nullptr &&
      !ctx.hsgp_f->empty()) {
    double hsgp_effect = (*ctx.hsgp_f)[i];
    if (data.hsgp_data.shared) {
      eta_num += hsgp_effect;
      eta_denom += hsgp_effect;
    } else {
      eta_num += hsgp_effect;
    }
  }

  if (layout.has_multiscale_temporal) {
    double ms_temporal_effect = 0.0;
    int t_idx = data.multiscale_temporal_data.time_index[i] - 1;
    if (ctx.trend != nullptr && t_idx >= 0 &&
        t_idx < static_cast<int>(data.multiscale_temporal_data.n_times)) {
      ms_temporal_effect += ctx.trend[t_idx];
    }
    if (ctx.seasonal != nullptr &&
        data.multiscale_temporal_data.seasonal_period > 0) {
      int s_idx = t_idx % data.multiscale_temporal_data.seasonal_period;
      ms_temporal_effect += ctx.seasonal[s_idx];
    }
    if (ctx.short_term != nullptr && t_idx >= 0 &&
        t_idx < static_cast<int>(data.multiscale_temporal_data.n_times)) {
      ms_temporal_effect += ctx.short_term[t_idx];
    }
    if (data.multiscale_temporal_data.shared) {
      eta_num += ms_temporal_effect;
      eta_denom += ms_temporal_effect;
    } else {
      eta_num += ms_temporal_effect;
    }
  }

  if (layout.has_latent && data.latent_n_factors > 0 &&
      ctx.latent_factors != nullptr && !ctx.latent_factors->empty()) {
    double latent_effect = 0.0;
    for (int k = 0; k < data.latent_n_factors; k++) {
      latent_effect += (*ctx.latent_factors)[i * data.latent_n_factors + k] *
                       (*ctx.latent_sigma)[k];
    }
    if (data.latent_shared) {
      eta_num += latent_effect;
      eta_denom += latent_effect;
    } else {
      eta_num += latent_effect;
    }
  }

  if (layout.has_spatiotemporal && ctx.st_delta != nullptr) {
    double st_effect = 0.0;
    if (data.st_is_hsgp) {
      int t = data.spatiotemporal_data.t_idx[i] - 1;
      int M = data.st_hsgp_data.m_total;
      int T_st = data.spatiotemporal_data.n_times;
      for (int j = 0; j < M; j++) {
        st_effect += data.st_hsgp_data.phi_flat[i * M + j] *
                     ctx.st_delta[j * T_st + t];
      }
    } else {
      int st_idx = data.spatiotemporal_data.st_flat[i];
      if (st_idx > 0) st_effect = ctx.st_delta[st_idx - 1];
    }
    if (data.spatiotemporal_data.shared) {
      eta_num += st_effect;
      eta_denom += st_effect;
    } else {
      eta_num += st_effect;
    }
  }

  if (layout.has_tvc && ctx.tvc_eta != nullptr && !ctx.tvc_eta->empty()) {
    double tvc_effect = (*ctx.tvc_eta)[i];
    if (data.tvc_data.shared) {
      eta_num += tvc_effect;
      eta_denom += tvc_effect;
    } else {
      eta_num += tvc_effect;
    }
  }

  if (layout.has_svc && ctx.svc_eta != nullptr) {
    double svc_effect = ctx.svc_eta[i];
    if (data.svc_data.shared) {
      eta_num += svc_effect;
      eta_denom += svc_effect;
    } else {
      eta_num += svc_effect;
    }
  }

  double logit_zi = 0.0;
  if (layout.has_zi) {
    logit_zi = tulpa_linalg::dot_product(
        &data.X_zi_flat[i * data.p_zi], ctx.beta_zi, data.p_zi);
  }

  double logit_oi = 0.0;
  if (layout.has_oi && data.p_oi > 0) {
    logit_oi = tulpa_linalg::dot_product(
        &data.X_oi_flat[i * data.p_oi], ctx.beta_oi, data.p_oi);
  }

  return compute_obs_ll_with_inflation(
      data, layout, i, eta_num, eta_denom, ctx.phi_num, ctx.phi_denom,
      logit_zi, logit_oi);
}

} // namespace tulpa_hmc

#endif // TULPA_HMC_OBSERVATION_LIKELIHOOD_H
