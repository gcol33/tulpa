#ifndef TULPA_PARAM_LAYOUT_H
#define TULPA_PARAM_LAYOUT_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// ParamLayout: maps named parameter blocks to positions in the flat
// parameter vector used by samplers.
//
// Legacy ratio layout:
//   [beta_num | beta_denom | log_sigma_re? | re? | phi_num? | phi_denom? |
//    spatial? | temporal? | zi? | oi? | svc? | gp? | hsgp? | latent? | st? | tvc?]
//
// Generic multi-process layout:
//   [process_0_beta | process_1_beta | ... | re_params | spatial_params |
//    temporal_params | svc_params | tvc_params | st_params | zi_params |
//    oi_params | latent_params | model_extra_params]
// ============================================================================
struct ParamLayout {
    int total_params = 0;

    // ================================================================
    // GENERIC MULTI-PROCESS LAYOUT
    // Used when n_processes > 0 in ModelData.
    // ================================================================
    std::vector<int> process_beta_start;
    std::vector<int> process_beta_count;
    int extra_offset = -1;          // Model-specific extra parameters
    int n_extra_params = 0;

    // ================================================================
    // LEGACY RATIO-SPECIFIC LAYOUT
    // Used only when n_processes == 0. Will move to numdenom.
    // Access via layout.legacy.beta_num_start, etc.
    // ================================================================
    struct LegacyRatioLayout {
        int beta_num_start = -1, beta_num_end = -1;
        int beta_denom_start = -1, beta_denom_end = -1;
        int log_phi_num_idx = -1;
        int log_phi_denom_idx = -1;
        bool has_phi_num = false;
        bool has_phi_denom = false;
    } legacy;

    // ================================================================
    // RANDOM EFFECTS
    // ================================================================
    bool has_re = false;
    int log_sigma_re_idx = -1;      // Legacy: index for single RE term
    int re_start = -1, re_end = -1; // Legacy: bounds for single RE term

    // Multi-term RE layout
    std::vector<int> log_sigma_re_multi;    // Index of log_sigma_re for each term
    std::vector<int> re_start_multi;        // Start index for each RE term
    std::vector<int> re_end_multi;          // End index for each RE term

    // Random slopes layout
    bool has_re_slopes = false;
    bool has_re_correlated_slopes = false;
    std::vector<int> re_n_coefs_multi;
    std::vector<bool> re_correlated_multi;
    std::vector<std::vector<int>> log_sigma_re_slopes;
    std::vector<int> chol_re_start_multi;
    std::vector<int> chol_re_end_multi;

    // ================================================================
    // SPATIAL
    // ================================================================
    bool has_spatial = false;
    bool is_bym2 = false;
    bool is_gp = false;
    bool is_icar_collapsed = false;
    bool is_bym2_collapsed = false;
    bool is_gp_collapsed = false;
    bool is_multiscale_gp = false;
    bool is_hsgp = false;
    bool is_car_proper = false;          // Proper CAR (rho estimated)

    int log_tau_spatial_idx = -1;
    int spatial_start = -1, spatial_end = -1;

    // BYM2 extras
    int log_sigma_bym2_idx = -1;
    int logit_rho_bym2_idx = -1;
    int theta_bym2_start = -1, theta_bym2_end = -1;

    // Proper CAR extra: logit-transformed rho (rho ∈ (rho_lower, rho_upper))
    int logit_rho_car_idx = -1;

    // GP spatial parameters
    int log_sigma2_gp_idx = -1;
    int log_phi_gp_idx = -1;
    int gp_w_start = -1, gp_w_end = -1;

    // Multi-scale GP parameters
    int log_sigma2_gp_local_idx = -1;
    int log_phi_gp_local_idx = -1;
    int log_sigma2_gp_regional_idx = -1;
    int log_phi_gp_regional_idx = -1;
    int gp_local_start = -1, gp_local_end = -1;
    int gp_regional_start = -1, gp_regional_end = -1;

    // HSGP parameters
    int log_sigma2_hsgp_idx = -1;
    int log_lengthscale_hsgp_idx = -1;
    int hsgp_beta_start = -1, hsgp_beta_end = -1;

    // ================================================================
    // TEMPORAL
    // ================================================================
    bool has_temporal = false;
    bool is_ar1 = false;
    bool is_temporal_gp = false;
    bool has_multiscale_temporal = false;

    int log_tau_temporal_idx = -1;
    int logit_rho_ar1_idx = -1;
    int temporal_start = -1, temporal_end = -1;

    // Temporal GP parameters
    int log_sigma2_temporal_gp_idx = -1;
    int logit_phi_temporal_gp_idx = -1;

    // Multi-scale temporal parameters
    int log_sigma2_trend_idx = -1;
    int log_sigma2_seasonal_idx = -1;
    int log_sigma2_short_idx = -1;
    int logit_rho_short_idx = -1;
    int trend_start = -1, trend_end = -1;
    int seasonal_start = -1, seasonal_end = -1;
    int short_term_start = -1, short_term_end = -1;

    // ================================================================
    // ZERO-INFLATION
    // ================================================================
    bool has_zi = false;
    int beta_zi_start = -1, beta_zi_end = -1;

    // ================================================================
    // ONE-INFLATION
    // ================================================================
    bool has_oi = false;
    int beta_oi_start = -1, beta_oi_end = -1;

    // ================================================================
    // SVC
    // ================================================================
    bool has_svc = false;
    int log_sigma2_svc_start = -1, log_sigma2_svc_end = -1;
    int log_phi_svc_start = -1, log_phi_svc_end = -1;
    int svc_w_start = -1, svc_w_end = -1;

    // ================================================================
    // LATENT FACTORS
    // ================================================================
    bool has_latent = false;
    int log_sigma_latent_start = -1, log_sigma_latent_end = -1;
    int latent_factor_start = -1, latent_factor_end = -1;

    // ================================================================
    // SPATIOTEMPORAL INTERACTION
    // ================================================================
    bool has_spatiotemporal = false;
    bool is_st_gp = false;
    bool is_st_hsgp = false;

    int log_tau_st_idx = -1;
    int log_tau_st2_idx = -1;
    int logit_rho_st_idx = -1;
    int log_phi_st_space_idx = -1;
    int log_phi_st_time_idx = -1;
    int st_delta_start = -1, st_delta_end = -1;

    // HSGP-ST parameters
    int log_sigma2_st_hsgp_idx = -1;
    int log_lengthscale_st_hsgp_idx = -1;

    // ================================================================
    // TVC
    // ================================================================
    bool has_tvc = false;
    int log_tau_tvc_start = -1, log_tau_tvc_end = -1;
    int logit_rho_tvc_start = -1, logit_rho_tvc_end = -1;
    int tvc_w_start = -1, tvc_w_end = -1;
};

} // namespace tulpa

#endif // TULPA_PARAM_LAYOUT_H
