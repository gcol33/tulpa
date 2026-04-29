#ifndef TULPA_MODEL_DATA_H
#define TULPA_MODEL_DATA_H

#include <vector>
#include <cstdint>
#include "tulpa/types.h"
#include "tulpa/gp_data.h"
#include "tulpa/hsgp_data.h"
#include "tulpa/temporal_data.h"
#include "tulpa/svc_data.h"
#include "tulpa/tvc_data.h"
#include "tulpa/st_data.h"

namespace tulpa {

// ============================================================================
// ABI version — pre-release, not yet maintained.
// Will start tracking layout / shim-family changes here once the package
// has its first tagged release. Until then this stays at 1; downstream
// packages should be rebuilt against the current tulpa source.
// ============================================================================
constexpr int TULPA_ABI_VERSION = 2;

// ============================================================================
// Per-process design matrix and fixed effects (generic multi-process interface)
// ============================================================================
struct ProcessData {
    std::vector<double> X_flat;  // Design matrix [N x p], row-major
    int p = 0;                    // Number of fixed effect columns
};

// ============================================================================
// Sharing specification: which latent components enter which processes
// ============================================================================
struct SharingSpec {
    std::vector<bool> spatial;
    std::vector<bool> temporal;
    std::vector<bool> re;
    std::vector<bool> latent;
    std::vector<bool> svc;
    std::vector<bool> tvc;
    std::vector<bool> st;
    std::vector<bool> zi;

    void init(int n_processes) {
        spatial.assign(n_processes, true);
        temporal.assign(n_processes, true);
        re.assign(n_processes, true);
        latent.assign(n_processes, true);
        svc.assign(n_processes, true);
        tvc.assign(n_processes, true);
        st.assign(n_processes, true);
        zi.assign(n_processes, false);
        if (n_processes > 0) zi[0] = true;
    }
};

// ============================================================================
// ModelData: the core data container passed to all samplers
//
// STABILITY CONTRACT:
//   - Generic multi-process fields (n_processes, processes, model_response_data,
//     likelihood_spec, sharing) are STABLE. Model packages depend on them.
//   - Shared infrastructure fields (spatial, temporal, RE, GP, etc.) are STABLE.
//     They are used by both generic and legacy paths.
//   - Legacy ratio fields (y_num, y_denom, X_num/denom_flat, p_num/p_denom,
//     model_type) are DEPRECATED — used only when n_processes == 0.
//     They will move to numdenom when it is rebuilt on tulpa.
//
// LAYOUT RULE: New fields go in the appropriate stable section.
//   Never insert fields before existing ones — append within sections.
//   Bump TULPA_ABI_VERSION when any field is added, removed, or reordered.
//
// When n_processes > 0, the generic multi-process interface is active.
// When n_processes == 0, the legacy ratio interface is used.
// ============================================================================
struct ModelData {
    // Unique ID for cache invalidation (incremented per construction)
    uint64_t unique_id;
    static inline uint64_t next_id() {
        static uint64_t counter = 0;
        return ++counter;
    }
    ModelData() : unique_id(next_id()) {}

    // ================================================================
    // GENERIC MULTI-PROCESS INTERFACE
    // Model packages set these instead of the legacy ratio fields.
    // ================================================================
    int n_processes = 0;  // 0 = legacy ratio mode, >0 = generic multi-process
    std::vector<ProcessData> processes;
    void* model_response_data = nullptr;  // Opaque — owned by model package
    const void* likelihood_spec = nullptr; // Points to LikelihoodSpec (owned by caller)
    SharingSpec sharing;

    // ================================================================
    // DIMENSIONS
    // ================================================================
    int N = 0;  // Number of observations

    // ================================================================
    // LEGACY RATIO-SPECIFIC FIELDS
    // Used only when n_processes == 0. Will move to numdenom.
    // Access via data.legacy.y_num, data.legacy.model_type, etc.
    // ================================================================
    struct LegacyRatioData {
        std::vector<int> y_num;
        std::vector<int> y_denom;
        std::vector<double> y_num_cont;     // Continuous numerator (gamma_gamma, lognormal)
        std::vector<double> y_denom_cont;   // Continuous denominator
        std::vector<double> X_num_flat;     // Numerator design matrix [N x p_num], row-major
        std::vector<double> X_denom_flat;   // Denominator design matrix [N x p_denom], row-major
        int p_num = 0;
        int p_denom = 0;
        ModelType model_type = ModelType::BINOMIAL;
    } legacy;

    // ================================================================
    // RANDOM EFFECTS
    // ================================================================
    // Legacy single-term RE
    std::vector<int> re_group;          // 1-based group index (0 = no RE)
    int n_re_groups = 0;

    // Multi-term RE structure
    int n_re_terms = 0;
    std::vector<std::vector<int>> re_group_multi;   // [term][obs] -> group (1-based), legacy
    std::vector<int> re_group_multi_flat;            // [obs * n_re_terms + term], obs-major
    std::vector<int> re_n_groups_multi;              // Groups per term
    std::vector<int> re_offsets;                     // Offset in flattened RE param vector per term
    int total_re_groups = 0;

    // Random slopes
    bool has_re_slopes = false;
    bool has_re_correlated_slopes = false;
    std::vector<int> re_n_coefs;                     // Coefficients per group per term
    std::vector<std::vector<double>> re_slope_matrices; // [term] -> flattened [N x n_slopes]
    std::vector<int> re_n_slopes;                    // Slope variables per term
    std::vector<bool> re_correlated;                 // Correlated slopes per term
    std::vector<int> re_n_chol;                      // Cholesky params per term
    int total_re_params = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;

    // RE parameterization: 0 = centered, 1 = non-centered
    int re_parameterization = 1;

    // ================================================================
    // SPATIAL
    // ================================================================
    SpatialType spatial_type = SpatialType::NONE;
    std::vector<int> spatial_group;     // Maps obs to spatial unit (1-based)
    int n_spatial_units = 0;

    // ICAR / BYM2 / CAR_PROPER adjacency (CSR format)
    std::vector<int> adj_row_ptr;
    std::vector<int> adj_col_idx;
    std::vector<int> n_neighbors;
    double bym2_scale_factor = 1.0;

    // Proper CAR rho bounds (eigenvalue-derived, default to (0, 1))
    // Only used when spatial_type == CAR_PROPER.
    double car_rho_lower = 0.0;
    double car_rho_upper = 1.0;

    // Precision mass matrix data (precomputed from Q)
    std::vector<double> spatial_Q_inv;  // (Q + lambda*I)^{-1}, column-major [S x S]
    std::vector<double> spatial_L_Q;    // Cholesky L of (Q + lambda*I), column-major [S x S]

    // GP spatial (single-scale NNGP)
    GPData gp_data;
    bool has_gp = false;
    double gp_sigma2_prior_U = 1.0;
    double gp_sigma2_prior_alpha = 0.01;
    double gp_phi_prior_lower = 0.01;
    double gp_phi_prior_upper = 10.0;
    int gp_parameterization = 1;        // 0=centered, 1=non-centered

    // Collapsed parameterization flags
    bool icar_collapsed = false;
    bool bym2_collapsed = false;
    bool gp_collapsed = false;

    // Multi-scale GP
    MultiscaleGPData multiscale_gp_data;
    bool has_multiscale_gp = false;
    bool msgp_is_hsgp = false;
    HSGPData msgp_hsgp_data;
    double ms_sigma2_local_prior_U = 1.0;
    double ms_sigma2_local_prior_alpha = 0.01;
    double ms_sigma2_regional_prior_U = 1.0;
    double ms_sigma2_regional_prior_alpha = 0.01;
    double ms_log_ls_local_mean = -1.0;
    double ms_log_ls_local_sd = 0.5;
    double ms_log_ls_regional_mean = 1.0;
    double ms_log_ls_regional_sd = 0.5;

    // HSGP (Hilbert Space GP)
    HSGPData hsgp_data;
    bool has_hsgp = false;
    int hsgp_m_per_dim = 15;
    double hsgp_boundary_factor = 1.5;

    // RSR (Restricted Spatial Regression)
    bool has_rsr = false;
    std::vector<double> rsr_projection;   // P_perp matrix (n x n, flattened)
    int rsr_n = 0;

    // ================================================================
    // SVC (Spatially-Varying Coefficients)
    // ================================================================
    SVCData svc_data;
    bool has_svc = false;
    bool svc_is_hsgp = false;
    HSGPData svc_hsgp_data;
    int svc_hsgp_m_per_dim = 6;
    double svc_hsgp_boundary_factor = 1.5;
    double svc_sigma2_prior_scale = 1.0;
    double svc_phi_prior_lower = 0.01;
    double svc_phi_prior_upper = 10.0;

    // ================================================================
    // TEMPORAL
    // ================================================================
    TemporalType temporal_type = TemporalType::NONE;
    std::vector<int> temporal_time_idx;     // Maps obs to time point (1-based)
    std::vector<int> temporal_group_idx;    // Maps obs to temporal group (1-based)
    int n_times = 0;
    int n_temporal_groups = 1;
    int n_temporal_params = 0;
    bool temporal_cyclic = false;
    bool temporal_shared = true;            // Legacy: shared between num/denom
    double tau_temporal_shape = 1.0;
    double tau_temporal_rate = 0.01;

    // Temporal GP (irregularly-spaced)
    TemporalGPData temporal_gp_data;
    bool has_temporal_gp = false;
    double temporal_gp_sigma2_prior_U = 1.0;
    double temporal_gp_sigma2_prior_alpha = 0.01;
    double temporal_gp_phi_prior_lower = 0.01;
    double temporal_gp_phi_prior_upper = 10.0;
    int temporal_gp_parameterization = 1;

    // Multi-scale temporal
    MultiscaleTemporalData multiscale_temporal_data;
    bool has_multiscale_temporal = false;
    double ms_sigma2_trend_prior_U = 1.0;
    double ms_sigma2_trend_prior_alpha = 0.01;
    double ms_sigma2_seasonal_prior_U = 1.0;
    double ms_sigma2_seasonal_prior_alpha = 0.01;
    double ms_sigma2_short_prior_U = 1.0;
    double ms_sigma2_short_prior_alpha = 0.01;

    // ================================================================
    // TVC (Temporally-Varying Coefficients)
    // ================================================================
    TVCData tvc_data;
    bool has_tvc = false;
    double tvc_tau_shape = 1.0;
    double tvc_tau_rate = 0.01;

    // ================================================================
    // ZERO-INFLATION
    // ================================================================
    ZIType zi_type = ZIType::NONE;
    std::vector<double> X_zi_flat;
    int p_zi = 0;
    double zi_prior_sd = 2.5;

    // ================================================================
    // ONE-INFLATION (OI-binomial and ZOIB)
    // ================================================================
    std::vector<double> X_oi_flat;
    int p_oi = 0;
    double oi_prior_sd = 2.5;

    // ================================================================
    // LATENT FACTORS
    // ================================================================
    bool has_latent = false;
    int latent_n_factors = 0;
    bool latent_shared = true;
    bool latent_scale = true;
    int latent_constraint = 0;          // 0 = sum_to_zero, 1 = first_zero
    double latent_sigma_prior_rate = 1.0;

    // ================================================================
    // SPATIOTEMPORAL INTERACTION
    // ================================================================
    bool has_spatiotemporal = false;
    bool st_is_hsgp = false;
    SpatiotemporalData spatiotemporal_data;
    HSGPData st_hsgp_data;
    int st_parameterization = 0;
    double st_sigma2_prior_U = 1.0;
    double st_sigma2_prior_alpha = 0.01;
    double st_phi_space_prior_lower = 0.01;
    double st_phi_space_prior_upper = 10.0;
    double st_phi_time_prior_lower = 0.01;
    double st_phi_time_prior_upper = 10.0;

    // Kronecker precision data for ST_IV (precomputed in R)
    std::vector<double> st_Qs_inv;
    std::vector<double> st_Ls;
    std::vector<double> st_Qt_inv;
    std::vector<double> st_Lt;

    // ================================================================
    // PRIOR HYPERPARAMETERS
    // ================================================================
    double sigma_beta = 2.5;
    double sigma_re_scale = 1.0;
    double phi_prior_shape = 1.0;
    double phi_prior_rate = 0.01;
    double tau_spatial_shape = 1.0;
    double tau_spatial_rate = 0.01;

    // ================================================================
    // COMPUTATION
    // ================================================================
    int n_threads = 1;
};

} // namespace tulpa

#endif // TULPA_MODEL_DATA_H
