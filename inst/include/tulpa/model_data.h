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
#include "tulpa/spde_model_data.h"

namespace tulpa {

// ============================================================================
// ABI version — bump on ANY change to an exported struct layout or to the
// registered-callable surface (added / removed / re-signed callables).
// Downstream packages check this at first NUTS use via check_abi_version()
// (<tulpa/nuts_api.h>) and must be rebuilt after a bump. Record every bump
// in the changelog below so the layout <-> version invariant stays auditable.
//
// 20 -> 21: added `tulpa_register_tgmrf` registered C callable + the
// process-global TgmrfSpec registry that backs `tgmrf_cpp()` (P7). New
// callable; existing layouts unchanged.
//
// 22 -> 23: appended `inv_metric_out` + `final_position` to NUTSResult
// so callers can resume / warm-start a chain. NUTSResult
// layout grew (two trailing pointers); NUTSFn signature unchanged.
//
// 23 -> 24: added the `tulpa_run_nuts_chains` registered C callable
// (multi-chain OpenMP across-chain runner). New callable
// only; no struct layout change (NUTSResult / NUTSFn unchanged).
//
// 24 -> 25: collapsed the 15 spatio-temporal nested-Laplace callables
// (`tulpa_nested_laplace_st_<spatial>_<temporal>`) to 5 per-spatial entries
// (`tulpa_nested_laplace_st_{icar,car_proper,bym2,hsgp,nngp}`) that select the
// temporal kernel at runtime via a `temporal_type` argument. Registered
// callables renamed + signatures changed (added const char* temporal_type,
// nullable rho_temporal_grid, int cyclic); the 15 NestedLaplaceSt*Fn typedefs
// became 5 NestedLaplaceSt<Spatial>Fn. No core struct layout change.
//
// 25 -> 26: removed the 8 family-enum single-point Laplace callables
// (`tulpa_laplace_mode_{dense,spatial,dense_multi_re,bym2,gp,multiscale_gp,
// multiscale_temporal,rsr}`) + their LaplaceMode*Fn typedefs. Dead surface:
// every consumer routes single-point Laplace through the LikelihoodSpec path
// (`tulpa_laplace_spec_*`). LaplaceShimResult is retained (shared with the
// spec shims). No core struct layout change.
//
// 26 -> 27: collapsed the 3 single-block temporal nested-Laplace callables
// (`tulpa_nested_laplace_{rw1,rw2,ar1}`) to one `tulpa_nested_laplace_temporal`
// that selects the kernel at runtime via a `temporal_type` argument (same
// make_temporal_ops registry the ST entries use); the 3 NestedLaplace{Rw1,Rw2,
// Ar1}Fn typedefs became one NestedLaplaceTemporalFn. No core struct layout
// change.
//
// 27 -> 28: added `field_coef` (default 1.0) to `JointArm`
// (src/laplace_newton_joint.h) so each arm in a joint fit can carry a
// per-arm constant multiplier on its field amplitude. Replaces the binary
// donor / copy-arm dichotomy with a per-arm scalar coefficient, while the
// existing `copy = list(arm, alpha_grid)` API desugars cleanly via
// `responses[[X]]$field_coef = list(name = "alpha", grid = G)`.
//
// 28 -> 29: added the `tulpa_register_cell_coupling` registered C callable
// (signature `tulpa::RegisterCellCouplingFn` in <tulpa/cell_coupling.h>)
// + the process-global CellCouplingSpec registry that backs the new
// `tulpa_nested_laplace_joint(cell_coupling = "<name>")` argument
// (Change 2b, Layer A). New callable + new R-side
// argument; no struct layout change. The inner-Newton per-cell branch
// that actually drives a non-separable spec lands with Layer B.
//
// 29 -> 30: added `coupled` (bool) + `cell_obs_map` (IntegerVector) to
// `JointArm` (src/laplace_newton_joint.h) and the inner-Newton dense
// per-cell branch that dispatches coupled arms to the registered
// CellCouplingSpec's `evaluate_cell` (Change 2b,
// Layer B.1). Single-arm path; sparse twin + cross-arm Hessian land
// with B.2 alongside tulpaObs `OccuCoverLognormalCoupling`.
//
// 30 -> 31: added the `CurvatureMode` enum + `CellDerivs.curvature` field
// (<tulpa/cell_coupling.h>) backing `control$hessian = "fisher"`
// (complete-data Fisher step curvature for the joint coupled Newton).
// CellDerivs layout grew by one trailing field.
//
// 31 -> 32: joint nested-Laplace multi-copy / svc-areal coupling work
// (JointArm surface growth in the joint solvers); precautionary bump for
// consumers driving the coupling shims.
// 32 -> 33: n_spatial_components added to ModelData so the sampler ICAR/BYM2/ST
// rank normalizer counts connected components (S - k), matching the Laplace
// path, instead of assuming a single component (S - 1).
// 33 -> 34: added gp_phi_prior_U / gp_phi_prior_alpha and svc_phi_prior_U /
// svc_phi_prior_alpha -- the PC range-prior anchors for the GP and SVC NNGP
// paths, replacing the Uniform-behind-a-wall range prior. Also grew
// LikelihoodSpec by extra_prior_arena (see likelihood.h).
// 34 -> 35: added car_adj_eigenvalues -- the eigenvalues of the symmetric
// normalized adjacency D^{-1/2} W D^{-1/2}, precomputed so the generic-NUTS
// CAR_proper log-prior evaluates the differentiable log-determinant
// log|D - rho W| = const + sum_i log(1 - rho * mu_i) in closed form rather
// than a per-gradient Cholesky (c).
// ============================================================================
constexpr int TULPA_ABI_VERSION = 35;

// ============================================================================
// Per-process design matrix and fixed effects (generic multi-process interface)
//
// LAYOUT RULE: append-only. Existing fields (X_flat, p) must keep their
// relative order so model packages compiled against an earlier ABI continue
// to bind correctly. New optional fields go at the end and default to safe
// no-ops (empty / zero).
// ============================================================================
struct ProcessData {
    std::vector<double> X_flat;  // Design matrix [N x p], row-major
    int p = 0;                    // Number of fixed effect columns

    // Optional per-process additive offset on the linear predictor.
    // When non-empty, must have length == ModelData::N. Added directly to
    // eta_k for this process before the likelihood sees it. Empty vector
    // means "no offset" (treated as zeros).
    std::vector<double> offset;
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
//   - Generic multi-process fields (n_processes, processes,
//     model_response_data, likelihood_spec, sharing) are STABLE.
//     Model packages depend on them.
//   - Shared infrastructure fields (spatial, temporal, RE, GP, etc.)
//     are STABLE.
//   - n_processes > 0 is the ONLY supported configuration. Ratio models
//     route through tulpaRatio's LikelihoodSpec.
//
// LAYOUT RULE: New fields go in the appropriate stable section.
//   Never insert fields before existing ones — append within sections.
//   Bump TULPA_ABI_VERSION when any field is added, removed, or reordered.
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
    // ================================================================
    int n_processes = 0;  // Must be > 0 after Phase D.
    std::vector<ProcessData> processes;
    void* model_response_data = nullptr;  // Opaque — owned by model package
    const void* likelihood_spec = nullptr; // Points to LikelihoodSpec (owned by caller)
    SharingSpec sharing;

    // ================================================================
    // DIMENSIONS
    // ================================================================
    int N = 0;  // Number of observations

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
    // Per-term flag: does the block carry the implicit group intercept (coef 0)?
    // Empty (the common case) means "all terms have an intercept" — see
    // re_term_has_intercept(). When false, all re_n_coefs[t] coefficients are
    // slopes and there is no implicit z=1 column (lme4 `(0 + x | g)`).
    std::vector<int> re_has_intercept;

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
    // Connected components of the adjacency graph. The intrinsic (ICAR) rank is
    // n_spatial_units - n_spatial_components; the sampler rank normalizer uses
    // it so a disconnected graph does not bias tau. Computed at data-load;
    // defaults to 1 (single component) when unset.
    int n_spatial_components = 1;

    // Proper CAR rho bounds (eigenvalue-derived, default to (0, 1))
    // Only used when spatial_type == CAR_PROPER.
    double car_rho_lower = 0.0;
    double car_rho_upper = 1.0;
    // Eigenvalues of the symmetric normalized adjacency D^{-1/2} W D^{-1/2}
    // (same spectrum as D^{-1} W). The generic-NUTS CAR_proper log-prior uses
    // them to evaluate the parameter-dependent part of the log-determinant,
    // sum_i log(1 - rho * mu_i), in autodiff-friendly closed form. Empty unless
    // spatial_type == CAR_PROPER on the sampler path.
    std::vector<double> car_adj_eigenvalues;

    // Precision mass matrix data (precomputed from Q)
    std::vector<double> spatial_Q_inv;  // (Q + lambda*I)^{-1}, column-major [S x S]
    std::vector<double> spatial_L_Q;    // Cholesky L of (Q + lambda*I), column-major [S x S]

    // GP spatial (single-scale NNGP)
    GPData gp_data;
    bool has_gp = false;
    double gp_sigma2_prior_U = 1.0;
    double gp_sigma2_prior_alpha = 0.01;
    // PC prior on the range: P(phi < gp_phi_prior_U) = gp_phi_prior_alpha.
    // phi is the range itself (every kernel is exp(-d / phi)), so the d = 2
    // density in pc_prior.h applies to it directly. -1.0 means unset; the R
    // layer requires the anchors rather than inventing them, matching the SPDE
    // field's prior_range contract.
    double gp_phi_prior_U = -1.0;
    double gp_phi_prior_alpha = -1.0;
    // 0 = centered (the spatial GP prior evaluates params directly as the field
    // w; the stored draws must match). 1 = non-centered would forward-transform
    // the stored draws as if params were z, but compute_gp_spatial_prior has no
    // z -> w branch, so 1 corrupts every stored GP field draw. Default centered
    // until a differentiable NC evaluator branch is restored.
    int gp_parameterization = 0;        // 0=centered, 1=non-centered

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

    // SPDE (Stochastic PDE Matern, Lindgren–Rue 2011). Active when
    // spatial_type == SpatialType::SPDE. Holds the FEM topology, per-obs
    // projection A, and the Q built at fixed (kappa, tau_spde). Joint NUTS
    // over (log_kappa, log_tau_spde) is deferred to a follow-on arc that
    // first extends arena AD with a sparse-Cholesky adjoint.
    SpdeModelData spde_data;
    bool has_spde = false;

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
    // PC prior on the range, NNGP path only; see gp_phi_prior_U. The HSGP path
    // reuses the same layout slot for an unbounded log-lengthscale under a
    // LogNormal(0, 1) prior and does not read these.
    double svc_phi_prior_U = -1.0;
    double svc_phi_prior_alpha = -1.0;

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
    bool temporal_shared = true;            // shared across processes
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

// Does RE term t carry the implicit group intercept (coef 0, z = 1)? Default
// true: callers that never set re_has_intercept (the overwhelming majority,
// since `(0 + x | g)` is rare) get the historical intercept-carrying block.
// When false, all re_n_coefs[t] coefficients are slopes read from the slope
// design matrix and there is no implicit intercept column.
inline bool re_term_has_intercept(const ModelData& data, int t) {
    if (t < 0 || t >= (int)data.re_has_intercept.size()) return true;
    return data.re_has_intercept[t] != 0;
}

} // namespace tulpa

#endif // TULPA_MODEL_DATA_H
