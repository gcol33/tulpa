// Analytical gradient for simple Poisson-Gamma models
// O(n) instead of O(n*p) - huge speedup for typical models
// =====================================================================

bool can_use_analytical_gradient(const ModelData& data, const ParamLayout& layout) {
  // Hand-coded gradients for basic models without complex structure
  bool is_basic_family = (data.legacy.model_type == ModelType::POISSON_GAMMA ||
                          data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.legacy.model_type == ModelType::NEGBIN_GAMMA ||
                          data.legacy.model_type == ModelType::BINOMIAL ||
                          data.legacy.model_type == ModelType::GAMMA_GAMMA ||
                          data.legacy.model_type == ModelType::BETA_BINOMIAL ||
                          data.legacy.model_type == ModelType::LOGNORMAL);

  // Check if spatial type is one we have hand-coded gradients for.
  // CAR_PROPER uses the same handcoded path as ICAR (with rho-aware Q),
  // so it qualifies. The CAR_PROPER-specific log|Q(rho)|/tr(Q^{-1}W) work
  // happens in the CAR_PROPER branch of the spatial-prior gradient block.
  bool spatial_is_icar_bym2 = (data.spatial_type == SpatialType::ICAR ||
                               data.spatial_type == SpatialType::BYM2 ||
                               data.spatial_type == SpatialType::CAR_PROPER);

  // Temporal is OK alone or combined with ICAR/BYM2 spatial (no spatiotemporal interaction)
  // Note: Temporal GP is excluded - use autodiff for that
  bool temporal_ok = !layout.has_temporal ||
                     (layout.has_temporal && !layout.is_temporal_gp &&
                      !layout.has_spatiotemporal &&
                      (!layout.has_spatial || spatial_is_icar_bym2));

  // Spatial is OK for ICAR/BYM2 (alone or combined with temporal)
  bool spatial_ok = !layout.has_spatial ||
                    (layout.has_spatial && spatial_is_icar_bym2 && !layout.has_spatiotemporal);

  // ZI is OK for basic models, including with ICAR/BYM2 spatial and temporal
  // Components are additive in eta; ZI modifies residuals independently
  bool zi_ok = !layout.has_zi ||
               (layout.has_zi &&
                (!layout.has_spatial || spatial_is_icar_bym2));

  // Random slopes: both correlated (|) and uncorrelated (||) are supported
  // Can combine with ICAR/BYM2 spatial, temporal, and ZI
  // Components are additive in eta; gradient scatter is independent
  bool slopes_ok = !layout.has_re_slopes ||
                   (layout.has_re_slopes &&
                    (!layout.has_spatial || spatial_is_icar_bym2));

  return (is_basic_family &&
          !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
          !layout.is_icar_collapsed && !layout.is_bym2_collapsed &&
          temporal_ok && spatial_ok && zi_ok && slopes_ok &&
          !layout.has_latent && !layout.has_spatiotemporal &&
          !layout.has_multiscale_temporal && !layout.has_tvc &&
          !layout.has_svc &&  // SVC has its own gradient function
          data.n_re_terms >= 0);  // All RE combos: single, crossed, slopes, crossed+slopes
}

// Forward declarations of shared gradient helpers (defined after main gradient function)
struct CommonGradParams;
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params, const ParamLayout& layout);
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom, double* grad);
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom, double* grad);
static inline void compute_obs_residuals(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double& dLL_deta_num, double& dLL_deta_denom);
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom, double* grad);
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom, double* grad);
static inline void accumulate_phi_likelihood_grad(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom, double* grad);
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad);
static inline void temporal_sum_to_zero_grad(
    const double* phi, int T, int base_idx, double lambda, double* grad);
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const double* phi_temporal, int T_len,
    const double* grad_temporal_lik, double* grad);
static inline double gp_pc_prior_grad_log_sigma2(
    double sigma2, double U, double alpha);
static inline std::pair<GPData, GPData> make_msgp_gp_views(
    const MultiscaleGPData& msgp);
static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* spatial_phi, double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2, double rho_bym2,
    const double* theta_bym2,
    const double* grad_spatial_lik, const double* grad_theta_lik,
    double* grad);

void compute_gradient_analytical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Fused log-posterior computation: accumulate observation log-likelihood
  // alongside gradients to avoid a separate O(N) pass.
  const bool compute_lp = (log_post_out != nullptr);
  double obs_log_lik = 0.0;

  // Extract parameters
  const double* beta_num = &params[layout.legacy.beta_num_start];
  const double* beta_denom = &params[layout.legacy.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0, tau_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, log_phi_num = 0.0;
  double phi_denom = 1.0, log_phi_denom = 0.0;
  if (layout.legacy.has_phi_num) {
    log_phi_num = params[layout.legacy.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.legacy.has_phi_denom) {
    log_phi_denom = params[layout.legacy.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
  }

#include "hmc_gradient_analytical_priors_basic.h"

#include "hmc_gradient_analytical_priors_misc.h"

#include "hmc_gradient_analytical_lik_vec.h"

#include "hmc_gradient_analytical_lik_scalar.h"

#include "hmc_gradient_analytical_post.h"

}
