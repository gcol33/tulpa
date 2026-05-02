// =====================================================================
// Composite hand-coded gradient: handles ANY combination of features.
// This is the catch-all H-mode function for exotic multi-feature combos
// that no specialized gradient function covers (e.g., HSGP+TVC, SVC+RW1,
// latent+spatial, etc.). Slower than specialized functions but much faster
// than A_r/N fallback.
//
// Architecture: single observation loop with conditional feature blocks.
// Each feature contributes additively to eta; gradient scattering is
// independent per feature. Structural/prior gradients computed after the
// observation loop.
// =====================================================================

void compute_gradient_composite(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                              data.legacy.model_type == ModelType::BETA_BINOMIAL);


#include "hmc_gradient_composite_phase1.h"

#include "hmc_gradient_composite_phase2_priors.h"

#include "hmc_gradient_composite_phase3_loop.h"

#include "hmc_gradient_composite_phase4_post.h"

}
