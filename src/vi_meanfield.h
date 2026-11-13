// vi_meanfield.h
// Mean-field Gaussian variational inference
// q(θ) = ∏ᵢ N(θᵢ; μᵢ, σᵢ²)

#ifndef TULPA_VI_MEANFIELD_H
#define TULPA_VI_MEANFIELD_H

#include "vi_types.h"
#include "vi_optimizer.h"
#include "hmc_sampler.h"
#include <random>
#include <cmath>

namespace tulpa {
namespace vi {

// Forward declaration - implemented in vi_sampler.cpp
double compute_log_joint_grad(
    const Eigen::VectorXd& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    Eigen::VectorXd& grad
);

// ---------------------------------------------------------------------
// Mean-Field ELBO and Gradient
// ---------------------------------------------------------------------

struct MeanFieldGradients {
  Eigen::VectorXd grad_mu;
  Eigen::VectorXd grad_log_sigma;
  double elbo;
};

// d/d log_sigma of H[q] = sum_i log_sigma_i + const. Added to the
// reparameterisation gradient here and evaluated on its own by the gradient
// probe, so the entropy term the ELBO carries and the entropy term its gradient
// carries are the same expression.
inline void meanfield_add_entropy_grad(const MeanFieldParams& /*params*/,
                                       Eigen::VectorXd& grad_log_sigma) {
  grad_log_sigma.array() += 1.0;
}

// Compute ELBO and gradients for mean-field VI
// Uses reparameterization trick: θ = μ + σ ⊙ ε, ε ~ N(0, I)
MeanFieldGradients compute_meanfield_elbo_grad(
    const MeanFieldParams& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
);

// ---------------------------------------------------------------------
// Mean-Field VI Fitting
// ---------------------------------------------------------------------

VIResult fit_meanfield(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
);

} // namespace vi
} // namespace tulpa

#endif // TULPA_VI_MEANFIELD_H
