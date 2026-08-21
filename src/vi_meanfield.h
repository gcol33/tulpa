// vi_meanfield.h
// Mean-field Gaussian variational inference
// q(θ) = ∏ᵢ N(θᵢ; μᵢ, σᵢ²)

#ifndef TULPA_VI_MEANFIELD_H
#define TULPA_VI_MEANFIELD_H

#include "vi_types.h"
#include "vi_optimizer.h"
#include "hmc_sampler.h"
#include "autodiff.h"
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
inline MeanFieldGradients compute_meanfield_elbo_grad(
    const MeanFieldParams& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
) {
  int D = params.dim();
  MeanFieldGradients result;
  result.grad_mu = Eigen::VectorXd::Zero(D);
  result.grad_log_sigma = Eigen::VectorXd::Zero(D);
  result.elbo = 0.0;

  std::normal_distribution<double> N01(0.0, 1.0);
  Eigen::VectorXd sigma = exp(params.log_sigma.array());

  for (int m = 0; m < mc_samples; ++m) {
    // Sample ε ~ N(0, I)
    Eigen::VectorXd eps(D);
    for (int i = 0; i < D; ++i) {
      eps(i) = N01(rng);
    }

    // Reparameterize: θ = μ + σ ⊙ ε
    Eigen::VectorXd theta = params.mu.array() + sigma.array() * eps.array();

    // Gradient of log p(y, θ) w.r.t. θ
    Eigen::VectorXd grad_theta;
    double log_joint = compute_log_joint_grad(theta, data, layout, grad_theta);

    // Chain rule for μ: ∂L/∂μ = ∂log_p/∂θ · ∂θ/∂μ = ∂log_p/∂θ
    result.grad_mu += grad_theta;

    // Chain rule for log_σ: ∂L/∂log_σᵢ = ∂log_p/∂θᵢ · ∂θᵢ/∂log_σᵢ
    //                                  = ∂log_p/∂θᵢ · εᵢ · σᵢ
    result.grad_log_sigma += (grad_theta.array() * eps.array() * sigma.array()).matrix();

    result.elbo += log_joint;
  }

  // Average over MC samples
  result.grad_mu /= mc_samples;
  result.grad_log_sigma /= mc_samples;
  result.elbo /= mc_samples;

  meanfield_add_entropy_grad(params, result.grad_log_sigma);

  // Add entropy to ELBO
  result.elbo += params.entropy();

  return result;
}

// ---------------------------------------------------------------------
// Mean-Field VI Fitting
// ---------------------------------------------------------------------

inline VIResult fit_meanfield(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  validate_vi_config(config);

  // Initialize parameters
  MeanFieldParams params(D);

  if (init_mu != nullptr && init_mu->size() == D) {
    params.mu = *init_mu;
  }

  // Initialize optimizer
  AdamOptimizer optimizer(config.adam_alpha, config.adam_beta1,
                          config.adam_beta2, config.adam_eps);
  AdamState state = optimizer.init_state(params.n_variational_params());

  // Convergence checker
  ConvergenceChecker checker(config.tol_grad, config.tol_rel_elbo, config.patience);

  // Random number generator
  std::mt19937 rng(config.seed);

  // Optimization loop
  VIResult result;
  result.variant_used = VIVariant::MEANFIELD;
  result.elbo_history.reserve(config.max_iter);

  for (int iter = 0; iter < config.max_iter; ++iter) {
    // Compute ELBO and gradients
    MeanFieldGradients grads = compute_meanfield_elbo_grad(
        params, data, layout, config.mc_samples, rng);

    result.elbo_history.push_back(grads.elbo);

    // Flatten gradients for optimizer
    Eigen::VectorXd grad_flat(2 * D);
    grad_flat.head(D) = grads.grad_mu;
    grad_flat.tail(D) = grads.grad_log_sigma;

    // Convergence test, Adam update and bookkeeping (see vi_adam_step).
    if (vi_adam_step(params, grad_flat, grads.elbo, iter, config, checker,
                     optimizer, state, result)) {
      break;
    }
  }

  // Store final parameters
  result.mu = params.mu;
  result.Sigma = params.covariance();
  result.L_factor = Eigen::MatrixXd();  // Not used for mean-field
  result.d_diag = exp(params.log_sigma.array());  // Store sigmas
  result.rank_used = 0;
  result.final_elbo = vi_final_elbo(result.elbo_history);

  // Generate posterior samples for diagnostics
  int n_samples = 1000;
  result.samples.resize(n_samples, D);
  for (int s = 0; s < n_samples; ++s) {
    result.samples.row(s) = params.sample(rng);
  }

  return result;
}

} // namespace vi
} // namespace tulpa

#endif // TULPA_VI_MEANFIELD_H
