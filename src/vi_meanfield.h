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

  // Add entropy gradient: ∂H/∂log_σ = 1 (since H = Σ log_σ + const)
  result.grad_log_sigma.array() += 1.0;

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

    // Gradient norm for convergence check
    double grad_norm = grad_flat.norm();

    // Check convergence
    std::string converged = checker.check(grads.elbo, grad_norm);
    if (!converged.empty()) {
      result.converged = true;
      result.iterations = iter + 1;
      if (config.verbose) {
        Rcpp::Rcout << "Converged at iteration " << iter + 1
                    << " (" << converged << ")\n";
      }
      break;
    }

    // Adam update
    Eigen::VectorXd params_flat = params.flatten();
    params_flat = optimizer.step_clipped(params_flat, grad_flat, state);
    params.unflatten(params_flat);

    // Progress reporting
    if (config.verbose && (iter + 1) % config.print_every == 0) {
      Rcpp::Rcout << "Iter " << iter + 1 << ": ELBO = " << grads.elbo
                  << ", |grad| = " << grad_norm << "\n";
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }

    result.iterations = iter + 1;
  }

  // Store final parameters
  result.mu = params.mu;
  result.Sigma = params.covariance();
  result.L_factor = Eigen::MatrixXd();  // Not used for mean-field
  result.d_diag = exp(params.log_sigma.array());  // Store sigmas
  result.rank_used = 0;
  result.final_elbo = result.elbo_history.back();

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
