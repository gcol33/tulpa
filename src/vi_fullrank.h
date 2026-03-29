// vi_fullrank.h
// Full-rank Gaussian variational inference
// q(θ) = N(θ; μ, LL') with Cholesky parameterization

#ifndef TULPA_VI_FULLRANK_H
#define TULPA_VI_FULLRANK_H

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
// Full-Rank ELBO and Gradient
// ---------------------------------------------------------------------

struct FullRankGradients {
  Eigen::VectorXd grad_mu;
  Eigen::MatrixXd grad_L;
  double elbo;
};

// Compute ELBO and gradients for full-rank VI
// Uses reparameterization trick: θ = μ + L ε, ε ~ N(0, I)
inline FullRankGradients compute_fullrank_elbo_grad(
    const FullRankParams& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
) {
  int D = params.dim();
  FullRankGradients result;
  result.grad_mu = Eigen::VectorXd::Zero(D);
  result.grad_L = Eigen::MatrixXd::Zero(D, D);
  result.elbo = 0.0;

  std::normal_distribution<double> N01(0.0, 1.0);

  for (int m = 0; m < mc_samples; ++m) {
    // Sample ε ~ N(0, I)
    Eigen::VectorXd eps(D);
    for (int i = 0; i < D; ++i) {
      eps(i) = N01(rng);
    }

    // Reparameterize: θ = μ + L ε
    Eigen::VectorXd theta = params.mu + params.L * eps;

    // Gradient of log p(y, θ) w.r.t. θ
    Eigen::VectorXd grad_theta;
    double log_joint = compute_log_joint_grad(theta, data, layout, grad_theta);

    // Chain rule for μ: ∂L/∂μ = ∂log_p/∂θ
    result.grad_mu += grad_theta;

    // Chain rule for L: ∂θ/∂L[i,j] = ε[j] (only if i >= j for lower triangular)
    // So ∂L/∂L = grad_theta * ε'
    result.grad_L.noalias() += grad_theta * eps.transpose();

    result.elbo += log_joint;
  }

  // Average over MC samples
  result.grad_mu /= mc_samples;
  result.grad_L /= mc_samples;
  result.elbo /= mc_samples;

  // Zero out upper triangle (L is lower triangular)
  for (int i = 0; i < D; ++i) {
    for (int j = i + 1; j < D; ++j) {
      result.grad_L(i, j) = 0.0;
    }
  }

  // Entropy gradient: ∂H/∂L[i,i] = 1/L[i,i] (only diagonal)
  for (int i = 0; i < D; ++i) {
    if (std::abs(params.L(i, i)) > 1e-10) {
      result.grad_L(i, i) += 1.0 / params.L(i, i);
    }
  }

  // Add entropy to ELBO
  result.elbo += params.entropy();

  return result;
}

// ---------------------------------------------------------------------
// Full-Rank VI Fitting
// ---------------------------------------------------------------------

inline VIResult fit_fullrank(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr,
    const Eigen::MatrixXd* init_L = nullptr
) {
  // Warn if D is large
  if (D > 200 && config.verbose) {
    Rcpp::Rcout << "Warning: Full-rank VI with D=" << D
                << " may be slow (O(D^3) per iteration)\n";
  }

  // Initialize parameters
  FullRankParams params(D);

  if (init_mu != nullptr && init_mu->size() == D) {
    params.mu = *init_mu;
  }
  if (init_L != nullptr && init_L->rows() == D && init_L->cols() == D) {
    params.L = *init_L;
  }

  // Number of variational parameters
  int n_L = D * (D + 1) / 2;
  int n_var_params = D + n_L;

  // Initialize optimizer
  AdamOptimizer optimizer(config.adam_alpha, config.adam_beta1,
                          config.adam_beta2, config.adam_eps);
  AdamState state = optimizer.init_state(n_var_params);

  // Convergence checker
  ConvergenceChecker checker(config.tol_grad, config.tol_rel_elbo, config.patience);

  // Random number generator
  std::mt19937 rng(config.seed);

  // Optimization loop
  VIResult result;
  result.variant_used = VIVariant::FULLRANK;
  result.elbo_history.reserve(config.max_iter);

  for (int iter = 0; iter < config.max_iter; ++iter) {
    // Compute ELBO and gradients
    FullRankGradients grads = compute_fullrank_elbo_grad(
        params, data, layout, config.mc_samples, rng);

    result.elbo_history.push_back(grads.elbo);

    // Flatten gradients: [grad_mu, grad_L_lower_triangle]
    Eigen::VectorXd grad_flat(n_var_params);
    grad_flat.head(D) = grads.grad_mu;

    // Flatten lower triangle of grad_L (row by row)
    int idx = D;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j <= i; ++j) {
        grad_flat(idx++) = grads.grad_L(i, j);
      }
    }

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

    // Ensure positive diagonal in L (for valid Cholesky)
    for (int i = 0; i < D; ++i) {
      if (params.L(i, i) < 0.01) {
        params.L(i, i) = 0.01;
      }
    }

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
  result.L_factor = params.L;
  result.d_diag = Eigen::VectorXd();  // Not used for full-rank
  result.rank_used = D;  // Full rank
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

#endif // TULPA_VI_FULLRANK_H
