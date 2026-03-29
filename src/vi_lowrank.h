// vi_lowrank.h
// Low-rank + diagonal Gaussian variational inference
// q(θ) = N(θ; μ, LL' + D) where L is D×r and D = diag(d²)

#ifndef TULPA_VI_LOWRANK_H
#define TULPA_VI_LOWRANK_H

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
// Low-Rank ELBO and Gradient
// ---------------------------------------------------------------------

struct LowRankGradients {
  Eigen::VectorXd grad_mu;
  Eigen::MatrixXd grad_L;
  Eigen::VectorXd grad_log_d;
  double elbo;
};

// Compute ELBO and gradients for low-rank VI
// Uses reparameterization: θ = μ + L η + d ⊙ ε
// where η ~ N(0, Iᵣ), ε ~ N(0, I_D)
inline LowRankGradients compute_lowrank_elbo_grad(
    const LowRankParams& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
) {
  int D = params.dim();
  int r = params.rank();
  LowRankGradients result;
  result.grad_mu = Eigen::VectorXd::Zero(D);
  result.grad_L = Eigen::MatrixXd::Zero(D, r);
  result.grad_log_d = Eigen::VectorXd::Zero(D);
  result.elbo = 0.0;

  std::normal_distribution<double> N01(0.0, 1.0);
  Eigen::VectorXd d = exp(params.log_d.array());

  for (int m = 0; m < mc_samples; ++m) {
    // Sample η ~ N(0, Iᵣ) for low-rank part
    Eigen::VectorXd eta(r);
    for (int i = 0; i < r; ++i) {
      eta(i) = N01(rng);
    }

    // Sample ε ~ N(0, I_D) for diagonal part
    Eigen::VectorXd eps(D);
    for (int i = 0; i < D; ++i) {
      eps(i) = N01(rng);
    }

    // Reparameterize: θ = μ + L η + d ⊙ ε
    Eigen::VectorXd theta = params.mu + params.L * eta + d.asDiagonal() * eps;

    // Gradient of log p(y, θ) w.r.t. θ
    Eigen::VectorXd grad_theta;
    double log_joint = compute_log_joint_grad(theta, data, layout, grad_theta);

    // Chain rule for μ: ∂L/∂μ = ∂log_p/∂θ
    result.grad_mu += grad_theta;

    // Chain rule for L: ∂θ/∂L = η' (outer product with grad_theta)
    result.grad_L.noalias() += grad_theta * eta.transpose();

    // Chain rule for log_d: ∂θ/∂log_dᵢ = εᵢ · dᵢ
    result.grad_log_d += (grad_theta.array() * eps.array() * d.array()).matrix();

    result.elbo += log_joint;
  }

  // Average over MC samples
  result.grad_mu /= mc_samples;
  result.grad_L /= mc_samples;
  result.grad_log_d /= mc_samples;
  result.elbo /= mc_samples;

  // Entropy gradient computation via matrix determinant lemma
  // Σ = LL' + D where D = diag(d²)
  // ∂H/∂log_d and ∂H/∂L require derivatives of log|Σ|

  Eigen::VectorXd d2 = d.array().square();

  // Compute I + L'D⁻¹L (r × r matrix)
  Eigen::MatrixXd LtDinvL = Eigen::MatrixXd::Identity(r, r);
  for (int i = 0; i < D; ++i) {
    LtDinvL.noalias() += params.L.row(i).transpose() * params.L.row(i) / d2(i);
  }

  // Cholesky of I + L'D⁻¹L for solving
  Eigen::LLT<Eigen::MatrixXd> llt(LtDinvL);

  // Entropy gradient for log_d:
  // ∂(0.5 log|Σ|)/∂log_d = 1 - diag(L (I + L'D⁻¹L)⁻¹ L') / d²
  Eigen::MatrixXd LtDinvL_inv = llt.solve(Eigen::MatrixXd::Identity(r, r));
  Eigen::MatrixXd L_Minv = params.L * LtDinvL_inv;
  Eigen::VectorXd L_Minv_Lt_diag(D);
  for (int i = 0; i < D; ++i) {
    L_Minv_Lt_diag(i) = L_Minv.row(i).dot(params.L.row(i));
  }
  result.grad_log_d.array() += 1.0 - L_Minv_Lt_diag.array() / d2.array();

  // Entropy gradient for L:
  // ∂(0.5 log|Σ|)/∂L = D⁻¹ L (I + L'D⁻¹L)⁻¹
  Eigen::MatrixXd DinvL(D, r);
  for (int i = 0; i < D; ++i) {
    DinvL.row(i) = params.L.row(i) / d2(i);
  }
  result.grad_L += DinvL * LtDinvL_inv;

  // Add entropy to ELBO
  result.elbo += params.entropy();

  return result;
}

// ---------------------------------------------------------------------
// Low-Rank VI Fitting
// ---------------------------------------------------------------------

inline VIResult fit_lowrank(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  // Determine rank
  int r = select_rank(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "Low-rank VI with D=" << D << ", rank=" << r << "\n";
  }

  // Initialize parameters
  LowRankParams params(D, r);

  if (init_mu != nullptr && init_mu->size() == D) {
    params.mu = *init_mu;
  }

  // Number of variational parameters: D (mu) + D*r (L) + D (log_d)
  int n_var_params = D + D * r + D;

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
  result.variant_used = VIVariant::LOWRANK;
  result.elbo_history.reserve(config.max_iter);

  for (int iter = 0; iter < config.max_iter; ++iter) {
    // Compute ELBO and gradients
    LowRankGradients grads = compute_lowrank_elbo_grad(
        params, data, layout, config.mc_samples, rng);

    result.elbo_history.push_back(grads.elbo);

    // Flatten gradients: [grad_mu, grad_L (column-major), grad_log_d]
    Eigen::VectorXd grad_flat(n_var_params);
    grad_flat.head(D) = grads.grad_mu;

    // Flatten L column by column
    Eigen::Map<Eigen::VectorXd>(grad_flat.data() + D, D * r) =
        Eigen::Map<const Eigen::VectorXd>(grads.grad_L.data(), D * r);

    grad_flat.tail(D) = grads.grad_log_d;

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
  result.L_factor = params.L;
  result.d_diag = exp(params.log_d.array());
  result.rank_used = r;
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

#endif // TULPA_VI_LOWRANK_H
