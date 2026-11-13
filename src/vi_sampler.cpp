// vi_sampler.cpp
// Variational Inference dispatcher (mean-field / low-rank / full-rank), and
// the per-variant ELBO gradients and fitters declared in vi_meanfield.h,
// vi_lowrank.h and vi_fullrank.h.
// The legacy ratio Rcpp entry points (cpp_vi_fit, cpp_vi_get_n_params) were
// removed in Phase D of the tulpaRatio migration; VI is
// reached from downstream packages via the C-callable shim `tulpa_fit_vi`
// (tulpa_shims_vi_ess.h) and the generic-layout LikelihoodSpec path.

#include "hmc_sampler.h"
#include "vi_types.h"
#include "vi_optimizer.h"
#include "vi_meanfield.h"
#include "vi_fullrank.h"
#include "vi_lowrank.h"
#include <Rcpp.h>

using namespace Rcpp;
using namespace tulpa_hmc;

namespace tulpa {
namespace vi {

// =====================================================================
// Bridge function: compute log joint and gradient
// =====================================================================

// This function bridges VI with the existing HMC gradient infrastructure
double compute_log_joint_grad(
    const Eigen::VectorXd& params_eigen,
    const ModelData& data,
    const ParamLayout& layout,
    Eigen::VectorXd& grad_eigen
) {
  int D = params_eigen.size();

  // Map Eigen to std::vector (zero-copy where possible)
  std::vector<double> params(params_eigen.data(), params_eigen.data() + D);

  // Compute gradient (H-mode returns log_post as byproduct via pointer)
  std::vector<double> grad(D);
  double log_post = 0.0;
  compute_gradient(params, data, layout, grad, &log_post);

  // Map back to Eigen
  grad_eigen = Eigen::Map<Eigen::VectorXd>(grad.data(), D);

  return log_post;
}

// =====================================================================
// VI Dispatcher
// =====================================================================

VIResult fit_vi(
    const ModelData& data,
    const ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  // Select variant
  VIVariant variant = select_variant(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "VI variant: " << variant_to_string(variant)
                << " (D=" << D << ")\n";
  }

  // Dispatch to appropriate implementation
  switch (variant) {
    case VIVariant::MEANFIELD:
      return fit_meanfield(data, layout, D, config, init_mu);

    case VIVariant::LOWRANK:
      return fit_lowrank(data, layout, D, config, init_mu);

    case VIVariant::FULLRANK:
      return fit_fullrank(data, layout, D, config, init_mu, nullptr);

    default:
      Rcpp::stop("Unknown VI variant");
  }

  return VIResult();  // Never reached
}


MeanFieldGradients compute_meanfield_elbo_grad(
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

VIResult fit_meanfield(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu) {
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


LowRankGradients compute_lowrank_elbo_grad(
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

  lowrank_add_entropy_grad(params, result.grad_L, result.grad_log_d);

  // Add entropy to ELBO
  result.elbo += params.entropy();

  return result;
}

VIResult fit_lowrank(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu) {
  // Determine rank
  int r = select_rank(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "Low-rank VI with D=" << D << ", rank=" << r << "\n";
  }

  validate_vi_config(config);

  // Initialize parameters
  LowRankParams params(D, r, config.seed);

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

    // Convergence test, Adam update and bookkeeping (see vi_adam_step).
    if (vi_adam_step(params, grad_flat, grads.elbo, iter, config, checker,
                     optimizer, state, result)) {
      break;
    }
  }

  // Store final parameters
  result.mu = params.mu;
  result.Sigma = params.covariance();
  result.L_factor = params.L;
  result.d_diag = exp(params.log_d.array());
  result.rank_used = r;
  result.final_elbo = vi_final_elbo(result.elbo_history);

  // Generate posterior samples for diagnostics
  int n_samples = 1000;
  result.samples.resize(n_samples, D);
  for (int s = 0; s < n_samples; ++s) {
    result.samples.row(s) = params.sample(rng);
  }

  return result;
}


FullRankGradients compute_fullrank_elbo_grad(
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

  fullrank_add_entropy_grad(params, result.grad_L);

  // Add entropy to ELBO
  result.elbo += params.entropy();

  return result;
}

VIResult fit_fullrank(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu,
    const Eigen::MatrixXd* init_L) {
  // Warn if D is large
  if (D > 200 && config.verbose) {
    Rcpp::Rcout << "Warning: Full-rank VI with D=" << D
                << " may be slow (O(D^3) per iteration)\n";
  }

  validate_vi_config(config);

  // Initialize parameters
  FullRankParams params(D);

  if (init_mu != nullptr && init_mu->size() == D) {
    params.mu = *init_mu;
  }
  if (init_L != nullptr && init_L->rows() == D && init_L->cols() == D) {
    params.L = *init_L;
    params.clamp_diagonal();
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

    // Convergence test, Adam update and bookkeeping (see vi_adam_step). The
    // Adam step lands back in the parameterization through
    // FullRankParams::unflatten, which is where the diagonal of L is held
    // positive so the Cholesky stays valid.
    if (vi_adam_step(params, grad_flat, grads.elbo, iter, config, checker,
                     optimizer, state, result)) {
      break;
    }
  }

  // Store final parameters
  result.mu = params.mu;
  result.Sigma = params.covariance();
  result.L_factor = params.L;
  result.d_diag = Eigen::VectorXd();  // Not used for full-rank
  result.rank_used = D;  // Full rank
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
