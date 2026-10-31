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

// d/dL of H[q] = sum_i log L(i,i) + const, which is 1 / L(i,i) on the diagonal
// and zero elsewhere. FullRankParams::clamp_diagonal() holds L(i,i) at or above
// MIN_DIAG on every route into L, so this is the exact derivative of the
// entropy() the ELBO adds. Added to the reparameterisation gradient here and
// evaluated on its own by the gradient probe.
inline void fullrank_add_entropy_grad(const FullRankParams& params,
                                      Eigen::MatrixXd& grad_L) {
  const int D = params.dim();
  for (int i = 0; i < D; ++i) {
    grad_L(i, i) += 1.0 / params.L(i, i);
  }
}

// Compute ELBO and gradients for full-rank VI
// Uses reparameterization trick: θ = μ + L ε, ε ~ N(0, I)
FullRankGradients compute_fullrank_elbo_grad(
    const FullRankParams& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
);

// ---------------------------------------------------------------------
// Full-Rank VI Fitting
// ---------------------------------------------------------------------

VIResult fit_fullrank(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr,
    const Eigen::MatrixXd* init_L = nullptr
);

} // namespace vi
} // namespace tulpa

#endif // TULPA_VI_FULLRANK_H
