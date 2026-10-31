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

// d/d(L, log_d) of H[q] = 0.5 log|LL' + D| + const, with D = diag(d^2).
// Via the matrix determinant lemma and Woodbury, with M = I + L'D^-1 L:
//
//   Sigma^-1 L                    = D^-1 L M^-1
//   d(0.5 log|Sigma|)/dL          = Sigma^-1 L
//   d(0.5 log|Sigma|)/d log_d_i   = 1 - (L M^-1 L')_ii / d_i^2
//
// Added to the reparameterisation gradient here and evaluated on its own by the
// gradient probe, so the entropy term the ELBO carries and the entropy term its
// gradient carries are the same expression.
inline void lowrank_add_entropy_grad(const LowRankParams& params,
                                     Eigen::MatrixXd& grad_L,
                                     Eigen::VectorXd& grad_log_d) {
  const int D = params.dim();
  const int r = params.rank();
  Eigen::VectorXd d2 = exp(2.0 * params.log_d.array());

  // I + L'D⁻¹L (r × r)
  Eigen::MatrixXd LtDinvL = Eigen::MatrixXd::Identity(r, r);
  for (int i = 0; i < D; ++i) {
    LtDinvL.noalias() += params.L.row(i).transpose() * params.L.row(i) / d2(i);
  }
  Eigen::LLT<Eigen::MatrixXd> llt(LtDinvL);
  Eigen::MatrixXd LtDinvL_inv = llt.solve(Eigen::MatrixXd::Identity(r, r));

  Eigen::MatrixXd L_Minv = params.L * LtDinvL_inv;
  Eigen::VectorXd L_Minv_Lt_diag(D);
  for (int i = 0; i < D; ++i) {
    L_Minv_Lt_diag(i) = L_Minv.row(i).dot(params.L.row(i));
  }
  grad_log_d.array() += 1.0 - L_Minv_Lt_diag.array() / d2.array();

  Eigen::MatrixXd DinvL(D, r);
  for (int i = 0; i < D; ++i) {
    DinvL.row(i) = params.L.row(i) / d2(i);
  }
  grad_L += DinvL * LtDinvL_inv;
}

// Compute ELBO and gradients for low-rank VI
// Uses reparameterization: θ = μ + L η + d ⊙ ε
// where η ~ N(0, Iᵣ), ε ~ N(0, I_D)
LowRankGradients compute_lowrank_elbo_grad(
    const LowRankParams& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
);

// ---------------------------------------------------------------------
// Low-Rank VI Fitting
// ---------------------------------------------------------------------

VIResult fit_lowrank(
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
);

} // namespace vi
} // namespace tulpa

#endif // TULPA_VI_LOWRANK_H
