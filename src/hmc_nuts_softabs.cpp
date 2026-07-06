// hmc_nuts_softabs.cpp
// SoftAbs per-trajectory metric (Riemannian-like divergence retry).

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <RcppEigen.h>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h
) {
  int p = static_cast<int>(params.size());
  hessian.resize(static_cast<size_t>(p) * p);

  // Base gradient
  std::vector<double> grad_base(p);
  compute_gradient(params, data, layout, grad_base);

  // Perturb each parameter and compute column of Hessian
  std::vector<double> params_pert = params;
  std::vector<double> grad_pert(p);
  for (int i = 0; i < p; i++) {
    double orig = params_pert[i];
    double hi = std::max(h, h * std::abs(orig));  // relative step for large params
    params_pert[i] = orig + hi;
    compute_gradient(params_pert, data, layout, grad_pert);
    for (int j = 0; j < p; j++) {
      hessian[static_cast<size_t>(i) * p + j] = (grad_pert[j] - grad_base[j]) / hi;
    }
    params_pert[i] = orig;
  }

  // Symmetrize: H = 0.5 * (H + H^T)
  for (int i = 0; i < p; i++) {
    for (int j = i + 1; j < p; j++) {
      double avg = 0.5 * (hessian[static_cast<size_t>(i) * p + j] +
                          hessian[static_cast<size_t>(j) * p + i]);
      hessian[static_cast<size_t>(i) * p + j] = avg;
      hessian[static_cast<size_t>(j) * p + i] = avg;
    }
  }
}

double compute_adaptive_nu_max(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    const DenseMassMatrix& mass,
    double epsilon
) {
  const int p = static_cast<int>(q.size());
  if (p <= 0 || !(epsilon > 0.0) || !std::isfinite(epsilon)) return epsilon;

  // The curvature estimate below builds a dense p x p Hessian (p + 1 gradient
  // sweeps, O(p^2) storage), affordable only in the same regime as the dense
  // mass matrix. Above that bound (spatial fields with thousands of latent
  // cells) return epsilon: nu_max = epsilon is the well-adapted operating point
  // (omega_max = 1, the floor below), so the scheme still resolves, without the
  // dense allocation.
  if (p > DENSE_MAX_PARAMS) return epsilon;

  // Local curvature C = -Hessian(log_post): symmetric, SPD near the mode.
  std::vector<double> H;
  compute_hessian_finite_diff(q, data, layout, H);
  if (static_cast<int>(H.size()) != p * p) return epsilon;

  // Power iteration on M^{-1} C. C and M^{-1} are both SPD, so their product is
  // self-adjoint in the M inner product: real positive spectrum, spectral
  // radius = lambda_max. Each step costs one C matvec (w = C x = -H x) and one
  // M^{-1} matvec (mass.inv_mass_times_p), and ||M^{-1} C x|| with ||x|| = 1
  // converges to lambda_max.
  std::vector<double> x(p), w(p), y(p);
  // Deterministic non-degenerate start (no RNG -> draws stay bit-identical).
  for (int i = 0; i < p; ++i) x[i] = 1.0 + 0.01 * static_cast<double>(i);
  double nrm = 0.0;
  for (double v : x) nrm += v * v;
  nrm = std::sqrt(nrm);
  if (!(nrm > 0.0)) return epsilon;
  for (double& v : x) v /= nrm;

  double lambda = 0.0;
  for (int it = 0; it < 100; ++it) {
    for (int i = 0; i < p; ++i) {
      double s = 0.0;
      const double* col = &H[static_cast<size_t>(i) * p];  // H symmetric
      for (int j = 0; j < p; ++j) s += col[j] * x[j];
      w[i] = -s;  // C x = -H x
    }
    mass.inv_mass_times_p(w.data(), y.data());
    double yn = 0.0;
    for (double v : y) yn += v * v;
    yn = std::sqrt(yn);
    if (!(yn > 0.0) || !std::isfinite(yn)) return epsilon;
    for (int i = 0; i < p; ++i) x[i] = y[i] / yn;
    if (it > 5 && std::abs(yn - lambda) <= 1e-6 * yn) { lambda = yn; break; }
    lambda = yn;
  }
  if (!(lambda > 0.0) || !std::isfinite(lambda)) return epsilon;

  double nu_max = std::sqrt(lambda) * epsilon;
  // Floor at epsilon (the well-adapted-metric value, omega_max = 1) and cap in
  // the band where the multistage schemes stay stable and the energy-error
  // objective is finite (leapfrog stability = 2; multistage a little beyond).
  if (!std::isfinite(nu_max) || nu_max < epsilon) nu_max = epsilon;
  if (nu_max > 3.0) nu_max = 3.0;
  return nu_max;
}

bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
) {
  // Map to Eigen (column-major)
  Eigen::Map<const Eigen::MatrixXd> H_map(neg_hessian.data(), p, p);

  // Eigendecomposition (symmetric)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(H_map);
  if (eigen.info() != Eigen::Success) return false;

  const auto& lambdas = eigen.eigenvalues();
  const auto& Q = eigen.eigenvectors();

  // Apply SoftAbs: f(?) = ? * coth(? * ?)
  // Properties: always positive, f(|?|>>0) ? |?|, f(0) ? 1/?
  Eigen::VectorXd softabs_inv_eig(p);
  for (int i = 0; i < p; i++) {
    double lam = lambdas(i);
    double al = alpha * lam;
    double f;
    if (std::abs(al) > 20.0) {
      f = std::abs(lam);
    } else if (std::abs(al) < 1e-10) {
      f = 1.0 / alpha;
    } else {
      f = lam * std::cosh(al) / std::sinh(al);
    }
    f = std::max(f, 1e-6);  // floor to ensure positive definiteness
    softabs_inv_eig(i) = 1.0 / f;
  }

  // Reconstruct G^{-1} = Q diag(1/f(?)) Q^T
  Eigen::MatrixXd G_inv_mat = Q * softabs_inv_eig.asDiagonal() * Q.transpose();

  // Cholesky of G^{-1}
  Eigen::LLT<Eigen::MatrixXd> llt(G_inv_mat);
  if (llt.info() != Eigen::Success) return false;
  Eigen::MatrixXd L_mat = llt.matrixL();

  // Copy to output (column-major)
  G_inv.resize(static_cast<size_t>(p) * p);
  L_G_inv.resize(static_cast<size_t>(p) * p);
  Eigen::Map<Eigen::MatrixXd>(G_inv.data(), p, p) = G_inv_mat;
  Eigen::Map<Eigen::MatrixXd>(L_G_inv.data(), p, p) = L_mat;

  return true;
}

}  // namespace tulpa_hmc
