// hmc_nuts_softabs.h
// Fragment of hmc_nuts_sampler.cpp. Included from the umbrella
// translation unit inside namespace tulpa_hmc; do NOT add a
// namespace wrapper here; do not list this file in the package SRCS —
// it is not a standalone translation unit.
// SoftAbs per-trajectory metric (Riemannian-like divergence retry).
#ifndef TULPA_HMC_NUTS_SOFTABS_H
#define TULPA_HMC_NUTS_SOFTABS_H


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

#endif  // TULPA_HMC_NUTS_SOFTABS_H
