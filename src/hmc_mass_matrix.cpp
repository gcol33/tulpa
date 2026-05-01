// hmc_mass_matrix.cpp
// Dense mass matrix adaptation for the HMC/NUTS backend.

#include "hmc_sampler.h"
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tulpa_hmc {

bool DenseMassMatrix::update_from_covariance(const double* cov, int n_samples) {
  // Map the covariance data into an Eigen matrix (column-major)
  Eigen::Map<const Eigen::MatrixXd> C(cov, n, n);

  // Eigendecomposition for condition number control.
  // Without conditioning, ill-conditioned mass matrices force epsilon to be
  // tiny (driven by the stiffest direction), making sampling extremely slow.
  // E.g., HSGP+RW1 gets epsilon=3.2e-5 unconditioned vs ~0.01 conditioned.
  //
  // Clip eigenvalue ratio to MAX_COND so the step size ratio between the
  // loosest and stiffest directions is at most about 100:1.
  constexpr double MAX_COND = 1e4;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
  if (eig.info() != Eigen::Success) {
    // Eigendecomposition failed; degrade to diagonal.
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  Eigen::VectorXd evals = eig.eigenvalues();
  double lambda_max = evals.maxCoeff();
  double lambda_floor = std::max(lambda_max / MAX_COND, 1e-8);
  bool clipped = false;
  for (int i = 0; i < n; i++) {
    if (evals[i] < lambda_floor) {
      evals[i] = lambda_floor;
      clipped = true;
    }
  }

  // Reconstruct conditioned covariance: V * diag(lambda_clipped) * V^T.
  Eigen::MatrixXd C_cond;
  if (clipped) {
    const Eigen::MatrixXd& V = eig.eigenvectors();
    C_cond = V * evals.asDiagonal() * V.transpose();
  } else {
    C_cond = C;
  }

  // Cholesky of the conditioned covariance should succeed after clipping.
  Eigen::LLT<Eigen::MatrixXd> llt(C_cond);
  if (llt.info() != Eigen::Success) {
    // Should not happen after eigenvalue clipping, but handle gracefully
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  // Store conditioned covariance as inv_mass_dense
  std::memcpy(inv_mass_dense.data(), C_cond.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Store Cholesky factor L
  Eigen::MatrixXd L_mat = llt.matrixL();
  std::memcpy(L_inv_mass.data(), L_mat.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Also update diagonal for fallback and find_reasonable_epsilon compatibility
  for (int i = 0; i < n; i++) {
    double var_i = C_cond(i, i);
    inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
    sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
  }

  adapted = true;
  return true;
}

// =====================================================================
// Parameter layout computation
// =====================================================================

}
