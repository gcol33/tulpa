// hmc_mass_matrix.cpp
// Dense mass matrix adaptation for the HMC/NUTS backend.

#include "hmc_sampler.h"
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tulpa_hmc {

// Bounds on an adapted inverse-mass diagonal entry, i.e. on the posterior
// variance the sampler credits a coordinate with. A variance estimated from
// a short warmup window can come back near zero (a coordinate the chain has
// not moved on yet) or enormous (one it has just escaped a bad start along),
// and either drives the shared step size off the whole trajectory. The
// window is six orders of magnitude wide, so it binds only on those two
// pathologies and never on an estimate the window can support.
constexpr double INV_MASS_MIN = 1e-3;
constexpr double INV_MASS_MAX = 1e3;

// One clamp, read off the diagonal of a column-major n x n source.
void DenseMassMatrix::set_diag_from(const double* diag_source) {
  for (int i = 0; i < n; i++) {
    double var_i = diag_source[static_cast<size_t>(i) * n + i];
    inv_mass_diag[i] = std::max(INV_MASS_MIN, std::min(var_i, INV_MASS_MAX));
    sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
  }
}

// The state a dense adaptation that could not complete leaves behind: the
// clamped diagonal, and a metric that says it is diagonal.
void DenseMassMatrix::degrade_to_diag(const double* diag_source) {
  type = MassMatrixType::DIAG;
  adapted = true;
  set_diag_from(diag_source);
}

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


  // Absolute floor under the eigenvalue clip, for a spectrum whose largest
  // eigenvalue is itself tiny: lambda_max / MAX_COND is then below the point
  // where 1 / sqrt(lambda) is representable, so the relative clip alone would
  // not keep the reconstructed mass finite.
  constexpr double LAMBDA_FLOOR_ABS = 1e-8;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
  if (eig.info() != Eigen::Success) {
    // Eigendecomposition failed; degrade to diagonal.
    degrade_to_diag(cov);
    return false;
  }

  Eigen::VectorXd evals = eig.eigenvalues();
  double lambda_max = evals.maxCoeff();
  double lambda_floor = std::max(lambda_max / MAX_COND, LAMBDA_FLOOR_ABS);
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
    degrade_to_diag(cov);
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
  set_diag_from(C_cond.data());

  adapted = true;
  return true;
}

}
