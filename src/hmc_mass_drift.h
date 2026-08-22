// hmc_mass_drift.h
// The fused drift q += coeff * M^-1 p, for the zero-allocation NUTS loop.
//
// Split out of hmc_nuts_optimized.cpp so a test can drive the fused kernel
// itself rather than a re-implementation of it: this and
// DenseMassMatrix::inv_mass_times_p are the two places the metric is applied
// to a momentum, and they have to agree on every metric the engine can build.
#ifndef TULPA_HMC_MASS_DRIFT_H
#define TULPA_HMC_MASS_DRIFT_H

#include <Eigen/Dense>

#include "hmc_sampler_mass_blocks.h"
#include "linalg_fast.h"

namespace tulpa_hmc {

// Drift: q += coeff * C * p, where C = M^{-1} carries the full mass structure
// (identity / block-diagonal / diagonal / dense). Factored out of the leapfrog
// step so every scheme's drift sub-steps reuse the same fused kernels. coeff is
// the scheme's drift coefficient times the step size; for the default leapfrog
// it is exactly the step size.
inline void apply_drift(
    double coeff, double* q, const double* p,
    const DenseMassMatrix& mass, double* scratch, int n) {
  if (mass.has_lowrank()) {
    // Diagonal everywhere, then each low-rank term rewrites its own block --
    // the same composition DenseMassMatrix::inv_mass_times_p takes, fused into
    // the axpy so the block is the only part written twice.
    tulpa_linalg::axpy_weighted(coeff, mass.inv_mass_diag.data(), p, q, n);
    for (const auto& t : mass.lowrank) {
      t.apply_inv(p, scratch);
      for (int i = t.start; i < t.start + t.n; i++) {
        q[i] += coeff * (scratch[i] - mass.inv_mass_diag[i] * p[i]);
      }
    }
    return;
  }
  (void) scratch;
  if (!mass.adapted) {
    tulpa_linalg::axpy(coeff, p, q, n);
  } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
    tulpa_linalg::axpy_weighted(coeff, mass.inv_mass_diag.data(), p, q, n);
    for (const auto& blk : mass.blocks) {
      if (blk.adapted) {
        double tmp[4];
        blk.matvec(p, tmp);
        for (int i = 0; i < blk.size; i++) {
          q[blk.start + i] += coeff * (tmp[i] - mass.inv_mass_diag[blk.start + i] * p[blk.start + i]);
        }
      }
    }
  } else if (mass.type == MassMatrixType::DIAG) {
    tulpa_linalg::axpy_weighted(coeff, mass.inv_mass_diag.data(), p, q, n);
  } else {
    if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(mass.inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> qv(q, n);
      qv.noalias() += coeff * (Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      tulpa_linalg::axpy_matvec(coeff, mass.inv_mass_dense.data(), p, q, n);
    }
  }
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_MASS_DRIFT_H
