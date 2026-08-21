// hmc_sampler_mass_blocks.h
// Fragment of hmc_sampler.h. Self-contained: defines symbols inside
// namespace tulpa_hmc.
// MassBlock (<=4x4) and the DenseMassMatrix container.
#ifndef TULPA_HMC_SAMPLER_MASS_BLOCKS_H
#define TULPA_HMC_SAMPLER_MASS_BLOCKS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <RcppEigen.h>

#include "hmc_sampler_decls.h"  // MassMatrixType (alias of tulpa::MassMatrixType)
#include "linalg_fast.h"        // tulpa_linalg::tri_solve_lower_transpose, etc.

namespace tulpa_hmc {

// =====================================================================
// Block-diagonal mass block (max 4×4, stack-allocated)
// =====================================================================

struct MassBlock {
  int start = 0;          // First param index in full parameter vector
  int size = 0;           // Block size (2-4)
  bool adapted = false;

  // Block mass storage (column-major, max 4×4)
  double inv_mass[16] = {};    // C_block (covariance block)
  double L_inv_mass[16] = {};  // Cholesky L where LL^T = C_block

  // Block-local Welford covariance accumulator
  int welford_n = 0;
  double welford_mean[4] = {};
  double welford_M2[16] = {};  // Running sum for covariance (column-major)

  void init(int s, int sz) {
    start = s;
    size = sz;
    adapted = false;
    std::memset(inv_mass, 0, sizeof(inv_mass));
    std::memset(L_inv_mass, 0, sizeof(L_inv_mass));
    // Initialize as identity
    for (int i = 0; i < sz; i++) {
      inv_mass[i * 4 + i] = 1.0;  // Using stride=4 (max block size)
      L_inv_mass[i * 4 + i] = 1.0;
    }
    reset_welford();
  }

  void reset_welford() {
    welford_n = 0;
    std::memset(welford_mean, 0, sizeof(welford_mean));
    std::memset(welford_M2, 0, sizeof(welford_M2));
  }

  // Extract block params and update Welford running stats
  void welford_update(const double* full_params) {
    welford_n++;
    double delta[4];
    for (int i = 0; i < size; i++) {
      delta[i] = full_params[start + i] - welford_mean[i];
      welford_mean[i] += delta[i] / welford_n;
    }
    for (int i = 0; i < size; i++) {
      double dx_new = full_params[start + i] - welford_mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        welford_M2[j * 4 + i] += val;  // stride=4
        if (i != j) {
          welford_M2[i * 4 + j] += val;
        }
      }
    }
  }

  // Compute covariance from Welford stats, Cholesky decompose, set adapted
  bool update_from_welford() {
    if (welford_n < 10) return false;

    // Compute sample covariance with small regularization
    double cov[16] = {};
    double scale = 1.0 / (welford_n - 1);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cov[i * 4 + j] = welford_M2[j * 4 + i] * scale;  // Note: M2 is col-major with stride 4
      }
      cov[i * 4 + i] += 1e-8;  // Regularization
    }

    // Try Cholesky decomposition
    double L[16] = {};
    if (!cholesky_small(cov, L, size)) return false;

    // Success: copy to block storage
    std::memcpy(inv_mass, cov, sizeof(inv_mass));
    std::memcpy(L_inv_mass, L, sizeof(L_inv_mass));
    adapted = true;
    return true;
  }

  // Tiny Cholesky for k<=4 (direct formula, no Eigen)
  // A and L use stride=4 (max block size)
  static bool cholesky_small(const double* A, double* L, int k) {
    std::memset(L, 0, 16 * sizeof(double));
    for (int i = 0; i < k; i++) {
      double sum = 0.0;
      for (int p = 0; p < i; p++) {
        sum += L[i * 4 + p] * L[i * 4 + p];
      }
      double diag = A[i * 4 + i] - sum;
      if (diag <= 0.0) return false;
      L[i * 4 + i] = std::sqrt(diag);
      for (int j = i + 1; j < k; j++) {
        double s = 0.0;
        for (int p = 0; p < i; p++) {
          s += L[j * 4 + p] * L[i * 4 + p];
        }
        L[j * 4 + i] = (A[j * 4 + i] - s) / L[i * 4 + i];
      }
    }
    return true;
  }

  // result[0..size-1] = C_block * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    for (int i = 0; i < size; i++) {
      double sum = 0.0;
      for (int j = 0; j < size; j++) {
        sum += inv_mass[i * 4 + j] * pb[j];
      }
      result[i] = sum;
    }
  }

  // p_block^T * C_block * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        result += pb[i] * inv_mass[i * 4 + j] * pb[j];
      }
    }
    return result;
  }

  // Sample momentum for block: p_block = L^{-T} z (back-substitution on tiny L)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!adapted) return;  // Non-adapted blocks use diagonal path
    std::normal_distribution<double> normal(0.0, 1.0);
    double z[4];
    for (int i = 0; i < size; i++) z[i] = normal(rng);

    // Back-substitution: solve L^T * p = z (upper triangular)
    double* pb = p_full + start;
    for (int i = size - 1; i >= 0; i--) {
      double sum = z[i];
      for (int j = i + 1; j < size; j++) {
        sum -= L_inv_mass[j * 4 + i] * pb[j];  // L^T[i][j] = L[j][i]
      }
      pb[i] = sum / L_inv_mass[i * 4 + i];
    }
  }
};

// =====================================================================
// Dense mass matrix for NUTS (encapsulates diag + dense state)
// =====================================================================

struct DenseMassMatrix {
  int n = 0;                              // Dimension
  MassMatrixType type = MassMatrixType::DIAG;
  bool adapted = false;

  // Diagonal (always available, used as fallback)
  std::vector<double> inv_mass_diag;      // M^{-1} diagonal = variance
  std::vector<double> sqrt_mass_diag;     // sqrt(M) diagonal = 1/sqrt(variance) for p sampling

  // Dense (only when type == DENSE)
  std::vector<double> inv_mass_dense;     // Full p×p M^{-1} = regularized sample covariance (column-major)
  std::vector<double> L_inv_mass;         // Cholesky factor L where LL^T = M^{-1} (column-major)

  // Scratch buffer for dense matvec results (avoids per-call allocation)
  std::vector<double> scratch;

  // Block-diagonal (only when type == BLOCK_DIAG)
  std::vector<MassBlock> blocks;
  std::vector<bool> in_block;  // in_block[i] = true if param i belongs to a block


  void init(int dim, MassMatrixType t) {
    n = dim;
    type = t;
    adapted = false;
    inv_mass_diag.assign(dim, 1.0);
    sqrt_mass_diag.assign(dim, 1.0);
    scratch.resize(dim);
    if (t == MassMatrixType::DENSE) {
      inv_mass_dense.assign(static_cast<size_t>(dim) * dim, 0.0);
      L_inv_mass.assign(static_cast<size_t>(dim) * dim, 0.0);
      // Initialize as identity
      for (int i = 0; i < dim; i++) {
        inv_mass_dense[static_cast<size_t>(i) * dim + i] = 1.0;
        L_inv_mass[static_cast<size_t>(i) * dim + i] = 1.0;
      }
    }
    if (t == MassMatrixType::BLOCK_DIAG) {
      in_block.assign(dim, false);
    }
  }

  // Initialize block-diagonal structure from block specifications
  // block_specs: vector of (start_index, block_size) pairs
  void init_block_diag(int dim, const std::vector<std::pair<int,int>>& block_specs) {
    init(dim, MassMatrixType::BLOCK_DIAG);
    blocks.clear();
    blocks.reserve(block_specs.size());
    for (const auto& spec : block_specs) {
      MassBlock blk;
      blk.init(spec.first, spec.second);
      blocks.push_back(blk);
      for (int i = spec.first; i < spec.first + spec.second; i++) {
        if (i < dim) in_block[i] = true;
      }
    }
  }

  // Update dense mass matrix from sample covariance
  // Returns true on success, false on Cholesky failure (degrades to diagonal)
  // Uses Eigen LLT for Cholesky decomposition
  bool update_from_covariance(const double* cov, int n_samples);

  // Sample momentum: p ~ N(0, M) where M = C^{-1}
  // DIAG: p[i] = z * sqrt_mass_diag[i]
  // BLOCK_DIAG: diagonal for non-block params, L^{-T} z for block params
  // DENSE: solve L^T * p = z  (back-substitution)
  // Uses Eigen triangular solve for dense case (n>=16) for SIMD acceleration.
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      // First: diagonal for all params
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
      // Then: overwrite block params with correlated samples
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          blk.sample_momentum(p, rng);
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
    } else {
      // Dense: p = L^{-T} z where LL^T = C (inv_mass)
      // We need p ~ N(0, C^{-1}), so sample z ~ N(0, I), then p = L^{-T} z
      std::vector<double> z(n);
      for (int i = 0; i < n; i++) {
        z[i] = normal(rng);
      }
      if (n >= 16) {
        Eigen::Map<const Eigen::MatrixXd> Lm(L_inv_mass.data(), n, n);
        Eigen::Map<const Eigen::VectorXd> zv(z.data(), n);
        Eigen::Map<Eigen::VectorXd> pv(p, n);
        // Solve L^T * p = z: transpose L then use upper-triangular solve
        pv.noalias() = Lm.transpose().triangularView<Eigen::Upper>().solve(zv);
      } else {
        tulpa_linalg::tri_solve_lower_transpose<
            tulpa_linalg::TriLayout::ColMajor>(L_inv_mass.data(), n, n,
                                               z.data(), p);
      }
    }
  }

  // Kinetic energy: 0.5 * p^T * C * p  where C = M^{-1}
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  double kinetic_energy(const double* p) const {
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_block[i]) {
          ke += inv_mass_diag[i] * p[i] * p[i];
        }
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          ke += blk.quadform(p);
        } else {
          for (int i = blk.start; i < blk.start + blk.size; i++) {
            ke += inv_mass_diag[i] * p[i] * p[i];
          }
        }
      }
      return 0.5 * ke;
    } else if (type == MassMatrixType::DIAG || !adapted) {
      return 0.5 * tulpa_linalg::weighted_norm_squared(p, inv_mass_diag.data(), n);
    } else if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      return 0.5 * pv.dot(Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      return 0.5 * tulpa_linalg::quadratic_form(p, inv_mass_dense.data(), n);
    }
  }

  // Compute C * p (for leapfrog position update: q += eps * C * p)
  // Result written to `result` buffer.
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  void inv_mass_times_p(const double* p, double* result) const {
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          double tmp[4];
          blk.matvec(p, tmp);
          for (int i = 0; i < blk.size; i++) {
            result[blk.start + i] = tmp[i];
          }
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
    } else if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> rv(result, n);
      rv.noalias() = Am.selfadjointView<Eigen::Lower>() * pv;
    } else {
      tulpa_linalg::symmatvec(inv_mass_dense.data(), p, result, n);
    }
  }

  // Compute diag(C) * p — uses diagonal only, even when dense is available.
  // Kept for backwards compatibility / debugging. The NUTS U-turn criterion
  // now uses inv_mass_times_p() for correct geometry with dense mass.
  void inv_mass_diag_times_p(const double* p, double* result) const {
    for (int i = 0; i < n; i++) {
      result[i] = inv_mass_diag[i] * p[i];
    }
  }

  // Set metric directly from precomputed G^{-1} and its Cholesky L.
  // Used by SoftAbs per-trajectory metric retry. No shrinkage applied.
  void set_from_metric(const std::vector<double>& g_inv,
                       const std::vector<double>& l_g_inv) {
    inv_mass_dense = g_inv;
    L_inv_mass = l_g_inv;
    for (int i = 0; i < n; i++) {
      inv_mass_diag[i] = g_inv[static_cast<size_t>(i) * n + i];
      sqrt_mass_diag[i] = 1.0 / std::sqrt(std::max(inv_mass_diag[i], 1e-10));
    }
    adapted = true;
  }

  // Set diagonal mass from WelfordStats output (same interface as before)
  // When type==DENSE, also populate the dense matrices as diagonal so that
  // the dense code paths (sample_momentum, kinetic_energy, inv_mass_times_p)
  // produce correct results even before full covariance is available.
  // When type==BLOCK_DIAG, diagonal is set normally; blocks are adapted separately
  // via their own Welford accumulators.
  void set_diagonal(const std::vector<double>& inv_m, const std::vector<double>& sqrt_m) {
    inv_mass_diag = inv_m;
    sqrt_mass_diag = sqrt_m;
    adapted = true;

    if (type == MassMatrixType::DENSE && !inv_mass_dense.empty()) {
      // Populate dense matrices as diagonal so dense code paths work correctly
      std::fill(inv_mass_dense.begin(), inv_mass_dense.end(), 0.0);
      std::fill(L_inv_mass.begin(), L_inv_mass.end(), 0.0);
      for (int i = 0; i < n; i++) {
        inv_mass_dense[static_cast<size_t>(i) * n + i] = inv_m[i];
        // L where LL^T = inv_mass (diagonal): L[i,i] = sqrt(inv_mass[i])
        L_inv_mass[static_cast<size_t>(i) * n + i] = std::sqrt(inv_m[i]);
      }
    }
  }
};

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_SAMPLER_MASS_BLOCKS_H
