// hmc_gp_cg.h
// Conjugate Gradient based GP computations for large-scale NNGP
// Uses iterative solvers instead of direct Cholesky for neighbor systems
// Also supports GPU-accelerated batched Cholesky via CUDA
//
// TODO: Wire into tulpa sampler (ported from numdenom, not yet integrated)

#ifndef TULPA_HMC_GP_CG_H
#define TULPA_HMC_GP_CG_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "hmc_gp.h"
#include "linalg_fast.h"
// #include "gpu_backend.h"  // Disabled: causing crashes

namespace tulpa_gp {

// GPSolver, GPSolverConfig, and parse_gp_solver are defined in hmc_gp.h

// =============================================================================
// CG-based linear system solvers for small dense systems
// These are for the k x k neighbor covariance matrices
// =============================================================================

// Solve C * x = b using CG for small dense SPD matrix C
// C_flat: k x k matrix in row-major order
// b: right-hand side (length k)
// x: solution (length k, modified in place)
// Returns: number of iterations (negative if failed)
inline int dense_cg_solve(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol = 1e-6,
    int maxiter = 100
) {
  // Initialize x = 0
  std::fill(x, x + k, 0.0);

  // r = b - C*x = b (since x=0)
  std::vector<double> r(k), p(k), Cp(k);
  std::copy(b, b + k, r.begin());
  std::copy(r.begin(), r.end(), p.begin());

  double r_dot_r = tulpa_linalg::dot_product(r.data(), r.data(), k);
  double b_norm = std::sqrt(tulpa_linalg::dot_product(b, b, k));
  if (b_norm < 1e-14) b_norm = 1.0;

  for (int iter = 0; iter < maxiter; iter++) {
    // Cp = C * p (dense matrix-vector product)
    for (int i = 0; i < k; i++) {
      Cp[i] = 0.0;
      for (int j = 0; j < k; j++) {
        Cp[i] += C_flat[i * k + j] * p[j];
      }
    }

    double p_dot_Cp = tulpa_linalg::dot_product(p.data(), Cp.data(), k);
    if (std::abs(p_dot_Cp) < 1e-30) {
      return -(iter + 1);  // Breakdown
    }

    double alpha = r_dot_r / p_dot_Cp;

    // x = x + alpha * p
    // r = r - alpha * Cp
    for (int i = 0; i < k; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Cp[i];
    }

    double r_dot_r_new = tulpa_linalg::dot_product(r.data(), r.data(), k);
    double r_norm = std::sqrt(r_dot_r_new);

    if (r_norm / b_norm < tol) {
      return iter + 1;  // Converged
    }

    double beta = r_dot_r_new / r_dot_r;
    r_dot_r = r_dot_r_new;

    // p = r + beta * p
    for (int i = 0; i < k; i++) {
      p[i] = r[i] + beta * p[i];
    }
  }

  return -maxiter;  // Did not converge
}

// Solve C * x = b using PCG with diagonal preconditioner
// Preconditioner M = diag(C) (Jacobi)
inline int dense_pcg_solve(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol = 1e-6,
    int maxiter = 100
) {
  // Initialize x = 0
  std::fill(x, x + k, 0.0);

  // Extract diagonal for preconditioner
  std::vector<double> M_inv(k);
  for (int i = 0; i < k; i++) {
    double diag = C_flat[i * k + i];
    M_inv[i] = (std::abs(diag) > 1e-14) ? 1.0 / diag : 1.0;
  }

  // r = b, z = M^{-1} r, p = z
  std::vector<double> r(k), z(k), p(k), Cp(k);
  std::copy(b, b + k, r.begin());

  for (int i = 0; i < k; i++) {
    z[i] = M_inv[i] * r[i];
  }
  std::copy(z.begin(), z.end(), p.begin());

  double r_dot_z = tulpa_linalg::dot_product(r.data(), z.data(), k);
  double b_norm = std::sqrt(tulpa_linalg::dot_product(b, b, k));
  if (b_norm < 1e-14) b_norm = 1.0;

  for (int iter = 0; iter < maxiter; iter++) {
    // Cp = C * p
    for (int i = 0; i < k; i++) {
      Cp[i] = 0.0;
      for (int j = 0; j < k; j++) {
        Cp[i] += C_flat[i * k + j] * p[j];
      }
    }

    double p_dot_Cp = tulpa_linalg::dot_product(p.data(), Cp.data(), k);
    if (std::abs(p_dot_Cp) < 1e-30) {
      return -(iter + 1);
    }

    double alpha = r_dot_z / p_dot_Cp;

    for (int i = 0; i < k; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Cp[i];
    }

    double r_norm = std::sqrt(tulpa_linalg::dot_product(r.data(), r.data(), k));
    if (r_norm / b_norm < tol) {
      return iter + 1;
    }

    // z = M^{-1} r
    for (int i = 0; i < k; i++) {
      z[i] = M_inv[i] * r[i];
    }

    double r_dot_z_new = tulpa_linalg::dot_product(r.data(), z.data(), k);
    double beta = r_dot_z_new / r_dot_z;
    r_dot_z = r_dot_z_new;

    for (int i = 0; i < k; i++) {
      p[i] = z[i] + beta * p[i];
    }
  }

  return -maxiter;
}

// =============================================================================
// GPU-accelerated NNGP log-likelihood
// =============================================================================

// Compute NNGP log-likelihood using GPU batched Cholesky
// Groups observations by neighbor count for efficient batching
double gp_nngp_log_lik_gpu(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  // Bounds validation
  if ((int)gp_data.nn_order.size() < N) return -INFINITY;
  if ((int)gp_data.nn_idx.size() < N * nn) return -INFINITY;
  if ((int)gp_data.nn_dist.size() < N * nn) return -INFINITY;
  if ((int)gp_data.nn_neighbor_dist.size() < N * nn * nn) return -INFINITY;  // Critical: prevents segfault
  if ((int)w.size() < N) return -INFINITY;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return -INFINITY;
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Collect all observations that have max neighbors for batched GPU processing
  // Observations with fewer neighbors are processed on CPU
  std::vector<int> batch_obs_indices;
  std::vector<int> cpu_obs_indices;

  for (int i = 1; i < N; i++) {
    int n_nb = 0;
    for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

    if (n_nb == nn) {
      batch_obs_indices.push_back(i);
    } else {
      cpu_obs_indices.push_back(i);
    }
  }

  // GPU batch processing for full-neighbor observations
  if (!batch_obs_indices.empty()) {
    int batch_size = (int)batch_obs_indices.size();

    // Build all C matrices and c vectors for GPU batch
    std::vector<std::vector<double>> C_batch(batch_size);
    std::vector<std::vector<double>> c_batch(batch_size);
    std::vector<std::vector<int>> neighbor_indices(batch_size);

    for (int b = 0; b < batch_size; b++) {
      int i = batch_obs_indices[b];
      C_batch[b].resize(nn * nn);
      c_batch[b].resize(nn);
      neighbor_indices[b].resize(nn);

      // Build c_vec (covariances to neighbors)
      for (int j = 0; j < nn; j++) {
        double d = gp_data.nn_dist[i * nn + j];
        c_batch[b][j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
      }

      // Build C_flat and get neighbor indices
      for (int j1 = 0; j1 < nn; j1++) {
        int raw1 = gp_data.nn_idx[i * nn + j1];
        neighbor_indices[b][j1] = (raw1 > 0) ? gp_data.nn_order[raw1 - 1] : -1;

        for (int j2 = 0; j2 < nn; j2++) {
          if (j1 == j2) {
            C_batch[b][j1 * nn + j2] = sigma2 + 1e-8;  // Jitter
          } else {
            double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
            C_batch[b][j1 * nn + j2] = compute_cov(d12, sigma2, phi, gp_data.cov_type);
          }
        }
      }
    }

    // Try GPU batched Cholesky+solve
    std::vector<std::vector<double>> alpha_batch;
    bool gpu_success = tulpa_gpu::gpu_batched_cholesky_solve(C_batch, c_batch, alpha_batch, nn);

    if (gpu_success) {
      // Compute log-likelihood contributions from GPU results
      for (int b = 0; b < batch_size; b++) {
        int i = batch_obs_indices[b];
        int obs_idx = gp_data.nn_order[i];
        if (obs_idx < 0 || obs_idx >= N) continue;

        // Conditional mean: mu = alpha' * w_neighbors
        double cond_mean = 0.0;
        for (int j = 0; j < nn; j++) {
          int idx = neighbor_indices[b][j];
          if (idx >= 0 && idx < N) {
            cond_mean += alpha_batch[b][j] * w[idx];
          }
        }

        // Conditional variance: sigma2_cond = sigma2 - c' * alpha
        double c_alpha = 0.0;
        for (int j = 0; j < nn; j++) {
          c_alpha += c_batch[b][j] * alpha_batch[b][j];
        }
        double cond_var = std::max(sigma2 - c_alpha, 1e-10);

        double resid = w[obs_idx] - cond_mean;
        log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
                   0.5 * resid * resid / cond_var;
      }
    } else {
      // GPU failed - add to CPU queue
      for (int b = 0; b < batch_size; b++) {
        cpu_obs_indices.push_back(batch_obs_indices[b]);
      }
    }
  }

  // CPU processing for non-full-neighbor observations (or GPU fallback)
  std::vector<double> c_vec(nn), C_flat(nn * nn), alpha(nn);
  std::vector<int> neighbor_idx(nn);

  for (int i : cpu_obs_indices) {
    int obs_idx = gp_data.nn_order[i];
    if (obs_idx < 0 || obs_idx >= N) continue;

    int n_nb = 0;
    for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

    if (n_nb == 0) {
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build c_vec
    for (int j = 0; j < n_nb; j++) {
      double d = gp_data.nn_dist[i * nn + j];
      c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // Build C_flat
    bool ok = true;
    for (int j1 = 0; j1 < n_nb && ok; j1++) {
      int raw1 = gp_data.nn_idx[i * nn + j1];
      if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) {
        ok = false;
        break;
      }
      neighbor_idx[j1] = gp_data.nn_order[raw1 - 1];

      for (int j2 = 0; j2 < n_nb; j2++) {
        if (j1 == j2) {
          C_flat[j1 * n_nb + j2] = sigma2 + 1e-8;
        } else {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          C_flat[j1 * n_nb + j2] = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    if (!ok) {
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // CPU Cholesky solve
    Eigen::MatrixXd C_sub(n_nb, n_nb);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        C_sub(j1, j2) = C_flat[j1 * n_nb + j2];
      }
    }

    Eigen::VectorXd c_sub(n_nb);
    for (int j = 0; j < n_nb; j++) c_sub(j) = c_vec[j];

    Eigen::LLT<Eigen::MatrixXd> llt(C_sub);
    if (llt.info() != Eigen::Success) {
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    Eigen::VectorXd alpha_eigen = llt.solve(c_sub);

    // Conditional mean
    double cond_mean = 0.0;
    for (int j = 0; j < n_nb; j++) {
      int idx = neighbor_idx[j];
      if (idx >= 0 && idx < N) {
        cond_mean += alpha_eigen(j) * w[idx];
      }
    }

    // Conditional variance
    double c_alpha = c_sub.dot(alpha_eigen);
    double cond_var = std::max(sigma2 - c_alpha, 1e-10);

    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// =============================================================================
// NNGP log-likelihood with configurable solver
// =============================================================================

// Compute NNGP log-likelihood using specified solver
// This is the main entry point - dispatches to Cholesky, CG, or GPU based on config
double gp_nngp_log_lik_with_solver(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data
) {
  const GPSolverConfig& config = gp_data.solver_config;
  GPSolver effective = config.effective_solver();

  // For Cholesky, use the existing implementation
  if (effective == GPSolver::CHOLESKY) {
    return gp_nngp_log_lik(w, sigma2, phi, gp_data);
  }

  // For GPU, use the GPU-accelerated implementation
  // Currently disabled due to platform compatibility issues
  // if (effective == GPSolver::GPU) {
  //   return gp_nngp_log_lik_gpu(w, sigma2, phi, gp_data);
  // }

  // GPU currently falls back to PCG
  if (effective == GPSolver::GPU) {
    // Fall through to CG/PCG implementation below
  }

  // CG/PCG implementation
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  // Bounds validation
  if (gp_data.nn_order.size() < (size_t)N) return -INFINITY;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return -INFINITY;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return -INFINITY;
  if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return -INFINITY;  // Critical: prevents segfault
  if (w.size() < (size_t)N) return -INFINITY;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return -INFINITY;

  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Preallocate work arrays
  std::vector<double> c_vec(nn), C_flat(nn * nn), alpha(nn), w_neighbors(nn);
  std::vector<int> neighbor_idx(nn);

  // Process remaining observations
  for (int i = 1; i < N; i++) {
    int obs_idx = gp_data.nn_order[i];
    if (obs_idx < 0 || obs_idx >= N) return -INFINITY;

    // Count neighbors
    int n_nb = 0;
    for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) {
      n_nb++;
    }

    if (n_nb == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build c_vec (covariances to neighbors)
    for (int j = 0; j < n_nb; j++) {
      double d = gp_data.nn_dist[i * nn + j];
      c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // Build C_flat (neighbor covariance matrix) and get neighbor indices
    bool ok = true;
    for (int j1 = 0; j1 < n_nb && ok; j1++) {
      int raw1 = gp_data.nn_idx[i * nn + j1];
      if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) {
        ok = false;
        break;
      }
      neighbor_idx[j1] = gp_data.nn_order[raw1 - 1];

      for (int j2 = 0; j2 < n_nb; j2++) {
        if (j1 == j2) {
          C_flat[j1 * n_nb + j2] = sigma2 + 1e-8;  // Jitter
        } else {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          C_flat[j1 * n_nb + j2] = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    if (!ok) {
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Solve C * alpha = c using CG or PCG
    int cg_result;
    if (effective == GPSolver::PCG) {
      cg_result = dense_pcg_solve(C_flat.data(), n_nb, c_vec.data(), alpha.data(),
                                   config.cg_tol, config.cg_maxiter);
    } else {
      cg_result = dense_cg_solve(C_flat.data(), n_nb, c_vec.data(), alpha.data(),
                                  config.cg_tol, config.cg_maxiter);
    }

    // If CG failed, fall back to marginal (conservative)
    if (cg_result < 0) {
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Conditional mean: mu = alpha' * w_neighbors
    double cond_mean = 0.0;
    for (int j = 0; j < n_nb; j++) {
      int idx = neighbor_idx[j];
      if (idx >= 0 && idx < N) {
        cond_mean += alpha[j] * w[idx];
      }
    }

    // Conditional variance: sigma2_cond = sigma2 - c' * alpha
    double c_alpha = 0.0;
    for (int j = 0; j < n_nb; j++) {
      c_alpha += c_vec[j] * alpha[j];
    }
    double cond_var = std::max(sigma2 - c_alpha, 1e-10);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// =============================================================================
// Gradients with CG solver
// =============================================================================

// Analytical gradient w.r.t. w using CG solver for linear systems
void gp_nngp_gradient_w_cg(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w
) {
  const GPSolverConfig& config = gp_data.solver_config;
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grad_w.assign(N, 0.0);

  if ((int)gp_data.nn_order.size() < N) return;

  // First observation
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  grad_w[first_idx] = -w[first_idx] / sigma2;

  // Preallocate
  std::vector<double> c_vec(nn), C_flat(nn * nn), alpha(nn);
  std::vector<int> neighbor_idx(nn);

  GPSolver effective = config.effective_solver();

  for (int i = 1; i < N; i++) {
    int obs_idx = gp_data.nn_order[i];
    if (obs_idx < 0 || obs_idx >= N) continue;

    int n_nb = 0;
    for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

    if (n_nb == 0) {
      grad_w[obs_idx] += -w[obs_idx] / sigma2;
      continue;
    }

    // Build c_vec
    for (int j = 0; j < n_nb; j++) {
      double d = gp_data.nn_dist[i * nn + j];
      c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // Build C_flat and neighbor indices
    bool ok = true;
    for (int j1 = 0; j1 < n_nb && ok; j1++) {
      int raw1 = gp_data.nn_idx[i * nn + j1];
      if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) {
        ok = false;
        break;
      }
      neighbor_idx[j1] = gp_data.nn_order[raw1 - 1];

      for (int j2 = 0; j2 < n_nb; j2++) {
        if (j1 == j2) {
          C_flat[j1 * n_nb + j2] = sigma2 + 1e-8;
        } else {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          C_flat[j1 * n_nb + j2] = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    if (!ok) {
      grad_w[obs_idx] += -w[obs_idx] / sigma2;
      continue;
    }

    // Solve for alpha
    int cg_result;
    if (effective == GPSolver::PCG) {
      cg_result = dense_pcg_solve(C_flat.data(), n_nb, c_vec.data(), alpha.data(),
                                   config.cg_tol, config.cg_maxiter);
    } else {
      cg_result = dense_cg_solve(C_flat.data(), n_nb, c_vec.data(), alpha.data(),
                                  config.cg_tol, config.cg_maxiter);
    }

    if (cg_result < 0) {
      grad_w[obs_idx] += -w[obs_idx] / sigma2;
      continue;
    }

    // Compute conditional statistics
    double cond_mean = 0.0;
    for (int j = 0; j < n_nb; j++) {
      int idx = neighbor_idx[j];
      if (idx >= 0 && idx < N) {
        cond_mean += alpha[j] * w[idx];
      }
    }

    double c_alpha = 0.0;
    for (int j = 0; j < n_nb; j++) {
      c_alpha += c_vec[j] * alpha[j];
    }
    double cond_var = std::max(sigma2 - c_alpha, 1e-10);

    double resid = w[obs_idx] - cond_mean;

    // Gradient contributions
    grad_w[obs_idx] += -resid / cond_var;
    for (int j = 0; j < n_nb; j++) {
      int idx = neighbor_idx[j];
      if (idx >= 0 && idx < N) {
        grad_w[idx] += alpha[j] * resid / cond_var;
      }
    }
  }
}

}  // namespace tulpa_gp

#endif  // TULPA_HMC_GP_CG_H
