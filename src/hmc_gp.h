// hmc_gp.h
// Gaussian Process spatial effects with NNGP approximation
// Supports single-scale GP and multi-scale (local + regional) GP

#ifndef TULPA_HMC_GP_H
#define TULPA_HMC_GP_H

#include <vector>
#include <cmath>
#include <random>
#include <RcppEigen.h>
#include "hmc_svc.h"  // Reuse covariance functions and NNGP infrastructure
#include "hmc_gp_cg.h"  // Iterative CG/PCG solvers (dense_cg_solve, dense_pcg_solve)
#include "tulpa/gp_data.h"
#include "tulpa/types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Verbose debug output (set to false for production)
#define GP_DEBUG_BOUNDS false

namespace tulpa_gp {

using tulpa::CovType;
using tulpa::GPData;
using tulpa::MultiscaleGPData;
using tulpa::MSGPSampler;
using tulpa::GPSolver;
using tulpa::GPSolverConfig;
using tulpa_svc::compute_cov;

// Parse sampler string to enum
inline MSGPSampler parse_msgp_sampler(const std::string& s) {
  if (s == "noncentered" || s == "auto") return MSGPSampler::NONCENTERED;
  if (s == "centered") return MSGPSampler::CENTERED;
  if (s == "interweaved") return MSGPSampler::INTERWEAVED;
  if (s == "adaptive") return MSGPSampler::ADAPTIVE;
  if (s == "riemannian") return MSGPSampler::RIEMANNIAN;
  if (s == "lbfgs") return MSGPSampler::LBFGS;
  return MSGPSampler::NONCENTERED;  // Default fallback
}

// =============================================================================
// L-BFGS MASS MATRIX ADAPTATION
// =============================================================================
//
// L-BFGS approximates the inverse Hessian using limited memory:
//   H_k ≈ (I - ρ_k s_k y_k^T) H_{k-1} (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
//
// where:
//   s_k = q_k - q_{k-1}     (position difference)
//   y_k = g_k - g_{k-1}     (gradient difference)
//   ρ_k = 1 / (y_k^T s_k)
//
// Storage: O(md) where m = memory size (typically 5-20), d = dimension
// Compute H*v: O(md) via two-loop recursion
// =============================================================================

struct LBFGSState {
    int m;                                    // Memory size (number of pairs to store)
    int d;                                    // Dimension
    int k;                                    // Current iteration count
    std::vector<std::vector<double>> s_list; // Position differences (circular buffer)
    std::vector<std::vector<double>> y_list; // Gradient differences (circular buffer)
    std::vector<double> rho_list;            // 1 / (y^T s) values
    double gamma;                            // Scaling factor for initial H_0

    LBFGSState() : m(0), d(0), k(0), gamma(1.0) {}

    LBFGSState(int memory_size, int dimension)
        : m(memory_size), d(dimension), k(0), gamma(1.0) {
        s_list.reserve(m);
        y_list.reserve(m);
        rho_list.reserve(m);
    }

    // Add a new (s, y) pair from position and gradient differences
    void add_pair(const std::vector<double>& s, const std::vector<double>& y) {
        double ys = 0.0;
        double yy = 0.0;
        for (int i = 0; i < d; i++) {
            ys += y[i] * s[i];
            yy += y[i] * y[i];
        }

        // Skip if curvature condition not satisfied (ensures positive definiteness)
        if (ys < 1e-10) return;

        double rho = 1.0 / ys;

        // Update scaling factor: gamma = (s^T y) / (y^T y)
        if (yy > 1e-10) {
            gamma = ys / yy;
        }

        // Add to circular buffer
        if ((int)s_list.size() < m) {
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(rho);
        } else {
            // Circular replacement
            int idx = k % m;
            s_list[idx] = s;
            y_list[idx] = y;
            rho_list[idx] = rho;
        }
        k++;
    }

    // Two-loop recursion: compute H_k * v in O(md) time
    void multiply_H(const std::vector<double>& v, std::vector<double>& result) const {
        if (d <= 0 || (int)v.size() != d) {
            result = v;
            return;
        }

        result.resize(d);
        for (int i = 0; i < d; i++) {
            result[i] = v[i];
        }

        int n_stored = std::min(k, (int)s_list.size());
        n_stored = std::min(n_stored, m);

        if (n_stored == 0) {
            for (int i = 0; i < d; i++) {
                result[i] *= gamma;
            }
            return;
        }

        std::vector<double> alpha(n_stored);

        // First loop: from newest to oldest
        for (int i = n_stored - 1; i >= 0; i--) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                dot += s_list[idx][j] * result[j];
            }
            alpha[i] = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                result[j] -= alpha[i] * y_list[idx][j];
            }
        }

        // Apply initial Hessian: r = gamma * q
        for (int i = 0; i < d; i++) {
            result[i] *= gamma;
        }

        // Second loop: from oldest to newest
        for (int i = 0; i < n_stored; i++) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                dot += y_list[idx][j] * result[j];
            }
            double beta = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                result[j] += (alpha[i] - beta) * s_list[idx][j];
            }
        }
    }

    // Kinetic energy: K = 0.5 * p^T * H * p
    double kinetic_energy(const std::vector<double>& p) const {
        if ((int)p.size() != d) return 0.0;
        std::vector<double> Hp;
        multiply_H(p, Hp);
        double ke = 0.0;
        for (int i = 0; i < d; i++) {
            ke += p[i] * Hp[i];
        }
        return 0.5 * ke;
    }

    // Get diagonal of B for momentum sampling: sqrt(1/gamma)
    std::vector<double> get_sqrt_B_diag() const {
        std::vector<double> result(d);
        double sqrt_inv_gamma = std::sqrt(1.0 / gamma);
        for (int i = 0; i < d; i++) {
            result[i] = sqrt_inv_gamma;
        }
        return result;
    }
};

// Parse solver string from R
inline GPSolver parse_gp_solver(const std::string& s) {
  if (s == "auto") return GPSolver::AUTO;
  if (s == "cholesky") return GPSolver::CHOLESKY;
  if (s == "cg") return GPSolver::CG;
  if (s == "pcg") return GPSolver::PCG;
  if (s == "gpu") return GPSolver::GPU;
  return GPSolver::AUTO;
}

// -----------------------------------------------------------------------------
// Neighbor-system solver dispatch
// -----------------------------------------------------------------------------
//
// Single source of truth for "solve C * alpha = c" inside the per-observation
// NNGP loop. Branches on `cfg.effective_solver()` and either
//   (a) does an Eigen LLT factorization in-place (Cholesky path), or
//   (b) calls dense_cg_solve / dense_pcg_solve from hmc_gp_cg.h.
//
// `llt` is reused as workspace by the Cholesky branch and ignored by CG.
// Returns true on success, false on failure (non-PSD or CG non-convergence).
//
// CG is an explicit user choice (`spatial_gp(solver = "cg")`); we do NOT
// silently fall back to Cholesky on CG failure — the caller treats failure
// the same way as a Cholesky non-PSD failure (typically: -INFINITY for
// log-lik, or zero contribution for gradients), so HMC will reject the step.
inline bool solve_neighbor_system(
    Eigen::MatrixXd& C_eigen, int n_nb,
    const Eigen::VectorXd& c_eigen,
    Eigen::VectorXd& alpha_out,
    Eigen::LLT<Eigen::MatrixXd>& llt,
    const GPSolverConfig& cfg
) {
  GPSolver effective = cfg.effective_solver();

  if (effective == GPSolver::CG || effective == GPSolver::PCG) {
    // CG path: copy the (top-left n_nb x n_nb) block of C into a row-major
    // scratch buffer for the solver.
    static thread_local std::vector<double> C_buf;
    static thread_local std::vector<double> b_buf;
    static thread_local std::vector<double> x_buf;
    C_buf.resize(n_nb * n_nb);
    b_buf.resize(n_nb);
    x_buf.resize(n_nb);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        C_buf[j1 * n_nb + j2] = C_eigen(j1, j2);
      }
      b_buf[j1] = c_eigen(j1);
    }
    int it = (effective == GPSolver::PCG)
      ? dense_pcg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                        cfg.cg_tol, cfg.cg_maxiter)
      : dense_cg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                       cfg.cg_tol, cfg.cg_maxiter);
    if (it < 0) return false;
    if (alpha_out.size() < n_nb) alpha_out.resize(n_nb);
    for (int j = 0; j < n_nb; j++) alpha_out(j) = x_buf[j];
    return true;
  }

  // Default / Cholesky path
  llt.compute(C_eigen.topLeftCorner(n_nb, n_nb));
  if (llt.info() != Eigen::Success) return false;
  if (alpha_out.size() < n_nb) alpha_out.resize(n_nb);
  alpha_out.head(n_nb) = llt.solve(c_eigen.head(n_nb));
  return true;
}

// Same as above but solves a SECOND system reusing the already-factored
// matrix when possible. For Cholesky, that's `llt.solve(rhs)`. For CG, we
// just call the iterative solver again — there is no factor to reuse.
inline bool solve_neighbor_system_second(
    const Eigen::MatrixXd& C_eigen, int n_nb,
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& out,
    const Eigen::LLT<Eigen::MatrixXd>& llt,
    const GPSolverConfig& cfg
) {
  GPSolver effective = cfg.effective_solver();

  if (effective == GPSolver::CG || effective == GPSolver::PCG) {
    static thread_local std::vector<double> C_buf;
    static thread_local std::vector<double> b_buf;
    static thread_local std::vector<double> x_buf;
    C_buf.resize(n_nb * n_nb);
    b_buf.resize(n_nb);
    x_buf.resize(n_nb);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        C_buf[j1 * n_nb + j2] = C_eigen(j1, j2);
      }
      b_buf[j1] = rhs(j1);
    }
    int it = (effective == GPSolver::PCG)
      ? dense_pcg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                        cfg.cg_tol, cfg.cg_maxiter)
      : dense_cg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                       cfg.cg_tol, cfg.cg_maxiter);
    if (it < 0) return false;
    if (out.size() < n_nb) out.resize(n_nb);
    for (int j = 0; j < n_nb; j++) out(j) = x_buf[j];
    return true;
  }

  if (out.size() < n_nb) out.resize(n_nb);
  out.head(n_nb) = llt.solve(rhs.head(n_nb));
  return true;
}

// -----------------------------------------------------------------------------
// Single-scale GP NNGP likelihood
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for single spatial field
// w: spatial effect values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
inline double gp_nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  // Bounds validation (always on - prevents UB from invalid data structures)
  if (gp_data.nn_order.size() < (size_t)N) return -INFINITY;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return -INFINITY;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return -INFINITY;  // Added: was missing
  if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return -INFINITY;  // Critical: prevents segfault
  if (w.size() < (size_t)N) return -INFINITY;
  if (gp_data.coords.size() < (size_t)(2 * N)) return -INFINITY;

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] gp_nngp_log_lik called: N=" << N << ", nn=" << nn << "\n";
  Rcpp::Rcout << "[GP_DEBUG] w.size()=" << w.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_order.size()=" << gp_data.nn_order.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_idx.size()=" << gp_data.nn_idx.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_dist.size()=" << gp_data.nn_dist.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] coords.size()=" << gp_data.coords.size() << "\n";

  // Validate sizes
  if (gp_data.nn_order.size() < (size_t)N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_order too small! size=" << gp_data.nn_order.size() << " < N=" << N << "\n";
    return -INFINITY;
  }
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_idx too small! size=" << gp_data.nn_idx.size() << " < N*nn=" << (N * nn) << "\n";
    return -INFINITY;
  }
  if (w.size() < (size_t)N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: w too small! size=" << w.size() << " < N=" << N << "\n";
    return -INFINITY;
  }
#endif

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] Accessing nn_order[0]...\n";
#endif
  int first_idx = gp_data.nn_order[0];

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] first_idx=" << first_idx << " (should be 0 to " << (N-1) << ")\n";
  if (first_idx < 0 || first_idx >= N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: first_idx out of bounds!\n";
    return -INFINITY;
  }
#endif

  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] First obs log_lik done, now processing remaining " << (N-1) << " observations\n";
#endif

  // Pre-allocate Eigen matrices/vectors for Cholesky/CG solve
  // Using Eigen avoids hand-rolled linear algebra bugs and leverages SIMD
  Eigen::VectorXd c_vec(nn);
  Eigen::MatrixXd C_mat(nn, nn);
  Eigen::VectorXd alpha(nn);
  Eigen::LLT<Eigen::MatrixXd> llt(nn);

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
#if GP_DEBUG_BOUNDS
    if (i < 5 || i == N-1) {
      Rcpp::Rcout << "[GP_DEBUG] Processing obs i=" << i << "\n";
    }
#endif

    int obs_idx = gp_data.nn_order[i];

    // Bounds check (always on)
    if (obs_idx < 0 || obs_idx >= N) return -INFINITY;

#if GP_DEBUG_BOUNDS
    if (obs_idx < 0 || obs_idx >= N) {
      Rcpp::Rcout << "[GP_DEBUG] ERROR: obs_idx=" << obs_idx << " out of bounds at i=" << i << "\n";
      return -INFINITY;
    }
#endif

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      // Bounds check (always on)
      if (nn_flat_idx < 0 || nn_flat_idx >= (int)gp_data.nn_idx.size()) return -INFINITY;
#if GP_DEBUG_BOUNDS
      if (nn_flat_idx < 0 || nn_flat_idx >= (int)gp_data.nn_idx.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_flat_idx=" << nn_flat_idx << " out of bounds (nn_idx.size=" << gp_data.nn_idx.size() << ")\n";
        return -INFINITY;
      }
#endif
      if (gp_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

#if GP_DEBUG_BOUNDS
    if (i < 5) {
      Rcpp::Rcout << "[GP_DEBUG]   n_neighbors=" << n_neighbors << "\n";
    }
#endif

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = gp_data.nn_dist[nn_flat_idx];
      c_vec(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int raw_nn_idx1 = gp_data.nn_idx[i * nn + j1];

      // Bounds check: nn_idx is 1-based from R, so subtract 1
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (i < 3 && j1 < 3) {
        Rcpp::Rcout << "[GP_DEBUG]   j1=" << j1 << " raw_nn_idx1=" << raw_nn_idx1 << "\n";
      }
      // nn_idx is 1-based from R, so subtract 1
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: raw_nn_idx1-1=" << (raw_nn_idx1 - 1) << " out of bounds for nn_order (size=" << gp_data.nn_order.size() << ")\n";
        return -INFINITY;
      }
#endif

      int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

      // Bounds check for coords access
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_idx1=" << nn_idx1 << " leads to coords out of bounds (coords.size=" << gp_data.coords.size() << ")\n";
        return -INFINITY;
      }
#endif

      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

        // Bounds check
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) {
          Rcpp::Rcout << "[GP_DEBUG] ERROR: raw_nn_idx2-1=" << (raw_nn_idx2 - 1) << " out of bounds for nn_order\n";
          return -INFINITY;
        }
#endif

        int nn_idx2 = gp_data.nn_order[raw_nn_idx2 - 1];

        if (j1 == j2) {
          C_mat(j1, j2) = sigma2;
        } else {
          // Phase 1.3: Use cached pairwise neighbor distances
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          C_mat(j1, j2) = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    // Solve C_mat * alpha = c_vec via the configured solver (Cholesky default,
    // CG/PCG opt-in via spatial_gp(solver = "cg"|"pcg")).
    // Add small jitter to diagonal for numerical stability — prevents
    // ill-conditioning when phi is very small or sigma2 is near zero.
    for (int j = 0; j < n_neighbors; j++) {
      C_mat(j, j) += 1e-8;
    }

    if (!solve_neighbor_system(C_mat, n_neighbors, c_vec, alpha, llt,
                               gp_data.solver_config)) {
      // Solver failed (non-PSD or CG non-convergence) — reject step.
      return -INFINITY;
    }

    // Conditional mean and variance
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int raw_nn_idx = gp_data.nn_idx[i * nn + j];

      // Bounds check
      if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: cond_mean raw_nn_idx-1=" << (raw_nn_idx - 1) << " out of bounds\n";
        return -INFINITY;
      }
#endif

      int nn_orig_idx = gp_data.nn_order[raw_nn_idx - 1];

      // Bounds check for w access
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_orig_idx=" << nn_orig_idx << " out of bounds for w (size=" << w.size() << ")\n";
        return -INFINITY;
      }
#endif

      cond_mean += alpha(j) * w[nn_orig_idx];
    }

    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec(j) * alpha(j);
    }
    double cond_var = std::max(1e-10, sigma2 - c_Cinv_c);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] gp_nngp_log_lik completed, log_lik=" << log_lik << "\n";
#endif

  return log_lik;
}

// -----------------------------------------------------------------------------
// Multi-scale GP likelihood
// -----------------------------------------------------------------------------

// Compute log-likelihood for multi-scale GP (local + regional)
// w_local: local-scale spatial effect (length n_obs)
// w_regional: regional-scale spatial effect (length n_obs)
// Each component evaluated independently with its own range constraint
inline double multiscale_gp_log_lik(
    const std::vector<double>& w_local,
    const std::vector<double>& w_regional,
    double sigma2_local,
    double phi_local,
    double sigma2_regional,
    double phi_regional,
    const MultiscaleGPData& ms_data
) {
  // Create temporary GPData structures for each scale
  GPData gp_local;
  gp_local.n_obs = ms_data.n_obs;
  gp_local.nn = ms_data.nn_local;
  gp_local.coords = ms_data.coords;
  gp_local.nn_idx = ms_data.nn_idx_local;
  gp_local.nn_dist = ms_data.nn_dist_local;
  gp_local.nn_neighbor_dist = ms_data.nn_neighbor_dist_local;
  gp_local.nn_order = ms_data.nn_order_local;
  gp_local.nn_order_inv = ms_data.nn_order_inv_local;
  gp_local.cov_type = ms_data.cov_type;

  GPData gp_regional;
  gp_regional.n_obs = ms_data.n_obs;
  gp_regional.nn = ms_data.nn_regional;
  gp_regional.coords = ms_data.coords;
  gp_regional.nn_idx = ms_data.nn_idx_regional;
  gp_regional.nn_dist = ms_data.nn_dist_regional;
  gp_regional.nn_neighbor_dist = ms_data.nn_neighbor_dist_regional;
  gp_regional.nn_order = ms_data.nn_order_regional;
  gp_regional.nn_order_inv = ms_data.nn_order_inv_regional;
  gp_regional.cov_type = ms_data.cov_type;

  // Compute log-likelihood for each scale
  double ll_local = gp_nngp_log_lik(w_local, sigma2_local, phi_local, gp_local);
  double ll_regional = gp_nngp_log_lik(w_regional, sigma2_regional, phi_regional, gp_regional);

  return ll_local + ll_regional;
}

// -----------------------------------------------------------------------------
// Priors for GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for spatial variance (PC prior style)
// P(sigma > U) = alpha => sigma ~ Exponential(rate = -log(alpha)/U)
inline double log_prior_sigma2_pc(double sigma2, double U, double alpha) {
  double rate = -std::log(alpha) / U;
  double sigma = std::sqrt(sigma2);
  // Exponential prior on sigma, transform to sigma2
  // p(sigma) = rate * exp(-rate * sigma)
  // Jacobian: d(sigma)/d(sigma2) = 1/(2*sigma)
  return std::log(rate) - rate * sigma - std::log(2.0 * sigma);
}

// Log prior for range parameter (uniform on log scale within bounds)
inline double log_prior_phi_uniform(double phi, double lower, double upper) {
  if (phi < lower || phi > upper) return -INFINITY;
  // Uniform on [lower, upper]
  return -std::log(upper - lower);
}

// Log prior for range with PC-style (favor larger ranges = simpler models)
inline double log_prior_phi_pc(double phi, double U, double alpha) {
  // P(phi < U) = alpha => 1/phi follows Exponential
  if (phi <= 0) return -INFINITY;
  double rate = -std::log(1.0 - alpha) / U;
  // Prior favors larger phi (simpler, smoother spatial structure)
  return std::log(rate) - rate / phi - 2.0 * std::log(phi);
}

// -----------------------------------------------------------------------------
// Gradient computation for GP parameters (for HMC)
// -----------------------------------------------------------------------------

// Struct to hold NNGP gradient results (for hand-coded gradients)
struct NNGPGradients {
  std::vector<double> grad_w;         // Gradient w.r.t. spatial effects
  double grad_log_sigma2;             // Gradient w.r.t. log(sigma2)
  double grad_log_phi;                // Gradient w.r.t. log(phi)
};

// Analytical gradient of NNGP log-likelihood w.r.t. w (spatial effects)
// Eigen LLT + OpenMP parallelized. Uses cached nn_neighbor_dist.
// Returns gradients w.r.t. w only; sigma2/phi gradients computed elsewhere.
inline void gp_nngp_gradient_w_analytical(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w  // Output: gradient (length n_obs)
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grad_w.assign(N, 0.0);

  // Validate input sizes
  if (gp_data.nn_order.size() < (size_t)N) return;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return;
  if (w.size() < (size_t)N) return;
  if (gp_data.coords.size() < (size_t)(2 * N)) return;
  if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  grad_w[first_idx] = -w[first_idx] / sigma2;

  // Thread-local workspace setup
  int n_threads = 1;
  #ifdef _OPENMP
  n_threads = std::max(1, std::min(omp_get_max_threads(), N - 1));
  #endif

  std::vector<double> tl_grad_w(n_threads * N, 0.0);

  struct ThreadWS {
    Eigen::MatrixXd C_eigen;
    Eigen::VectorXd c_eigen, w_nb_eigen;
    Eigen::LLT<Eigen::MatrixXd> llt;
    std::vector<int> nb_idx;
    ThreadWS(int nn_) : C_eigen(nn_, nn_), c_eigen(nn_),
                        w_nb_eigen(nn_), llt(nn_), nb_idx(nn_) {}
  };
  std::vector<ThreadWS> ws_vec(n_threads, ThreadWS(nn));

  #ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads)
  #endif
  {
    int tid = 0;
    #ifdef _OPENMP
    tid = omp_get_thread_num();
    #endif

    double* my_grad_w = &tl_grad_w[tid * N];
    auto& C_eigen = ws_vec[tid].C_eigen;
    auto& c_eigen = ws_vec[tid].c_eigen;
    auto& w_nb_eigen = ws_vec[tid].w_nb_eigen;
    auto& llt = ws_vec[tid].llt;
    auto& nb_idx = ws_vec[tid].nb_idx;

    #ifdef _OPENMP
    #pragma omp for schedule(dynamic)
    #endif
    for (int i = 1; i < N; i++) {
      int obs_idx = gp_data.nn_order[i];
      if (obs_idx < 0 || obs_idx >= N) continue;

      // Count actual neighbors
      int n_nb = 0;
      for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

      if (n_nb == 0) {
        my_grad_w[obs_idx] += -w[obs_idx] / sigma2;
        continue;
      }

      // Build c_vec
      for (int j = 0; j < n_nb; j++) {
        double d = gp_data.nn_dist[i * nn + j];
        c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
      }

      // Validate neighbor indices
      bool ok = true;
      for (int j = 0; j < n_nb && ok; j++) {
        int raw = gp_data.nn_idx[i * nn + j];
        if (raw - 1 < 0 || raw - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
        int idx = gp_data.nn_order[raw - 1];
        if (idx < 0 || idx >= N) { ok = false; break; }
        nb_idx[j] = idx;
      }
      if (!ok) {
        my_grad_w[obs_idx] += -w[obs_idx] / sigma2;
        continue;
      }

      // Build C_mat using cached distances (symmetric fill)
      for (int j1 = 0; j1 < n_nb; j1++) {
        C_eigen(j1, j1) = sigma2 + 1e-8;
        for (int j2 = j1 + 1; j2 < n_nb; j2++) {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
          C_eigen(j1, j2) = cov_val;
          C_eigen(j2, j1) = cov_val;
        }
      }

      // Configurable solver: Cholesky (default) or CG/PCG (opt-in).
      Eigen::VectorXd alpha_vec(n_nb);
      if (!solve_neighbor_system(C_eigen, n_nb, c_eigen, alpha_vec, llt,
                                 gp_data.solver_config)) {
        my_grad_w[obs_idx] += -w[obs_idx] / sigma2;
        continue;
      }

      // Conditional mean and variance
      for (int j = 0; j < n_nb; j++) w_nb_eigen(j) = w[nb_idx[j]];
      double cond_mean = alpha_vec.head(n_nb).dot(w_nb_eigen.head(n_nb));
      double c_Cinv_c = c_eigen.head(n_nb).dot(alpha_vec.head(n_nb));
      double cond_var = std::max(sigma2 - c_Cinv_c, 1e-10);
      double resid = w[obs_idx] - cond_mean;

      // Gradient w.r.t. w
      my_grad_w[obs_idx] += -resid / cond_var;
      double r_over_v = resid / cond_var;
      for (int j = 0; j < n_nb; j++) {
        my_grad_w[nb_idx[j]] += alpha_vec(j) * r_over_v;
      }
    }
  }

  // Reduce thread-local accumulators
  for (int t = 0; t < n_threads; t++) {
    const double* tg = &tl_grad_w[t * N];
    for (int k = 0; k < N; k++) grad_w[k] += tg[k];
  }
}

// Covariance derivative w.r.t. phi: dk(d)/dphi
inline double dcov_dphi(double d, double phi, double cov_val, tulpa_svc::CovType cov_type) {
  if (d < 1e-10) return 0.0;
  switch (cov_type) {
    case tulpa_svc::CovType::EXPONENTIAL:
      return cov_val * d / (phi * phi);
    case tulpa_svc::CovType::MATERN: {
      double u = 1.732050808 * d / phi;
      return (1.0 + u > 1e-10) ? cov_val * u * u / (phi * (1.0 + u)) : 0.0;
    }
    case tulpa_svc::CovType::GAUSSIAN:
      return cov_val * d * d / (phi * phi * phi);
    default:
      return cov_val * d / (phi * phi);
  }
}

// Fully analytical NNGP gradients — Eigen LLT + OpenMP parallelized
// Uses cached nn_neighbor_dist (no coord recomputation), symmetric C_mat fill
// Complexity: O(N * nn³) Cholesky-dominated, parallelized across observations
inline void gp_nngp_gradients(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    NNGPGradients& grads,
    double /* epsilon */ = 1e-6
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grads.grad_w.assign(N, 0.0);
  grads.grad_log_sigma2 = 0.0;
  grads.grad_log_phi = 0.0;

  // Validate
  if ((int)gp_data.nn_order.size() < N || (int)gp_data.nn_idx.size() < N * nn ||
      (int)gp_data.nn_dist.size() < N * nn || (int)w.size() < N ||
      (int)gp_data.coords.size() < 2 * N ||
      (int)gp_data.nn_neighbor_dist.size() < N * nn * nn) return;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  double w0 = w[first_idx];
  grads.grad_w[first_idx] = -w0 / sigma2;
  grads.grad_log_sigma2 += 0.5 * (w0 * w0 / sigma2 - 1.0);

  // Thread-local workspace setup
  int n_threads = 1;
  #ifdef _OPENMP
  n_threads = std::max(1, std::min(omp_get_max_threads(), N - 1));
  #endif

  // Per-thread accumulators: grad_w[tid * N + k], sigma2[tid], phi[tid]
  std::vector<double> tl_grad_w(n_threads * N, 0.0);
  std::vector<double> tl_sigma2(n_threads, 0.0);
  std::vector<double> tl_phi(n_threads, 0.0);

  // Per-thread Eigen workspaces (avoid per-iteration allocation)
  struct ThreadWS {
    Eigen::MatrixXd C_eigen;
    Eigen::VectorXd c_eigen, dc_eigen, w_nb_eigen;
    Eigen::LLT<Eigen::MatrixXd> llt;
    std::vector<int> nb_idx;
    ThreadWS(int nn_) : C_eigen(nn_, nn_), c_eigen(nn_), dc_eigen(nn_),
                        w_nb_eigen(nn_), llt(nn_), nb_idx(nn_) {}
  };
  std::vector<ThreadWS> ws_vec(n_threads, ThreadWS(nn));

  #ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads)
  #endif
  {
    int tid = 0;
    #ifdef _OPENMP
    tid = omp_get_thread_num();
    #endif

    double* my_grad_w = &tl_grad_w[tid * N];
    auto& C_eigen = ws_vec[tid].C_eigen;
    auto& c_eigen = ws_vec[tid].c_eigen;
    auto& dc_eigen = ws_vec[tid].dc_eigen;
    auto& w_nb_eigen = ws_vec[tid].w_nb_eigen;
    auto& llt = ws_vec[tid].llt;
    auto& nb_idx = ws_vec[tid].nb_idx;

    #ifdef _OPENMP
    #pragma omp for schedule(dynamic)
    #endif
    for (int i = 1; i < N; i++) {
      int obs_idx = gp_data.nn_order[i];
      if (obs_idx < 0 || obs_idx >= N) continue;

      // Count neighbors
      int n_nb = 0;
      for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

      if (n_nb == 0) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      // Build c_vec, dc_vec (covariances and phi derivatives)
      for (int j = 0; j < n_nb; j++) {
        double d = gp_data.nn_dist[i * nn + j];
        c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
        dc_eigen(j) = dcov_dphi(d, phi, c_eigen(j), gp_data.cov_type);
      }

      // Validate neighbor indices
      bool ok = true;
      for (int j = 0; j < n_nb && ok; j++) {
        int raw = gp_data.nn_idx[i * nn + j];
        if (raw - 1 < 0 || raw - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
        int idx = gp_data.nn_order[raw - 1];
        if (idx < 0 || idx >= N) { ok = false; break; }
        nb_idx[j] = idx;
      }
      if (!ok) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      // Build C_mat using cached nn_neighbor_dist (symmetric fill, upper triangle only)
      for (int j1 = 0; j1 < n_nb; j1++) {
        C_eigen(j1, j1) = sigma2 + 1e-8;  // Diagonal + jitter
        for (int j2 = j1 + 1; j2 < n_nb; j2++) {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
          C_eigen(j1, j2) = cov_val;
          C_eigen(j2, j1) = cov_val;
        }
      }

      // Configurable solver: factorize once (Cholesky) or run CG twice
      // (alpha = C^{-1}c, beta = C^{-1}w_nb).
      Eigen::VectorXd alpha_vec(n_nb);
      if (!solve_neighbor_system(C_eigen, n_nb, c_eigen, alpha_vec, llt,
                                 gp_data.solver_config)) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      for (int j = 0; j < n_nb; j++) w_nb_eigen(j) = w[nb_idx[j]];
      Eigen::VectorXd beta_vec(n_nb);
      if (!solve_neighbor_system_second(C_eigen, n_nb, w_nb_eigen, beta_vec,
                                        llt, gp_data.solver_config)) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      // Conditional mean and variance
      double mu = alpha_vec.head(n_nb).dot(w_nb_eigen.head(n_nb));
      double c_alpha = c_eigen.head(n_nb).dot(alpha_vec.head(n_nb));
      double v = std::max(sigma2 - c_alpha, 1e-10);
      double r = w[obs_idx] - mu;

      // Gradient w.r.t. w
      my_grad_w[obs_idx] += -r / v;
      double r_over_v = r / v;
      for (int j = 0; j < n_nb; j++) my_grad_w[nb_idx[j]] += alpha_vec(j) * r_over_v;

      // Gradient w.r.t. sigma2
      double dll_dv = 0.5 * (r * r / v - 1.0) / v;
      tl_sigma2[tid] += dll_dv * (1.0 - c_alpha / sigma2) * sigma2;

      // Gradient w.r.t. phi — cached distances, symmetric quadratic form
      double alpha_dc = alpha_vec.head(n_nb).dot(dc_eigen.head(n_nb));
      double dc_beta = dc_eigen.head(n_nb).dot(beta_vec);

      double alpha_dC_alpha = 0.0, alpha_dC_beta = 0.0;
      for (int j1 = 0; j1 < n_nb; j1++) {
        for (int j2 = j1 + 1; j2 < n_nb; j2++) {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          double dC_jk = dcov_dphi(d12, phi, C_eigen(j1, j2), gp_data.cov_type);
          // Symmetric: accumulate both (j1,j2) and (j2,j1)
          alpha_dC_alpha += 2.0 * alpha_vec(j1) * dC_jk * alpha_vec(j2);
          alpha_dC_beta += alpha_vec(j1) * dC_jk * beta_vec(j2) +
                           alpha_vec(j2) * dC_jk * beta_vec(j1);
        }
      }

      double dv_dphi = -2.0 * alpha_dc + alpha_dC_alpha;
      double dr_dphi = -dc_beta + alpha_dC_beta;
      tl_phi[tid] += (dll_dv * dv_dphi + (-r / v) * dr_dphi) * phi;
    }
  }

  // Reduce thread-local accumulators
  for (int t = 0; t < n_threads; t++) {
    const double* tg = &tl_grad_w[t * N];
    for (int k = 0; k < N; k++) grads.grad_w[k] += tg[k];
    grads.grad_log_sigma2 += tl_sigma2[t];
    grads.grad_log_phi += tl_phi[t];
  }
}

// Numerical gradient of NNGP log-likelihood w.r.t. w (spatial effects)
// For use in HMC updates
inline void gp_gradient_w(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w,  // Output: gradient (length n_obs)
    double epsilon = 1e-6
) {
  int N = gp_data.n_obs;
  grad_w.resize(N);

  double base_ll = gp_nngp_log_lik(w, sigma2, phi, gp_data);

  // Finite difference for each w[i]
  std::vector<double> w_plus = w;
  for (int i = 0; i < N; i++) {
    w_plus[i] = w[i] + epsilon;
    double ll_plus = gp_nngp_log_lik(w_plus, sigma2, phi, gp_data);
    grad_w[i] = (ll_plus - base_ll) / epsilon;
    w_plus[i] = w[i];  // Reset
  }
}

// =============================================================================
// Non-centered NNGP parameterization
// =============================================================================
// Instead of sampling w ~ NNGP(0, sigma2, phi) directly (centered),
// sample z ~ N(0, I) and transform z -> w via the NNGP autoregressive structure:
//   w[order[0]] = sqrt(sigma2) * z[0]
//   w[order[i]] = sum_j B[i,j] * w[nb_j(i)] + sqrt(d_i) * z[i]
// where B[i,:] = C_nb^{-1} c_i (regression coefficients) and
//       d_i = sigma2 - c_i' C_nb^{-1} c_i (conditional variance).
//
// This improves posterior geometry for large N, reducing NUTS treedepth.
// The prior on z is N(0,I), and no Jacobian is needed since we sample in z-space.

struct NNGPNCWorkspace {
    int N = 0, nn = 0;
    std::vector<double> w;          // Transformed spatial effects (N)
    std::vector<double> sqrt_d;     // sqrt(conditional variance) per obs (N)
    std::vector<double> B_flat;     // Regression coefficients (N * nn)
    std::vector<int> B_n_nb;        // Number of actual neighbors per obs (N)
    std::vector<int> nb_idx_flat;   // Neighbor indices per obs (N * nn), 0-based in w
    std::vector<double> adj;        // Adjoint accumulator (N)
    std::vector<double> L_flat;     // Cached Cholesky factors (N * nn * nn) for backward phi grad

    void init(int N_, int nn_) {
        if (N == N_ && nn == nn_) return;
        N = N_; nn = nn_;
        w.resize(N);
        sqrt_d.resize(N);
        B_flat.assign(N * nn, 0.0);
        B_n_nb.resize(N, 0);
        nb_idx_flat.assign(N * nn, -1);
        adj.resize(N, 0.0);
        L_flat.assign(N * nn * nn, 0.0);
    }
};

// Forward pass: z -> w via NNGP autoregressive structure
// z and w are both indexed by LOCATION (0-based), matching the parameter layout.
// Caches B, sqrt_d, nb_idx for backward pass.
// O(N * nn^3) due to per-observation Cholesky. Sequential (causal dependency).
// Uses Eigen LLT for vectorized Cholesky (~2x vs hand-rolled).
inline void nngp_nc_forward(
    const double* z,          // z[loc_idx], indexed by location, length N
    double sigma2, double phi,
    const GPData& gp_data,
    NNGPNCWorkspace& ws
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    ws.init(N, nn);

    // Pre-allocated Eigen workspace (reused across iterations)
    Eigen::MatrixXd C_eigen(nn, nn);
    Eigen::VectorXd c_eigen(nn);
    Eigen::LLT<Eigen::MatrixXd> llt(nn);

    // First observation: marginal N(0, sigma2)
    int first_loc = gp_data.nn_order[0];
    ws.sqrt_d[0] = std::sqrt(sigma2);
    ws.w[first_loc] = ws.sqrt_d[0] * z[first_loc];
    ws.B_n_nb[0] = 0;

    for (int i = 1; i < N; i++) {
        int obs_loc = gp_data.nn_order[i];
        if (obs_loc < 0 || obs_loc >= N) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Count neighbors
        int n_nb = 0;
        for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

        if (n_nb == 0) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build c_vec (covariance between obs and its neighbors)
        for (int j = 0; j < n_nb; j++) {
            double d = gp_data.nn_dist[i * nn + j];
            c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
        }

        // Validate neighbor indices and build C_mat (symmetric fill)
        bool ok = true;
        for (int j = 0; j < n_nb && ok; j++) {
            int raw = gp_data.nn_idx[i * nn + j];
            if (raw - 1 < 0 || raw - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
            int loc = gp_data.nn_order[raw - 1];
            if (loc < 0 || loc >= N) { ok = false; break; }
            ws.nb_idx_flat[i * nn + j] = loc;
        }
        if (!ok) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build C_mat using cached distances (symmetric, upper triangle only)
        for (int j1 = 0; j1 < n_nb; j1++) {
            C_eigen(j1, j1) = sigma2 + 1e-8;  // Jitter for stability
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
                C_eigen(j1, j2) = cov_val;
                C_eigen(j2, j1) = cov_val;
            }
        }

        // Eigen Cholesky: C = LL', alpha = C^{-1}c
        llt.compute(C_eigen.topLeftCorner(n_nb, n_nb));
        if (llt.info() != Eigen::Success) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        Eigen::VectorXd alpha_vec = llt.solve(c_eigen.head(n_nb));

        // Cache Cholesky factor L for backward phi gradient
        Eigen::MatrixXd L_mat = llt.matrixL();
        for (int j1 = 0; j1 < n_nb; j1++) {
            for (int j2 = 0; j2 <= j1; j2++) {
                ws.L_flat[i * nn * nn + j1 * nn + j2] = L_mat(j1, j2);
            }
        }

        // Store B and compute conditional variance d_i
        double c_alpha = 0.0;
        for (int j = 0; j < n_nb; j++) {
            ws.B_flat[i * nn + j] = alpha_vec(j);
            c_alpha += c_eigen(j) * alpha_vec(j);
        }
        ws.B_n_nb[i] = n_nb;

        double d_i = std::max(sigma2 - c_alpha, 1e-10);
        ws.sqrt_d[i] = std::sqrt(d_i);

        // Forward transform: w[loc] = B @ w_neighbors + sqrt(d_i) * z[loc]
        double mu = 0.0;
        for (int j = 0; j < n_nb; j++) {
            mu += alpha_vec(j) * ws.w[ws.nb_idx_flat[i * nn + j]];
        }
        ws.w[obs_loc] = mu + ws.sqrt_d[i] * z[obs_loc];
    }
}

// Backward pass: given dL/dw from likelihood, compute gradients for z, log_sigma2, log_phi.
// z and grad_z are indexed by LOCATION (matching parameter layout).
// adj is indexed by NNGP order (internal).
//
// Adjoint propagation (reverse NNGP order) is sequential.
// Phi gradient loop is independent per observation — OpenMP parallelized.
// Uses Eigen for triangular solves (from cached L).
inline void nngp_nc_backward(
    const double* z,            // z[loc_idx], location-indexed
    double sigma2, double phi,
    const GPData& gp_data,
    const NNGPNCWorkspace& ws,
    const double* dL_dw,        // Likelihood gradient w.r.t. w[loc] (location-indexed)
    double* grad_z,             // Output: full gradient for z[loc] (prior + likelihood)
    double& grad_log_sigma2_lik,// Output: likelihood contribution to sigma2 gradient
    double& grad_log_phi_lik,   // Output: likelihood contribution to phi gradient
    double& grad_log_phi_jac    // Output: Jacobian contribution to phi gradient
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    const std::vector<int>& nn_order_inv = gp_data.nn_order_inv;

    // Initialize adjoint from direct likelihood contribution (NNGP-order indexed)
    std::vector<double>& adj = const_cast<NNGPNCWorkspace&>(ws).adj;
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        adj[i] = dL_dw[loc];
    }

    // Backward adjoint propagation (reverse NNGP order) — SEQUENTIAL
    for (int i = N - 1; i >= 1; i--) {
        int n_nb = ws.B_n_nb[i];
        for (int j = 0; j < n_nb; j++) {
            int nb_loc = ws.nb_idx_flat[i * nn + j];
            if (nb_loc >= 0 && nb_loc < N) {
                int nb_nngp = nn_order_inv[nb_loc];
                if (nb_nngp >= 0 && nb_nngp < N) {
                    adj[nb_nngp] += ws.B_flat[i * nn + j] * adj[i];
                }
            }
        }
    }

    // z gradients: prior (-z) + likelihood (sqrt_d * adj)
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        grad_z[loc] = -z[loc] + ws.sqrt_d[i] * adj[i];
    }

    // --- Hyperparameter gradients ---

    // sigma2 likelihood gradient
    grad_log_sigma2_lik = 0.0;
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        grad_log_sigma2_lik += adj[i] * 0.5 * ws.sqrt_d[i] * z[loc];
    }

    // phi gradients (likelihood + Jacobian) — OpenMP parallelized
    // Each observation's phi contribution is independent.
    grad_log_phi_lik = 0.0;
    grad_log_phi_jac = 0.0;

    // Thread-local workspace setup
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = std::max(1, std::min(omp_get_max_threads(), N - 1));
    #endif

    std::vector<double> tl_phi_lik(n_threads, 0.0);
    std::vector<double> tl_phi_jac(n_threads, 0.0);

    struct BackwardWS {
        Eigen::MatrixXd L_eigen, C_eigen;
        Eigen::VectorXd c_eigen, dc_eigen, rhs_eigen, dalpha_eigen, alpha_eigen;
        std::vector<double> dC_alpha;
        BackwardWS(int nn_) : L_eigen(nn_, nn_), C_eigen(nn_, nn_),
                              c_eigen(nn_), dc_eigen(nn_), rhs_eigen(nn_),
                              dalpha_eigen(nn_), alpha_eigen(nn_), dC_alpha(nn_) {}
    };
    std::vector<BackwardWS> bws_vec(n_threads, BackwardWS(nn));

    #ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads)
    #endif
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        auto& L_eigen = bws_vec[tid].L_eigen;
        auto& C_eigen = bws_vec[tid].C_eigen;
        auto& c_eigen = bws_vec[tid].c_eigen;
        auto& dc_eigen = bws_vec[tid].dc_eigen;
        auto& rhs_eigen = bws_vec[tid].rhs_eigen;
        auto& dalpha_eigen = bws_vec[tid].dalpha_eigen;
        auto& alpha_eigen = bws_vec[tid].alpha_eigen;
        auto& dC_alpha = bws_vec[tid].dC_alpha;

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (int i = 1; i < N; i++) {
            int obs_loc = gp_data.nn_order[i];
            int n_nb = ws.B_n_nb[i];
            if (n_nb == 0 || obs_loc < 0 || obs_loc >= N) continue;

            // Rebuild c_vec, dc_vec, and C_mat for phi derivatives
            for (int j = 0; j < n_nb; j++) {
                double d = gp_data.nn_dist[i * nn + j];
                c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
                dc_eigen(j) = dcov_dphi(d, phi, c_eigen(j), gp_data.cov_type);
                alpha_eigen(j) = ws.B_flat[i * nn + j];
            }

            // Rebuild C_mat from cached distances (needed for dcov_dphi)
            for (int j1 = 0; j1 < n_nb; j1++) {
                C_eigen(j1, j1) = sigma2;
                for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                    double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                    double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
                    C_eigen(j1, j2) = cov_val;
                    C_eigen(j2, j1) = cov_val;
                }
            }

            // Restore cached L factor into Eigen matrix
            L_eigen.topLeftCorner(n_nb, n_nb).setZero();
            for (int j1 = 0; j1 < n_nb; j1++) {
                for (int j2 = 0; j2 <= j1; j2++) {
                    L_eigen(j1, j2) = ws.L_flat[i * nn * nn + j1 * nn + j2];
                }
            }

            // dC/dphi * alpha (using properly rebuilt C_mat for dcov_dphi)
            std::fill(dC_alpha.begin(), dC_alpha.begin() + n_nb, 0.0);
            for (int j1 = 0; j1 < n_nb; j1++) {
                for (int j2 = 0; j2 < n_nb; j2++) {
                    if (j1 != j2) {
                        double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                        double dC_jk = dcov_dphi(d12, phi, C_eigen(j1, j2), gp_data.cov_type);
                        dC_alpha[j1] += dC_jk * alpha_eigen(j2);
                    }
                }
            }

            // dalpha/dphi = C^{-1} (dc/dphi - dC/dphi * alpha)
            // Use Eigen triangular solve with cached L
            for (int j = 0; j < n_nb; j++) rhs_eigen(j) = dc_eigen(j) - dC_alpha[j];
            auto L_sub = L_eigen.topLeftCorner(n_nb, n_nb);
            Eigen::VectorXd y_temp = L_sub.triangularView<Eigen::Lower>().solve(rhs_eigen.head(n_nb));
            dalpha_eigen.head(n_nb) = L_sub.transpose().triangularView<Eigen::Upper>().solve(y_temp);

            // dd/dphi = -2 * dc' * alpha + alpha' * dC * alpha
            double alpha_dc = 0.0, alpha_dC_alpha = 0.0;
            for (int j = 0; j < n_nb; j++) {
                alpha_dc += alpha_eigen(j) * dc_eigen(j);
                alpha_dC_alpha += alpha_eigen(j) * dC_alpha[j];
            }
            double dd_dphi = -2.0 * alpha_dc + alpha_dC_alpha;

            // Likelihood: dw[loc]/dphi = sum_j dalpha[j]*w[nb_j] + dd_dphi/(2*sqrt(d_i))*z[loc]
            double dw_dphi = 0.0;
            for (int j = 0; j < n_nb; j++) {
                dw_dphi += dalpha_eigen(j) * ws.w[ws.nb_idx_flat[i * nn + j]];
            }
            double sqrt_di = ws.sqrt_d[i];
            if (sqrt_di > 1e-15) {
                dw_dphi += dd_dphi / (2.0 * sqrt_di) * z[obs_loc];
            }
            tl_phi_lik[tid] += adj[i] * dw_dphi * phi;

            // Jacobian: d/d(phi) [0.5*log(d_i)] = 0.5 * dd_dphi / d_i
            double d_i = sqrt_di * sqrt_di;
            if (d_i > 1e-15) {
                tl_phi_jac[tid] += 0.5 * dd_dphi / d_i * phi;
            }
        }
    }

    // Reduce thread-local phi accumulators
    for (int t = 0; t < n_threads; t++) {
        grad_log_phi_lik += tl_phi_lik[t];
        grad_log_phi_jac += tl_phi_jac[t];
    }
}

} // namespace tulpa_gp

#endif // TULPA_HMC_GP_H
