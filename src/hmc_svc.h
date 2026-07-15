// hmc_svc.h
// Spatially-Varying Coefficients (SVC) with NNGP approximation
// Implements Nearest Neighbor Gaussian Process for scalable GP inference

#ifndef TULPA_HMC_SVC_H
#define TULPA_HMC_SVC_H

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <vector>
#include <cmath>
#include <algorithm>
#include <Rcpp.h>  // For Rcpp::Rcout in debug
#include "tulpa/svc_data.h"
#include "tulpa/types.h"
#include "linalg_fast.h"  // shared small-dense Cholesky / NNGP solve core

// Fallback definition of M_PI if not provided by <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_svc {

using tulpa::CovType;
using tulpa::SVCData;

// -----------------------------------------------------------------------------
// Covariance functions
// -----------------------------------------------------------------------------

// Exponential covariance: sigma^2 * exp(-d / phi)
inline double cov_exponential(double d, double sigma2, double phi) {
  return sigma2 * std::exp(-d / phi);
}

// Matern 3/2 covariance: sigma^2 * (1 + sqrt(3)*d/phi) * exp(-sqrt(3)*d/phi)
inline double cov_matern32(double d, double sigma2, double phi) {
  double r = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + r) * std::exp(-r);
}

// Gaussian (squared exponential) covariance: sigma^2 * exp(-(d/phi)^2)
inline double cov_gaussian(double d, double sigma2, double phi) {
  double r = d / phi;
  return sigma2 * std::exp(-r * r);
}

// Spherical covariance
inline double cov_spherical(double d, double sigma2, double phi) {
  if (d >= phi) return 0.0;
  double r = d / phi;
  return sigma2 * (1.0 - 1.5 * r + 0.5 * r * r * r);
}

// Generic covariance function dispatcher
inline double compute_cov(double d, double sigma2, double phi, CovType cov_type) {
  switch (cov_type) {
    case CovType::EXPONENTIAL:
      return cov_exponential(d, sigma2, phi);
    case CovType::MATERN:
      return cov_matern32(d, sigma2, phi);
    case CovType::GAUSSIAN:
      return cov_gaussian(d, sigma2, phi);
    case CovType::SPHERICAL:
      return cov_spherical(d, sigma2, phi);
    default:
      return cov_exponential(d, sigma2, phi);
  }
}

// -----------------------------------------------------------------------------
// NNGP likelihood computation
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for a single SVC term
// w: vector of SVC values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
// Returns log p(w | sigma2, phi) under NNGP approximation
inline double nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const SVCData& svc_data
) {
  int N = svc_data.n_obs;
  int nn = svc_data.nn;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = svc_data.nn_order[0];
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
    int obs_idx = svc_data.nn_order[i];

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      if (svc_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build covariance vector c(s_i, s_{N(i)}) and matrix C(s_{N(i)}, s_{N(i)})
    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = svc_data.nn_dist[nn_flat_idx];
      c_vec[j] = compute_cov(d, sigma2, phi, svc_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int nn_idx1 = svc_data.nn_order[svc_data.nn_idx[i * nn + j1] - 1];
      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int nn_idx2 = svc_data.nn_order[svc_data.nn_idx[i * nn + j2] - 1];

        if (j1 == j2) {
          C_mat[j1 * n_neighbors + j2] = sigma2;
        } else {
          // Compute distance between neighbors
          double d12 = std::sqrt(
            std::pow(svc_data.coords[nn_idx1 * 2] - svc_data.coords[nn_idx2 * 2], 2) +
            std::pow(svc_data.coords[nn_idx1 * 2 + 1] - svc_data.coords[nn_idx2 * 2 + 1], 2)
          );
          C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, svc_data.cov_type);
        }
      }
    }

    // Gather neighbor values in c_vec order, then shared factor/solve core.
    // This kernel uses a larger jitter / variance floor (1e-6) to prevent
    // ill-conditioning on near-duplicate coordinates.
    std::vector<double> w_nb(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
      int nn_orig_idx = svc_data.nn_order[svc_data.nn_idx[i * nn + j] - 1];
      w_nb[j] = w[nn_orig_idx];
    }
    double cond_mean, cond_var;
    tulpa_linalg::nngp_conditional_moments(
        C_mat.data(), c_vec.data(), w_nb.data(), n_neighbors, sigma2,
        /*jitter=*/1e-6, /*var_floor=*/1e-6, cond_mean, cond_var);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// SVC contribution to linear predictor
// -----------------------------------------------------------------------------

// Compute SVC contribution to linear predictor for all observations
// eta_svc[i] = sum_j X_svc[i,j] * w_j[i]
inline void compute_svc_eta(
    const std::vector<double>& w_flat,  // n_obs x n_svc flattened
    const SVCData& svc_data,
    std::vector<double>& eta_svc         // Output: length n_obs
) {
  int N = svc_data.n_obs;
  int n_svc = svc_data.n_svc;

  std::fill(eta_svc.begin(), eta_svc.end(), 0.0);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n_svc; j++) {
      // w_flat is stored as [w1[1..N], w2[1..N], ...]
      double w_ij = w_flat[j * N + i];
      double x_ij = svc_data.X_svc[i * n_svc + j];
      eta_svc[i] += x_ij * w_ij;
    }
  }
}

// -----------------------------------------------------------------------------
// Sum-to-zero constraint for identifiability
// -----------------------------------------------------------------------------

// Apply soft sum-to-zero constraint on SVC weights (for each SVC term)
// Without this, beta and mean(w) are not separately identifiable.
// Uses mean(w) with lambda_mean penalty: -0.5 * lambda_mean * N * mean(w)^2
// This is equivalent to: mean(w) ~ N(0, 1/lambda_mean)
inline double svc_sum_to_zero_penalty(
    const std::vector<double>& w_flat,
    const SVCData& svc_data,
    double lambda_mean = 1.0
) {
  int n_obs = svc_data.n_obs;
  int n_svc = svc_data.n_svc;

  double penalty = 0.0;

  for (int j = 0; j < n_svc; j++) {
    double sum = 0.0;
    for (int i = 0; i < n_obs; i++) {
      sum += w_flat[j * n_obs + i];
    }
    double mean_w = sum / n_obs;
    penalty -= 0.5 * lambda_mean * n_obs * mean_w * mean_w;
  }

  return penalty;
}

// -----------------------------------------------------------------------------
// Prior on GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for sigma2 (spatial variance): Half-Cauchy or exponential
inline double log_prior_sigma2(double sigma2, double scale) {
  // Half-Cauchy(0, scale): 2 / (pi * scale * (1 + (sigma2/scale)^2))
  // On log scale for sigma = sqrt(sigma2)
  double sigma = std::sqrt(sigma2);
  return std::log(2.0 / (M_PI * scale)) - std::log(1.0 + sigma * sigma / (scale * scale));
}

// Log prior for phi (range parameter): Uniform or exponential
inline double log_prior_phi(double phi, double lower, double upper) {
  // Uniform(lower, upper)
  if (phi < lower || phi > upper) return -INFINITY;
  return -std::log(upper - lower);
}

// Parse covariance type from string
inline CovType parse_cov_type(const std::string& cov_str) {
  static const tulpa::EnumEntry<CovType> table[] = {
    {"exponential", CovType::EXPONENTIAL}, {"matern", CovType::MATERN},
    {"gaussian", CovType::GAUSSIAN}, {"spherical", CovType::SPHERICAL}
  };
  return tulpa::parse_enum(cov_str, table, CovType::EXPONENTIAL);
}

// -----------------------------------------------------------------------------
// Gradient computation for SVC parameters (for hand-coded HMC gradients)
// -----------------------------------------------------------------------------

// Struct to hold SVC NNGP gradient results
struct SVCGradients {
  std::vector<double> grad_w;         // Gradient w.r.t. spatial effects (length n_obs)
  double grad_log_sigma2;             // Gradient w.r.t. log(sigma2)
  double grad_log_phi;                // Gradient w.r.t. log(phi)
};

// Covariance derivative w.r.t. phi: dk(d)/dphi. Single source of truth for
// every NNGP gradient path (SVC and GP); see dcov_dphi in hmc_gp_gradients.h,
// which delegates here.
//
// sigma2 is needed for SPHERICAL alone: the other kernels' derivatives are
// proportional to k(d), so they can be written from cov_val, but the spherical
// polynomial's is not. It was previously omitted for that reason and fell
// through to the exponential derivative -- the value used one kernel and its
// gradient another.
inline double dcov_dphi_svc(double d, double phi, double cov_val, double sigma2,
                            CovType cov_type) {
  if (d < 1e-10) return 0.0;
  switch (cov_type) {
    case CovType::EXPONENTIAL:
      // k = s2*exp(-d/phi) -> dk/dphi = k*d/phi^2
      return cov_val * d / (phi * phi);
    case CovType::MATERN: {
      // k = s2*(1+u)*exp(-u), u = sqrt(3)*d/phi -> dk/dphi = k*u^2/(phi*(1+u))
      double u = 1.732050808 * d / phi;  // sqrt(3) * d / phi
      return (1.0 + u > 1e-10) ? cov_val * u * u / (phi * (1.0 + u)) : 0.0;
    }
    case CovType::GAUSSIAN:
      // k = s2*exp(-(d/phi)^2) -> dk/dphi = k*2*d^2/phi^3
      return cov_val * 2.0 * d * d / (phi * phi * phi);
    case CovType::SPHERICAL: {
      // k = s2*(1 - 1.5r + 0.5r^3) for r = d/phi < 1, else 0 (and flat there)
      // -> dk/dphi = s2 * 1.5 * r * (1 - r^2) / phi
      if (d >= phi) return 0.0;
      double r = d / phi;
      return sigma2 * 1.5 * r * (1.0 - r * r) / phi;
    }
    default:
      return cov_val * d / (phi * phi);
  }
}

// SVC gradient diagnostics: compile-time-false like GP_DEBUG_BOUNDS /
// GP_AUTODIFF_DEBUG. The gradient runs on parallel-chain workers, where
// Rcpp::Rcout (R API) is not thread-safe and a file-static first-call
// counter races; flip the macro locally for a serial debugging session.
#ifndef SVC_GRADIENT_DEBUG
#define SVC_GRADIENT_DEBUG false
#endif

// Fully analytical NNGP gradients for SVC - single pass, no redundant function calls
// Complexity: O(N * nn²) - ~4x faster than numerical
inline void svc_nngp_gradients(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const SVCData& svc_data,
    SVCGradients& grads
) {
  int N = svc_data.n_obs;
  int nn = svc_data.nn;
  const bool debug = SVC_GRADIENT_DEBUG;

  grads.grad_w.assign(N, 0.0);
  grads.grad_log_sigma2 = 0.0;
  grads.grad_log_phi = 0.0;

  // Validate - with debug output
  bool val_fail = false;
  if ((int)svc_data.nn_order.size() < N) {
    if (debug) Rcpp::Rcout << "[SVC DEBUG] FAIL: nn_order.size()=" << svc_data.nn_order.size() << " < N=" << N << "\n";
    val_fail = true;
  }
  if ((int)svc_data.nn_idx.size() < N * nn) {
    if (debug) Rcpp::Rcout << "[SVC DEBUG] FAIL: nn_idx.size()=" << svc_data.nn_idx.size() << " < N*nn=" << N*nn << "\n";
    val_fail = true;
  }
  if ((int)svc_data.nn_dist.size() < N * nn) {
    if (debug) Rcpp::Rcout << "[SVC DEBUG] FAIL: nn_dist.size()=" << svc_data.nn_dist.size() << " < N*nn=" << N*nn << "\n";
    val_fail = true;
  }
  if ((int)w.size() < N) {
    if (debug) Rcpp::Rcout << "[SVC DEBUG] FAIL: w.size()=" << w.size() << " < N=" << N << "\n";
    val_fail = true;
  }
  if ((int)svc_data.coords.size() < 2 * N) {
    if (debug) Rcpp::Rcout << "[SVC DEBUG] FAIL: coords.size()=" << svc_data.coords.size() << " < 2*N=" << 2*N << "\n";
    val_fail = true;
  }
  if (val_fail) {
    return;
  }

  if (debug) {
    Rcpp::Rcout << "[SVC DEBUG] Validation passed: N=" << N << ", nn=" << nn
                << ", sigma2=" << sigma2 << ", phi=" << phi << "\n";
  }

  // First observation: marginal N(0, sigma2)
  int first_idx = svc_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) {
    if (debug) Rcpp::Rcout << "[SVC DEBUG] FAIL: first_idx=" << first_idx << " out of bounds [0," << N << ")\n";
    return;
  }
  double w0 = w[first_idx];
  grads.grad_w[first_idx] = -w0 / sigma2;
  grads.grad_log_sigma2 += 0.5 * (w0 * w0 / sigma2 - 1.0);  // Will multiply by sigma2 at end

  // Preallocate work arrays (reused across iterations)
  std::vector<double> c_vec(nn), dc_vec(nn), C_mat(nn * nn), L(nn * nn);
  std::vector<double> y_vec(nn), alpha(nn), y2(nn), beta(nn), w_nb(nn);
  std::vector<int> nb_idx(nn);

  for (int i = 1; i < N; i++) {
    int obs_idx = svc_data.nn_order[i];
    if (obs_idx < 0 || obs_idx >= N) continue;

    // Count neighbors
    int n_nb = 0;
    for (int j = 0; j < nn && svc_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

    if (n_nb == 0) {
      double wi = w[obs_idx];
      grads.grad_w[obs_idx] += -wi / sigma2;
      grads.grad_log_sigma2 += 0.5 * (wi * wi / sigma2 - 1.0);
      continue;
    }

    // Build c_vec, dc_vec (covariances and derivatives)
    for (int j = 0; j < n_nb; j++) {
      double d = svc_data.nn_dist[i * nn + j];
      c_vec[j] = compute_cov(d, sigma2, phi, svc_data.cov_type);
      dc_vec[j] = dcov_dphi_svc(d, phi, c_vec[j], sigma2, svc_data.cov_type);
    }

    // Build C_mat and get neighbor indices
    bool ok = true;
    for (int j1 = 0; j1 < n_nb && ok; j1++) {
      int raw1 = svc_data.nn_idx[i * nn + j1];
      if (raw1 - 1 < 0 || raw1 - 1 >= (int)svc_data.nn_order.size()) { ok = false; break; }
      int idx1 = svc_data.nn_order[raw1 - 1];
      if (idx1 < 0 || idx1 >= N) { ok = false; break; }
      nb_idx[j1] = idx1;

      for (int j2 = 0; j2 < n_nb; j2++) {
        if (j1 == j2) {
          C_mat[j1 * n_nb + j2] = sigma2;
        } else {
          int raw2 = svc_data.nn_idx[i * nn + j2];
          if (raw2 - 1 < 0 || raw2 - 1 >= (int)svc_data.nn_order.size()) { ok = false; break; }
          int idx2 = svc_data.nn_order[raw2 - 1];
          double dx = svc_data.coords[idx1 * 2] - svc_data.coords[idx2 * 2];
          double dy = svc_data.coords[idx1 * 2 + 1] - svc_data.coords[idx2 * 2 + 1];
          C_mat[j1 * n_nb + j2] = compute_cov(std::sqrt(dx*dx + dy*dy), sigma2, phi, svc_data.cov_type);
        }
      }
    }
    if (!ok) {
      double wi = w[obs_idx];
      grads.grad_w[obs_idx] += -wi / sigma2;
      grads.grad_log_sigma2 += 0.5 * (wi * wi / sigma2 - 1.0);
      continue;
    }

    // Cholesky: C = LL'
    std::fill(L.begin(), L.begin() + n_nb * n_nb, 0.0);
    for (int j = 0; j < n_nb; j++) {
      double s = 0.0;
      for (int k = 0; k < j; k++) s += L[j * n_nb + k] * L[j * n_nb + k];
      double diag = C_mat[j * n_nb + j] - s;
      // Numerical stability: larger minimum diagonal for better conditioning
      L[j * n_nb + j] = (diag > 1e-6) ? std::sqrt(diag) : 1e-3;
      for (int k = j + 1; k < n_nb; k++) {
        double t = 0.0;
        for (int m = 0; m < j; m++) t += L[k * n_nb + m] * L[j * n_nb + m];
        L[k * n_nb + j] = (C_mat[k * n_nb + j] - t) / L[j * n_nb + j];
      }
    }

    // Solve L*y = c, L'*alpha = y => alpha = C^{-1}c
    for (int j = 0; j < n_nb; j++) {
      double s = 0.0;
      for (int k = 0; k < j; k++) s += L[j * n_nb + k] * y_vec[k];
      y_vec[j] = (c_vec[j] - s) / L[j * n_nb + j];
    }
    for (int j = n_nb - 1; j >= 0; j--) {
      double s = 0.0;
      for (int k = j + 1; k < n_nb; k++) s += L[k * n_nb + j] * alpha[k];
      alpha[j] = (y_vec[j] - s) / L[j * n_nb + j];
    }

    // Get w_neighbors and solve for beta = C^{-1}w_nb
    for (int j = 0; j < n_nb; j++) w_nb[j] = w[nb_idx[j]];
    for (int j = 0; j < n_nb; j++) {
      double s = 0.0;
      for (int k = 0; k < j; k++) s += L[j * n_nb + k] * y2[k];
      y2[j] = (w_nb[j] - s) / L[j * n_nb + j];
    }
    for (int j = n_nb - 1; j >= 0; j--) {
      double s = 0.0;
      for (int k = j + 1; k < n_nb; k++) s += L[k * n_nb + j] * beta[k];
      beta[j] = (y2[j] - s) / L[j * n_nb + j];
    }

    // Conditional mean and variance
    double mu = 0.0, c_alpha = 0.0;
    for (int j = 0; j < n_nb; j++) { mu += alpha[j] * w_nb[j]; c_alpha += c_vec[j] * alpha[j]; }
    // Numerical stability: larger floor for conditional variance
    double v = std::max(sigma2 - c_alpha, 1e-6);
    double r = w[obs_idx] - mu;

    // Gradient w.r.t. w
    grads.grad_w[obs_idx] += -r / v;
    for (int j = 0; j < n_nb; j++) grads.grad_w[nb_idx[j]] += alpha[j] * r / v;

    // Gradient w.r.t. sigma2: dv/ds2 = 1 - c'α/s2
    double dll_dv = 0.5 * (r * r / v - 1.0) / v;
    grads.grad_log_sigma2 += dll_dv * (1.0 - c_alpha / sigma2) * sigma2;

    // Gradient w.r.t. phi: compute quadratic forms on-the-fly
    double alpha_dc = 0.0, dc_beta = 0.0;
    for (int j = 0; j < n_nb; j++) { alpha_dc += alpha[j] * dc_vec[j]; dc_beta += dc_vec[j] * beta[j]; }

    // alpha' * dC/dphi * alpha and alpha' * dC/dphi * beta (computed on-the-fly)
    double alpha_dC_alpha = 0.0, alpha_dC_beta = 0.0;
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        double dC_jk = 0.0;
        if (j1 != j2) {
          double dx = svc_data.coords[nb_idx[j1] * 2] - svc_data.coords[nb_idx[j2] * 2];
          double dy = svc_data.coords[nb_idx[j1] * 2 + 1] - svc_data.coords[nb_idx[j2] * 2 + 1];
          double d12 = std::sqrt(dx*dx + dy*dy);
          dC_jk = dcov_dphi_svc(d12, phi, C_mat[j1 * n_nb + j2], sigma2,
                                svc_data.cov_type);
        }
        alpha_dC_alpha += alpha[j1] * dC_jk * alpha[j2];
        alpha_dC_beta += alpha[j1] * dC_jk * beta[j2];
      }
    }

    double dv_dphi = -2.0 * alpha_dc + alpha_dC_alpha;
    double dr_dphi = -dc_beta + alpha_dC_beta;
    grads.grad_log_phi += (dll_dv * dv_dphi + (-r / v) * dr_dphi) * phi;
  }

  if (debug) {
    double sum_abs_grad_w = 0.0;
    for (int i = 0; i < N; i++) {
      sum_abs_grad_w += std::abs(grads.grad_w[i]);
    }
    Rcpp::Rcout << "[SVC DEBUG] Output: sum|grad_w|=" << sum_abs_grad_w
                << ", grad_log_sigma2=" << grads.grad_log_sigma2
                << ", grad_log_phi=" << grads.grad_log_phi << "\n";
  }
}

} // namespace tulpa_svc

#endif // TULPA_HMC_SVC_H
