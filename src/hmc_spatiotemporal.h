// hmc_spatiotemporal.h
// Spatiotemporal interaction effects for HMC backend
// Supports Type I-IV interactions (Knorr-Held) and separable/non-separable GP

#ifndef TULPA_HMC_SPATIOTEMPORAL_H
#define TULPA_HMC_SPATIOTEMPORAL_H

#include <vector>
#include <cmath>
#include "hmc_temporal.h"
#include "hmc_svc.h"
#include "linalg_fast.h"  // shared small-dense Cholesky / NNGP solve core

// Use canonical type definitions from exported headers
#include "tulpa/st_data.h"
#include "tulpa/types.h"

namespace tulpa_spatiotemporal {

using tulpa::TemporalType;
using tulpa::CovType;
using tulpa::STType;
using tulpa::NonsepType;
using tulpa::SpatiotemporalData;
using tulpa_svc::compute_cov;

// =====================================================================
// Type I: IID interaction
// =====================================================================

// Log-prior for Type I (IID) interaction
// delta[s,t] ~ N(0, sigma2) independently
inline double type_i_log_prior(
    const double* delta,  // Flattened S*T vector
    int S,
    int T,
    double tau  // precision = 1/sigma2
) {
  int n = S * T;
  double quad = 0.0;

  for (int i = 0; i < n; i++) {
    quad += delta[i] * delta[i];
  }

  // log p(delta|tau) = (n/2) log(tau) - (tau/2) sum(delta^2) + const
  return 0.5 * n * std::log(tau) - 0.5 * tau * quad;
}

// =====================================================================
// Type II: Structured time at each location
// =====================================================================

// Log-prior for Type II interaction
// For each spatial unit s, delta[s,*] follows temporal structure
inline double type_ii_log_prior(
    const double* delta,  // Flattened S*T vector (row-major: delta[s*T + t], t contiguous)
    int S,
    int T,
    double tau,
    TemporalType temp_type,
    bool cyclic
) {
  double log_prior = 0.0;

  // For each spatial unit
  for (int s = 0; s < S; s++) {
    // Extract temporal series for this location
    std::vector<double> delta_s(T);
    for (int t = 0; t < T; t++) {
      delta_s[t] = delta[s * T + t];  // row-major: t contiguous within unit s
    }

    // Apply temporal prior
    if (temp_type == TemporalType::RW1) {
      double quad = tulpa_temporal::rw1_quadratic_form(delta_s.data(), T, cyclic);
      int rank = tulpa_temporal::rw1_rank(T, cyclic);
      log_prior += 0.5 * rank * std::log(tau) - 0.5 * tau * quad;
    } else if (temp_type == TemporalType::RW2) {
      double quad = tulpa_temporal::rw2_quadratic_form(delta_s.data(), T, cyclic);
      int rank = tulpa_temporal::rw2_rank(T, cyclic);
      log_prior += 0.5 * rank * std::log(tau) - 0.5 * tau * quad;
    }
    // AR1 would need rho parameter
  }

  return log_prior;
}

// =====================================================================
// Type III: Structured space at each time point
// =====================================================================

// Compute ICAR quadratic form for a single vector
inline double icar_quad_form(
    const double* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx
) {
  double quad = 0.0;

  for (int i = 0; i < n; i++) {
    for (int idx = adj_row_ptr[i]; idx < adj_row_ptr[i + 1]; idx++) {
      int j = adj_col_idx[idx] - 1;  // Convert to 0-based
      if (j > i) {  // Only count each edge once
        double diff = phi[i] - phi[j];
        quad += diff * diff;
      }
    }
  }

  return quad;
}

// Log-prior for Type III interaction
// For each time point t, delta[*,t] follows spatial structure (ICAR)
inline double type_iii_log_prior(
    const double* delta,  // Flattened S*T vector
    int S,
    int T,
    double tau,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    int rank_deficiency = 1  // 1 for ICAR
) {
  double log_prior = 0.0;
  int rank = S - rank_deficiency;

  // For each time point
  for (int t = 0; t < T; t++) {
    // Extract spatial field for this time point
    std::vector<double> delta_t(S);
    for (int s = 0; s < S; s++) {
      delta_t[s] = delta[s * T + t];  // row-major: delta[s*T + t]
    }

    // Apply ICAR prior
    double quad = icar_quad_form(delta_t.data(), S, adj_row_ptr, adj_col_idx);
    log_prior += 0.5 * rank * std::log(tau) - 0.5 * tau * quad;
  }

  return log_prior;
}

// =====================================================================
// Type IV: Fully structured (Kronecker)
// =====================================================================

// Log-prior for Type IV interaction
// Precision Q_delta = Q_s (x) Q_t (Kronecker product)
// Quadratic form: delta' Q_delta delta = sum over (s1,t1,s2,t2) of delta[s1,t1] * Q_s[s1,s2] * Q_t[t1,t2] * delta[s2,t2]
// This can be computed as: sum_t1,t2 Q_t[t1,t2] * (delta[*,t1]' Q_s delta[*,t2])
inline double type_iv_log_prior(
    const double* delta,  // Flattened S*T vector (row-major: delta[s*T + t])
    int S,
    int T,
    double tau_space,
    double tau_time,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    TemporalType temp_type,
    bool cyclic
) {
  // For Type IV, we compute the Kronecker quadratic form
  // delta' (Q_s (x) Q_t) delta

  // Step 1: Compute Q_s * delta for each time point
  // Q_s[i,j] = n_neighbors[i] if i==j, -1 if neighbors, 0 otherwise
  std::vector<std::vector<double>> Q_s_delta(T, std::vector<double>(S, 0.0));

  for (int t = 0; t < T; t++) {
    for (int i = 0; i < S; i++) {
      double val = 0.0;
      int n_neigh = adj_row_ptr[i + 1] - adj_row_ptr[i];
      val += n_neigh * delta[i * T + t];  // Diagonal: n_neighbors

      for (int idx = adj_row_ptr[i]; idx < adj_row_ptr[i + 1]; idx++) {
        int j = adj_col_idx[idx] - 1;  // 0-based
        val -= delta[j * T + t];  // Off-diagonal: -1
      }
      Q_s_delta[t][i] = val;
    }
  }

  // Step 2: Compute delta[*,t1]' Q_s delta[*,t2] for all t1, t2
  std::vector<std::vector<double>> inner_prods(T, std::vector<double>(T, 0.0));

  for (int t1 = 0; t1 < T; t1++) {
    for (int t2 = 0; t2 < T; t2++) {
      double ip = 0.0;
      for (int s = 0; s < S; s++) {
        ip += delta[s * T + t1] * Q_s_delta[t2][s];
      }
      inner_prods[t1][t2] = ip;
    }
  }

  // Step 3: Apply temporal quadratic form
  double quad = 0.0;

  if (temp_type == TemporalType::RW1) {
    // RW1: Q_t[t,t] = 2 (interior), 1 (boundary); Q_t[t,t+1] = -1
    for (int t = 0; t < T - 1; t++) {
      // (delta[*,t] - delta[*,t+1])' Q_s (delta[*,t] - delta[*,t+1])
      // = inner_prods[t,t] - 2*inner_prods[t,t+1] + inner_prods[t+1,t+1]
      quad += inner_prods[t][t] - 2.0 * inner_prods[t][t + 1] + inner_prods[t + 1][t + 1];
    }
    if (cyclic) {
      quad += inner_prods[T-1][T-1] - 2.0 * inner_prods[T-1][0] + inner_prods[0][0];
    }
  } else if (temp_type == TemporalType::RW2) {
    // RW2: second differences
    for (int t = 0; t < T - 2; t++) {
      // (delta[*,t] - 2*delta[*,t+1] + delta[*,t+2])' Q_s (...)
      std::vector<double> diff2(S);
      for (int s = 0; s < S; s++) {
        diff2[s] = delta[s * T + t] - 2.0 * delta[s * T + t + 1] + delta[s * T + t + 2];
      }
      // Apply Q_s to diff2
      for (int s = 0; s < S; s++) {
        double Qs_diff2 = 0.0;
        int n_neigh = adj_row_ptr[s + 1] - adj_row_ptr[s];
        Qs_diff2 += n_neigh * diff2[s];
        for (int idx = adj_row_ptr[s]; idx < adj_row_ptr[s + 1]; idx++) {
          int j = adj_col_idx[idx] - 1;
          Qs_diff2 -= diff2[j];
        }
        quad += diff2[s] * Qs_diff2;
      }
    }
    // Cyclic extension if needed
  }

  // Compute log-determinant contribution
  // For Kronecker: log|Q_s (x) Q_t| = T*log|Q_s| + S*log|Q_t|
  // But for improper priors (ICAR, RW), we use rank instead

  int rank_space = S - 1;  // ICAR rank deficiency
  int rank_time = (temp_type == TemporalType::RW1) ? (T - 1) : (T - 2);
  if (cyclic) rank_time = T;

  int total_rank = rank_space * rank_time;

  double log_prior = 0.5 * total_rank * std::log(tau_space * tau_time);
  log_prior -= 0.5 * tau_space * tau_time * quad;

  return log_prior;
}

// =====================================================================
// Non-separable GP covariance functions
// =====================================================================

// Gneiting (2002) non-separable covariance
// C(h, u) = sigma2 / (a*|u|^(2*alpha) + 1)^tau * exp(-c*||h||^(2*gamma) / (a*|u|^(2*alpha) + 1)^(beta*gamma))
inline double gneiting_cov(
    double h,           // Spatial distance
    double u,           // Temporal distance (absolute)
    double sigma2,
    double phi_space,   // Spatial range
    double phi_time,    // Temporal range
    double alpha = 1.0, // Temporal smoothness (0.5 to 1)
    double gamma = 0.5, // Spatial smoothness (0.5 to 1)
    double beta = 1.0   // Space-time interaction (0 to 1)
) {
  // Rescale distances by range parameters
  double u_scaled = u / phi_time;
  double h_scaled = h / phi_space;

  // Parameters for the class
  double a = 1.0;  // Can be estimated
  double c = 1.0;  // Can be estimated
  double tau = 1.0;

  double denom = std::pow(a * std::pow(std::abs(u_scaled), 2.0 * alpha) + 1.0, tau);
  double exp_arg = -c * std::pow(h_scaled, 2.0 * gamma) /
                   std::pow(a * std::pow(std::abs(u_scaled), 2.0 * alpha) + 1.0, beta * gamma);

  return sigma2 / denom * std::exp(exp_arg);
}

// Separable covariance (product)
inline double separable_cov(
    double h,
    double u,
    double sigma2_space,
    double sigma2_time,
    double phi_space,
    double phi_time,
    CovType cov_space,
    CovType cov_time
) {
  double c_space = compute_cov(h, sigma2_space, phi_space, cov_space);
  double c_time = compute_cov(u, sigma2_time, phi_time, cov_time);
  return c_space * c_time;
}

// Sum covariance
inline double sum_cov(
    double h,
    double u,
    double sigma2_space,
    double sigma2_time,
    double phi_space,
    double phi_time,
    CovType cov_space,
    CovType cov_time
) {
  double c_space = compute_cov(h, sigma2_space, phi_space, cov_space);
  double c_time = compute_cov(u, sigma2_time, phi_time, cov_time);
  return c_space + c_time;
}

// =====================================================================
// Non-separable GP NNGP likelihood
// =====================================================================

// Compute NNGP log-likelihood for spatiotemporal GP
inline double st_gp_nngp_log_lik(
    const std::vector<double>& w,      // ST effect (length N)
    double sigma2,
    double phi_space,
    double phi_time,
    const SpatiotemporalData& st_data
) {
  int N = w.size();
  int nn = st_data.nn;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = st_data.nn_order[0];
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
    int obs_idx = st_data.nn_order[i];

    // Count actual neighbors
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      if (st_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

    if (n_neighbors == 0) {
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build covariance structures
    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double h = st_data.nn_dist_space[nn_flat_idx];
      double u = st_data.nn_dist_time[nn_flat_idx];

      if (st_data.nonsep_type == NonsepType::GNEITING) {
        c_vec[j] = gneiting_cov(h, u, sigma2, phi_space, phi_time);
      } else if (st_data.nonsep_type == NonsepType::SUM) {
        c_vec[j] = sum_cov(h, u, sigma2, sigma2, phi_space, phi_time,
                          st_data.cov_space, st_data.cov_time);
      } else {
        // Separable (product)
        c_vec[j] = separable_cov(h, u, std::sqrt(sigma2), std::sqrt(sigma2),
                                 phi_space, phi_time,
                                 st_data.cov_space, st_data.cov_time);
      }
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int idx1 = st_data.nn_order[st_data.nn_idx[i * nn + j1] - 1];
      for (int j2 = j1; j2 < n_neighbors; j2++) {
        int idx2 = st_data.nn_order[st_data.nn_idx[i * nn + j2] - 1];

        double h12, u12;
        if (j1 == j2) {
          h12 = 0.0;
          u12 = 0.0;
        } else {
          // Compute distance between neighbors j1 and j2
          h12 = std::sqrt(
            std::pow(st_data.coords[idx1 * 2] - st_data.coords[idx2 * 2], 2) +
            std::pow(st_data.coords[idx1 * 2 + 1] - st_data.coords[idx2 * 2 + 1], 2)
          );
          u12 = std::abs(st_data.time_values[idx1] - st_data.time_values[idx2]);
        }

        double cov_val;
        if (j1 == j2) {
          cov_val = sigma2;
        } else if (st_data.nonsep_type == NonsepType::GNEITING) {
          cov_val = gneiting_cov(h12, u12, sigma2, phi_space, phi_time);
        } else if (st_data.nonsep_type == NonsepType::SUM) {
          cov_val = sum_cov(h12, u12, sigma2, sigma2, phi_space, phi_time,
                           st_data.cov_space, st_data.cov_time);
        } else {
          cov_val = separable_cov(h12, u12, std::sqrt(sigma2), std::sqrt(sigma2),
                                  phi_space, phi_time,
                                  st_data.cov_space, st_data.cov_time);
        }

        C_mat[j1 * n_neighbors + j2] = cov_val;
        C_mat[j2 * n_neighbors + j1] = cov_val;
      }
    }

    // Gather neighbor values in c_vec order, then shared factor/solve core
    std::vector<double> w_nb(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
      int nn_orig_idx = st_data.nn_order[st_data.nn_idx[i * nn + j] - 1];
      w_nb[j] = w[nn_orig_idx];
    }
    double cond_mean, cond_var;
    tulpa_linalg::nngp_conditional_moments(
        C_mat.data(), c_vec.data(), w_nb.data(), n_neighbors, sigma2,
        tulpa_linalg::kCholJitter, tulpa_linalg::kCholJitter,
        cond_mean, cond_var);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// =====================================================================
// Master function: spatiotemporal log-prior
// =====================================================================

inline double spatiotemporal_log_prior(
    const double* delta,
    double tau,
    double tau2,        // Second precision (for Type IV: temporal)
    double rho,         // AR1 autocorrelation if needed
    double phi_space,   // GP range parameters
    double phi_time,
    const SpatiotemporalData& st_data
) {
  if (st_data.type == STType::NONE) {
    return 0.0;
  }

  int S = st_data.n_spatial;
  int T = st_data.n_times;

  switch (st_data.type) {
    case STType::TYPE_I:
      return type_i_log_prior(delta, S, T, tau);

    case STType::TYPE_II:
      return type_ii_log_prior(delta, S, T, tau,
                               st_data.temporal_type, st_data.temporal_cyclic);

    case STType::TYPE_III:
      return type_iii_log_prior(delta, S, T, tau,
                                st_data.adj_row_ptr, st_data.adj_col_idx);

    case STType::TYPE_IV:
      return type_iv_log_prior(delta, S, T, tau, tau2,
                               st_data.adj_row_ptr, st_data.adj_col_idx,
                               st_data.temporal_type, st_data.temporal_cyclic);

    case STType::SEPARABLE:
    case STType::NONSEP_GP: {
      // Convert delta to vector for GP function
      int N = st_data.n_params;
      std::vector<double> w(delta, delta + N);
      double sigma2 = 1.0 / tau;
      return st_gp_nngp_log_lik(w, sigma2, phi_space, phi_time, st_data);
    }

    default:
      return 0.0;
  }
}

// =====================================================================
// Gradient helpers (numerical)
// =====================================================================

inline void spatiotemporal_gradient_delta(
    const double* delta,
    double tau,
    double tau2,
    double rho,
    double phi_space,
    double phi_time,
    const SpatiotemporalData& st_data,
    double* grad,
    double epsilon = 1e-6
) {
  int n_params = st_data.n_params;

  double base_ll = spatiotemporal_log_prior(delta, tau, tau2, rho,
                                            phi_space, phi_time, st_data);

  std::vector<double> delta_plus(delta, delta + n_params);

  for (int i = 0; i < n_params; i++) {
    delta_plus[i] = delta[i] + epsilon;
    double ll_plus = spatiotemporal_log_prior(delta_plus.data(), tau, tau2, rho,
                                              phi_space, phi_time, st_data);
    grad[i] = (ll_plus - base_ll) / epsilon;
    delta_plus[i] = delta[i];
  }
}

// =====================================================================
// Sum-to-zero constraint for interactions
// =====================================================================

// Apply soft sum-to-zero constraint marginally. Each margin gets the precision
// for its own length -- a space margin sums S terms, a time margin sums T --
// so the two cannot share one constant (see tulpa/soft_sum_to_zero.h).
inline double st_sum_to_zero_penalty(
    const double* delta,
    int S,
    int T,
    bool marginal_space = true,
    bool marginal_time = true
) {
  double penalty = 0.0;

  if (marginal_space) {
    // For each time point, spatial effects sum to zero
    const double lambda_s = tulpa::s2z_precision(S);
    for (int t = 0; t < T; t++) {
      double sum = 0.0;
      for (int s = 0; s < S; s++) {
        sum += delta[s * T + t];
      }
      penalty -= 0.5 * lambda_s * sum * sum;
    }
  }

  if (marginal_time) {
    // For each spatial unit, temporal effects sum to zero
    const double lambda_t = tulpa::s2z_precision(T);
    for (int s = 0; s < S; s++) {
      double sum = 0.0;
      for (int t = 0; t < T; t++) {
        sum += delta[s * T + t];
      }
      penalty -= 0.5 * lambda_t * sum * sum;
    }
  }

  return penalty;
}

} // namespace tulpa_spatiotemporal

#endif // TULPA_HMC_SPATIOTEMPORAL_H
