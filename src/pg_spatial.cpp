// pg_spatial.cpp
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) prior for areal data

#include "pg_spatial.h"
#include "pg_shared.h"
#include <Rcpp.h>
#include <cmath>
#include <vector>

using namespace Rcpp;

namespace tulpa {

// ---------------------------------------------------------------------
// ICAR (Intrinsic Conditional Autoregressive) prior
// ---------------------------------------------------------------------

// Update spatial effects with ICAR prior
// Each phi_i is updated given all other phi values and the data
//
// Full conditional for phi_i:
// phi_i | ... ~ N(m_i, v_i)
// v_i = 1 / (tau * n_i + sum(omega_j for j in group i))
// m_i = v_i * (tau * sum(phi_j for neighbors j) + sum(kappa_j - omega_j * offset_j for j in group i))
//
// For group-level spatial effects:
// - Each group j has a spatial effect phi_j
// - Observations in group j share phi_j
// - ICAR prior relates phi across spatial units

NumericVector update_spatial_icar(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& offset,
    const IntegerVector& group,
    const List& adj_list,
    const IntegerVector& n_neighbors,
    double tau,
    double* removed_mean
) {
  int N = kappa.size();
  int J = adj_list.size();  // Number of spatial units
  NumericVector phi(J);

  // Accumulate data sufficient statistics by group
  NumericVector sum_omega(J);
  NumericVector sum_resid(J);

  for (int i = 0; i < N; i++) {
    int g = group[i] - 1;  // Convert to 0-based
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * offset[i];
  }

  // Single-site Gibbs updates for each spatial unit
  // Could be improved with block updates, but this is simple and works
  for (int j = 0; j < J; j++) {
    // Get neighbors - use eager copy to avoid GC issues with lazy views
    std::vector<int> neighbors = Rcpp::as<std::vector<int>>(adj_list[j]);
    int n_j = n_neighbors[j];

    // Sum of neighbor values
    double neighbor_sum = 0.0;
    for (int k = 0; k < n_j; k++) {
      int neighbor_idx = neighbors[k] - 1;  // Convert to 0-based
      neighbor_sum += phi[neighbor_idx];
    }

    // Full conditional parameters
    double prec = tau * n_j + sum_omega[j];
    double mean_num = tau * neighbor_sum + sum_resid[j];

    // Handle isolated units (no neighbors)
    if (n_j == 0) {
      // Fall back to exchangeable prior with the data
      if (sum_omega[j] > 0) {
        prec = sum_omega[j] + 0.001;  // Small prior precision
        mean_num = sum_resid[j];
      } else {
        phi[j] = 0.0;
        continue;
      }
    }

    double post_var = 1.0 / prec;
    double post_mean = mean_num / prec;

    phi[j] = R::rnorm(post_mean, std::sqrt(post_var));
  }

  // Center the spatial effects (sum-to-zero) and report the removed mean so the
  // caller can absorb it into the intercept -- eta is then unchanged and the
  // move is posterior-invariant (matching the negbin kernel). Discarding the
  // mean instead lags the intercept behind the field each sweep and drives tau.
  double mean_phi = 0.0;
  for (int j = 0; j < J; j++) {
    mean_phi += phi[j];
  }
  mean_phi /= J;

  for (int j = 0; j < J; j++) {
    phi[j] -= mean_phi;
  }
  if (removed_mean) *removed_mean = mean_phi;

  return phi;
}

// Update spatial precision tau with gamma prior
// ICAR precision structure: phi' Q phi where Q_ii = n_neighbors[i], Q_ij = -1 if neighbors
//
// Posterior for tau:
// tau | phi ~ Gamma(a + (J-1)/2, b + 0.5 * phi' Q phi)

double update_tau_icar(
    const NumericVector& phi,
    const List& adj_list,
    const IntegerVector& n_neighbors,
    double prior_shape,
    double prior_rate
) {
  int J = phi.size();

  // Compute phi' Q phi = sum_i n_i * phi_i^2 - 2 * sum_{i~j} phi_i * phi_j
  // = sum_i n_i * phi_i^2 - sum_i phi_i * sum_{j~i} phi_j
  double quad_form = 0.0;

  for (int i = 0; i < J; i++) {
    // Diagonal contribution: n_i * phi_i^2
    quad_form += n_neighbors[i] * phi[i] * phi[i];

    // Off-diagonal contribution: -2 * sum over neighbors (count each edge once)
    std::vector<int> neighbors = Rcpp::as<std::vector<int>>(adj_list[i]);
    for (int k = 0; k < n_neighbors[i]; k++) {
      int j = neighbors[k] - 1;
      if (j > i) {  // Count each edge once
        quad_form -= 2.0 * phi[i] * phi[j];
      }
    }
  }

  // ICAR rank is J - k for k connected components (one constant null direction
  // per component), so the shape is (J - k)/2, not (J - 1)/2. A disconnected
  // adjacency (spatial(by=) replication makes this routine) otherwise biases
  // tau upward. Component count by BFS on the (1-based) adjacency list, matching
  // the negbin kernel's convention.
  int k_comp = 0;
  {
    std::vector<int> seen(J, 0);
    std::vector<int> stack;
    for (int s0 = 0; s0 < J; s0++) {
      if (seen[s0]) continue;
      seen[s0] = 1; stack.clear(); stack.push_back(s0);
      while (!stack.empty()) {
        int s = stack.back(); stack.pop_back();
        std::vector<int> nb = Rcpp::as<std::vector<int>>(adj_list[s]);
        for (int e = 0; e < n_neighbors[s]; e++) {
          int t = nb[e] - 1;
          if (t >= 0 && t < J && !seen[t]) { seen[t] = 1; stack.push_back(t); }
        }
      }
      k_comp++;
    }
  }

  // Posterior parameters
  double post_shape = prior_shape + (J - k_comp) / 2.0;
  double post_rate = prior_rate + quad_form / 2.0;

  return R::rgamma(post_shape, 1.0 / post_rate);
}

// ---------------------------------------------------------------------
// BYM2 (scaled version of BYM)
// ---------------------------------------------------------------------

// BYM2 decomposes spatial effect as:
// u = sigma * (sqrt(rho) * phi_scaled * scale_factor + sqrt(1-rho) * theta)
// where:
//   phi_scaled is scaled ICAR (sum to zero, variance ~1)
//   theta is iid N(0,1)
//   sigma is total SD
//   rho is proportion of variance from structured component
//   scale_factor is computed from eigenvalues of Q

// Update BYM2 spatial effects
// Returns the combined effect u for each spatial unit
// Also updates phi_scaled and theta in place via references
NumericVector update_spatial_bym2(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& offset,
    const IntegerVector& group,
    const List& adj_list,
    const IntegerVector& n_neighbors,
    NumericVector& phi_scaled,  // Input/output: structured component
    NumericVector& theta,       // Input/output: unstructured component
    double sigma_spatial,
    double rho,
    double scale_factor
) {
  int N = kappa.size();
  int J = adj_list.size();

  // Precompute weights
  double sqrt_rho = std::sqrt(rho + 1e-10);
  double sqrt_1_rho = std::sqrt(1.0 - rho + 1e-10);

  // Accumulate data sufficient statistics by group
  NumericVector sum_omega(J);
  NumericVector sum_resid(J);

  for (int i = 0; i < N; i++) {
    int g = group[i] - 1;
    sum_omega[g] += omega[i];
    // Residual for the phi_scaled update removes only theta's contribution (the
    // OTHER component). Removing phi's own contribution too, as before, biases
    // the conditional mean toward zero (an artificial anti-autocorrelation).
    double theta_contrib = sigma_spatial * sqrt_1_rho * theta[g];
    sum_resid[g] += kappa[i] - omega[i] * (offset[i] + theta_contrib);
  }

  // Update phi_scaled (structured component with ICAR prior)
  // The effective precision for phi_scaled comes from both data and ICAR prior
  // phi_scaled has unit variance marginally, so ICAR precision is 1.0
  for (int j = 0; j < J; j++) {
    std::vector<int> neighbors = Rcpp::as<std::vector<int>>(adj_list[j]);
    int n_j = n_neighbors[j];

    // Sum of neighbor values
    double neighbor_sum = 0.0;
    for (int k = 0; k < n_j; k++) {
      int neighbor_idx = neighbors[k] - 1;
      neighbor_sum += phi_scaled[neighbor_idx];
    }

    // Effective coefficient in eta for phi_scaled
    double coef = sigma_spatial * sqrt_rho * scale_factor;

    // Full conditional for phi_scaled_j
    // Prior: ICAR with unit precision
    // Likelihood contribution: sum_i omega_i * (coef * phi_scaled_j)^2 - 2 * coef * sum_i (resid_i * phi_scaled_j)
    double prior_prec = (n_j > 0) ? n_j : 0.001;
    double data_prec = sum_omega[j] * coef * coef;

    double post_prec = prior_prec + data_prec;
    double post_mean_num = neighbor_sum + sum_resid[j] * coef;

    phi_scaled[j] = R::rnorm(post_mean_num / post_prec, std::sqrt(1.0 / post_prec));
  }

  // Center phi_scaled
  double mean_phi = 0.0;
  for (int j = 0; j < J; j++) {
    mean_phi += phi_scaled[j];
  }
  mean_phi /= J;
  for (int j = 0; j < J; j++) {
    phi_scaled[j] -= mean_phi;
  }

  // Recompute residuals for theta update
  for (int j = 0; j < J; j++) {
    sum_resid[j] = 0.0;
  }
  for (int i = 0; i < N; i++) {
    int g = group[i] - 1;
    // Residual for the theta update removes only phi's contribution.
    double phi_contrib = sigma_spatial * sqrt_rho * phi_scaled[g] * scale_factor;
    sum_resid[g] += kappa[i] - omega[i] * (offset[i] + phi_contrib);
  }

  // Update theta (unstructured component with N(0,1) prior)
  for (int j = 0; j < J; j++) {
    double coef = sigma_spatial * sqrt_1_rho;
    double prior_prec = 1.0;  // N(0,1) prior
    double data_prec = sum_omega[j] * coef * coef;

    double post_prec = prior_prec + data_prec;
    double post_mean_num = sum_resid[j] * coef;

    theta[j] = R::rnorm(post_mean_num / post_prec, std::sqrt(1.0 / post_prec));
  }

  // Compute combined spatial effect u
  NumericVector u(J);
  for (int j = 0; j < J; j++) {
    u[j] = sigma_spatial * (sqrt_rho * phi_scaled[j] * scale_factor + sqrt_1_rho * theta[j]);
  }

  return u;
}

// Update sigma_spatial with half-Cauchy prior — delegates to shared helper
double update_sigma_spatial(
    const NumericVector& u,
    double scale
) {
  return tulpa::update_sigma_halfcauchy(u, scale);
}

// Update rho (mixing proportion) with beta prior via a grid approximation of
// the Polya-Gamma full conditional.
double update_rho_bym2(
    const NumericVector& phi_scaled,
    const NumericVector& theta,
    double sigma_spatial,
    double scale_factor,
    const NumericVector& sum_omega,
    const NumericVector& sum_resid,
    double alpha,
    double beta
) {
  int J = phi_scaled.size();

  // Evaluate the log-posterior on a grid over rho in (0, 1) and sample.
  int n_grid = 20;
  NumericVector log_probs(n_grid);
  NumericVector rho_vals(n_grid);

  for (int k = 0; k < n_grid; k++) {
    double rho = (k + 0.5) / n_grid;  // Avoid exact 0 and 1
    rho_vals[k] = rho;

    double sqrt_rho = std::sqrt(rho + 1e-10);
    double sqrt_1_rho = std::sqrt(1.0 - rho + 1e-10);

    // Log-likelihood contribution
    // Polya-Gamma full conditional for rho through u_j(rho): the quadratic
    // data-precision term and the linear data-fit term. The phi_scaled (scaled
    // ICAR) and theta (iid) priors do not depend on rho, so there is no
    // log-determinant term.
    double log_lik = 0.0;
    for (int j = 0; j < J; j++) {
      double u_j = sigma_spatial * (sqrt_rho * phi_scaled[j] * scale_factor + sqrt_1_rho * theta[j]);
      log_lik += -0.5 * sum_omega[j] * u_j * u_j + sum_resid[j] * u_j;
    }

    // Beta prior: (alpha-1)*log(rho) + (beta-1)*log(1-rho)
    double log_prior = (alpha - 1.0) * std::log(rho + 1e-10) + (beta - 1.0) * std::log(1.0 - rho + 1e-10);

    log_probs[k] = log_lik + log_prior;
  }

  // Normalize and sample
  double max_log_prob = log_probs[0];
  for (int k = 1; k < n_grid; k++) {
    if (log_probs[k] > max_log_prob) max_log_prob = log_probs[k];
  }

  NumericVector probs(n_grid);
  double sum_probs = 0.0;
  for (int k = 0; k < n_grid; k++) {
    probs[k] = std::exp(log_probs[k] - max_log_prob);
    sum_probs += probs[k];
  }

  // Sample from discrete distribution
  double u = R::runif(0.0, sum_probs);
  double cumsum = 0.0;
  for (int k = 0; k < n_grid; k++) {
    cumsum += probs[k];
    if (u <= cumsum) {
      return rho_vals[k];
    }
  }

  return rho_vals[n_grid - 1];
}

} // namespace tulpa
