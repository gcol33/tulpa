// pg_spatial.cpp
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) prior for areal data

#include "bym2_mixing.h"
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

void update_spatial_icar(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& offset,
    const IntegerVector& group,
    const PgAdjacency& adj,
    double tau,
    NumericVector& phi,
    double& removed_mean
) {
  const int N = kappa.size();
  const int J = adj.n;

  std::vector<double> sum_omega(J, 0.0), sum_resid(J, 0.0);
  pg_accumulate_stats(N, group.begin(), J, omega.begin(), kappa.begin(),
                      offset.begin(), sum_omega.data(), sum_resid.data());

  // Single-site sweep. Each neighbour sum reads the current phi, so a
  // neighbour the sweep has already reached contributes its new value and one
  // it has not contributes the previous sweep's.
  for (int j = 0; j < J; j++) {
    const int n_j = adj.degree(j);

    double neighbor_sum = 0.0;
    for (int e = adj.row_ptr[j]; e < adj.row_ptr[j + 1]; e++) {
      neighbor_sum += phi[adj.col_idx[e]];
    }

    double prec = tau * n_j + sum_omega[j];
    double mean_num = tau * neighbor_sum + sum_resid[j];

    if (n_j == 0) {
      if (sum_omega[j] > 0) {
        prec = sum_omega[j] + PG_ICAR_ISOLATED_PREC;
        mean_num = sum_resid[j];
      } else {
        phi[j] = 0.0;
        continue;
      }
    }

    phi[j] = R::rnorm(mean_num / prec, std::sqrt(1.0 / prec));
  }

  // Component-level block update. Shifting every unit of one graph component
  // by a constant leaves the ICAR prior unchanged, so the level's conditional
  // comes from the likelihood alone: delta_c ~ N((B_c - C_c)/A_c, 1/A_c) with
  // A_c = sum_omega, B_c = sum_resid and C_c = sum_omega * phi over the
  // component. Drawn jointly under sum_c n_c delta_c = 0, which holds the
  // overall field level fixed -- that direction is confounded with the
  // intercept and is handled by the centring below. Without this the k - 1
  // component contrasts move only through the single-site sweep and mix at the
  // rate of a random walk across the components.
  const int k = adj.n_components;
  if (k > 1) {
    std::vector<double> A(k, 0.0), Bnum(k, 0.0), d(k, 0.0);
    std::vector<int> n_c(k, 0);
    for (int j = 0; j < J; j++) {
      const int c = adj.component[j];
      A[c] += sum_omega[j];
      Bnum[c] += sum_resid[j] - sum_omega[j] * phi[j];
      n_c[c]++;
    }
    bool ok = true;
    for (int c = 0; c < k; c++) if (!(A[c] > 0.0)) ok = false;
    if (ok) {
      for (int c = 0; c < k; c++) {
        d[c] = R::rnorm(Bnum[c] / A[c], std::sqrt(1.0 / A[c]));
      }
      // Condition on sum_c n_c delta_c = 0 (kriging correction against the
      // independent N(m_c, 1/A_c) draws).
      double aSa = 0.0, ad = 0.0;
      for (int c = 0; c < k; c++) {
        aSa += static_cast<double>(n_c[c]) * n_c[c] / A[c];
        ad += n_c[c] * d[c];
      }
      if (aSa > 0.0) {
        for (int c = 0; c < k; c++) d[c] -= (n_c[c] / A[c]) * ad / aSa;
      }
      for (int j = 0; j < J; j++) phi[j] += d[adj.component[j]];
    }
  }

  // Centre the spatial effects (sum-to-zero) and report the removed mean so the
  // caller can absorb it into the intercept -- eta is then unchanged and the
  // move is posterior-invariant. Discarding the mean instead lags the intercept
  // behind the field each sweep and drives tau.
  double mean_phi = 0.0;
  for (int j = 0; j < J; j++) mean_phi += phi[j];
  mean_phi /= J;

  for (int j = 0; j < J; j++) phi[j] -= mean_phi;
  removed_mean = mean_phi;
}

double update_tau_icar(
    const NumericVector& phi,
    const PgAdjacency& adj,
    double prior_shape,
    double prior_rate
) {
  const int J = adj.n;

  // phi' Q phi = sum_i n_i phi_i^2 - 2 sum_{i ~ j, j > i} phi_i phi_j
  double quad_form = 0.0;
  for (int i = 0; i < J; i++) {
    quad_form += adj.degree(i) * phi[i] * phi[i];
    for (int e = adj.row_ptr[i]; e < adj.row_ptr[i + 1]; e++) {
      const int j = adj.col_idx[e];
      if (j > i) quad_form -= 2.0 * phi[i] * phi[j];
    }
  }

  const double post_shape = prior_shape + (J - adj.n_components) / 2.0;
  const double post_rate = prior_rate + quad_form / 2.0;

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

void update_spatial_bym2(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& offset,
    const IntegerVector& group,
    const PgAdjacency& adj,
    NumericVector& phi_scaled,
    NumericVector& theta,
    double sigma_spatial,
    double rho,
    double scale_factor,
    NumericVector& u,
    double& removed_mean
) {
  const int N = kappa.size();
  const int J = adj.n;

  const double sqrt_rho = bym2_sd_structured(rho);
  const double sqrt_1_rho = bym2_sd_unstructured(rho);

  std::vector<double> sum_omega(J, 0.0), sum_resid(J, 0.0);
  std::vector<double> work_offset(N);

  // Residual for the phi_scaled update removes only theta's contribution (the
  // OTHER component); removing phi's own contribution too biases the
  // conditional mean toward zero.
  for (int i = 0; i < N; i++) {
    const int g = group[i] - 1;
    work_offset[i] = offset[i] + sigma_spatial * sqrt_1_rho * theta[g];
  }
  pg_accumulate_stats(N, group.begin(), J, omega.begin(), kappa.begin(),
                      work_offset.data(), sum_omega.data(), sum_resid.data());

  // Update phi_scaled (structured component with ICAR prior). phi_scaled has
  // unit marginal variance, so the ICAR precision is 1.
  for (int j = 0; j < J; j++) {
    const int n_j = adj.degree(j);

    double neighbor_sum = 0.0;
    for (int e = adj.row_ptr[j]; e < adj.row_ptr[j + 1]; e++) {
      neighbor_sum += phi_scaled[adj.col_idx[e]];
    }

    const double coef = sigma_spatial * sqrt_rho * scale_factor;
    const double prior_prec = (n_j > 0) ? n_j : PG_ICAR_ISOLATED_PREC;
    const double data_prec = sum_omega[j] * coef * coef;

    const double post_prec = prior_prec + data_prec;
    const double post_mean_num = neighbor_sum + sum_resid[j] * coef;

    phi_scaled[j] = R::rnorm(post_mean_num / post_prec, std::sqrt(1.0 / post_prec));
  }

  // Centre phi_scaled (sum-to-zero on the structured component) and report the
  // field level removed from u_j = sigma*(sqrt_rho*phi_scaled_j*scale + ...),
  // so the caller can absorb it into the intercept -- eta is then unchanged.
  double mean_phi = 0.0;
  for (int j = 0; j < J; j++) mean_phi += phi_scaled[j];
  mean_phi /= J;
  for (int j = 0; j < J; j++) phi_scaled[j] -= mean_phi;
  removed_mean = sigma_spatial * sqrt_rho * scale_factor * mean_phi;

  // Residual for the theta update removes only phi's contribution.
  for (int i = 0; i < N; i++) {
    const int g = group[i] - 1;
    work_offset[i] = offset[i] +
        sigma_spatial * sqrt_rho * phi_scaled[g] * scale_factor;
  }
  pg_accumulate_stats(N, group.begin(), J, omega.begin(), kappa.begin(),
                      work_offset.data(), sum_omega.data(), sum_resid.data());

  // Update theta (unstructured component with N(0,1) prior)
  for (int j = 0; j < J; j++) {
    const double coef = sigma_spatial * sqrt_1_rho;
    const double prior_prec = 1.0;
    const double data_prec = sum_omega[j] * coef * coef;

    const double post_prec = prior_prec + data_prec;
    const double post_mean_num = sum_resid[j] * coef;

    theta[j] = R::rnorm(post_mean_num / post_prec, std::sqrt(1.0 / post_prec));
  }

  for (int j = 0; j < J; j++) {
    u[j] = sigma_spatial * (sqrt_rho * phi_scaled[j] * scale_factor +
                            sqrt_1_rho * theta[j]);
  }
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

    double sqrt_rho = bym2_sd_structured(rho);
    double sqrt_1_rho = bym2_sd_unstructured(rho);

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
    double log_prior = (alpha - 1.0) * bym2_log_rho(rho) + (beta - 1.0) * bym2_log1m_rho(rho);

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

// Update sigma_spatial from its Polya-Gamma full conditional given the
// standardized BYM2 field v_j = sqrt(rho)*phi_scaled_j*scale_factor +
// sqrt(1-rho)*theta_j, where u_j = sigma * v_j. The convolution u is
// deterministic in (phi_scaled, theta, sigma, rho), so sigma's conditional
// flows through the likelihood (as the rho update does), NOT through an iid
// half-Cauchy on u. The PG likelihood is Gaussian in sigma:
//   exp(sigma * A - 0.5 * sigma^2 * B),  A = sum_j sum_resid_j v_j,
//                                        B = sum_j sum_omega_j v_j^2.
// A half-normal N+(0, scale^2) prior adds precision 1/scale^2 and mean 0, so
// the posterior is N(A/(B + 1/scale^2), 1/(B + 1/scale^2)) truncated to
// sigma > 0.
double update_sigma_spatial_bym2(
    const NumericVector& phi_scaled,
    const NumericVector& theta,
    double rho,
    double scale_factor,
    const NumericVector& sum_omega,
    const NumericVector& sum_resid,
    double prior_scale
) {
  int J = phi_scaled.size();
  double sqrt_rho = bym2_sd_structured(rho);
  double sqrt_1_rho = bym2_sd_unstructured(rho);

  double A = 0.0, B = 0.0;
  for (int j = 0; j < J; j++) {
    double v_j = sqrt_rho * phi_scaled[j] * scale_factor + sqrt_1_rho * theta[j];
    A += sum_resid[j] * v_j;
    B += sum_omega[j] * v_j * v_j;
  }

  double prior_prec = (prior_scale > 0.0) ? 1.0 / (prior_scale * prior_scale) : 0.0;
  double post_prec = B + prior_prec;
  if (post_prec <= 1e-12) post_prec = 1e-12;
  double post_mean = A / post_prec;
  double post_sd = 1.0 / std::sqrt(post_prec);

  return rtruncnorm_pos(post_mean, post_sd);
}

} // namespace tulpa
