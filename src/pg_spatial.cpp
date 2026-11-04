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
    const PgIcarPrior& prior,
    double tau,
    double beta0,
    double prior_beta_sd,
    NumericVector& phi,
    double& removed_mean
) {
  const PgAdjacency& adj = prior.adj;
  const int N = kappa.size();
  const int J = adj.n;

  std::vector<double> sum_omega(J, 0.0), sum_resid(J, 0.0);
  pg_accumulate_stats(N, group.begin(), J, omega.begin(), kappa.begin(),
                      offset.begin(), sum_omega.data(), sum_resid.data());
  PgInterceptLevel level(beta0, 0.0, prior_beta_sd, 1.0, phi.begin(), J);
  PgIcarSweep sweep(prior, phi.begin());

  // Single-site sweep. Each neighbour sum reads the current phi, so a
  // neighbour the sweep has already reached contributes its new value and one
  // it has not contributes the previous sweep's.
  for (int j = 0; j < J; j++) {
    double neighbor_sum = 0.0;
    for (int e = adj.row_ptr[j]; e < adj.row_ptr[j + 1]; e++) {
      neighbor_sum += phi[adj.col_idx[e]];
    }

    double prior_prec = 0.0, prior_num = 0.0;
    sweep.site(j, phi[j], neighbor_sum, tau, prior_prec, prior_num);
    double prec = prior_prec + sum_omega[j];
    double mean_num = prior_num + sum_resid[j];
    level.add(phi[j], prec, mean_num);

    const double x_new = R::rnorm(mean_num / prec, std::sqrt(1.0 / prec));
    level.moved(phi[j], x_new);
    sweep.moved(j, phi[j], x_new);
    phi[j] = x_new;
  }

  // Component-level block update: delta_c shifts every unit of component c.
  // Q is invariant to it, so the conditional is the likelihood's,
  // N((B_c - C_c)/A_c, 1/A_c) with A_c = sum_omega, B_c = sum_resid and
  // C_c = sum_omega * phi over the component, times the augmentation's factor:
  // under sum_c n_c delta_c = 0 the weighted total does not move, and
  // (S_c + n_c delta_c)^2 / n_c adds precision tau n_c and -tau S_c to the
  // precision-weighted mean. The constraint holds the overall field level
  // fixed -- that direction is confounded with the intercept and is handled by
  // the centring below. Without this the k - 1 component contrasts move only
  // through the single-site sweep and mix at the rate of a random walk across
  // the components.
  const int k = adj.n_components;
  if (k > 1) {
    std::vector<double> A(k, 0.0), Bnum(k, 0.0), S(k, 0.0), d(k, 0.0);
    const std::vector<int>& n_c = prior.comp_size;
    for (int j = 0; j < J; j++) {
      const int c = adj.component[j];
      A[c] += sum_omega[j];
      Bnum[c] += sum_resid[j] - sum_omega[j] * phi[j];
      S[c] += phi[j];
    }
    for (int c = 0; c < k; c++) {
      A[c] += tau * n_c[c];
      Bnum[c] -= tau * S[c];
      d[c] = R::rnorm(Bnum[c] / A[c], std::sqrt(1.0 / A[c]));
    }
    // Condition on sum_c n_c delta_c = 0 (kriging correction against the
    // independent N(m_c, 1/A_c) draws).
    double aSa = 0.0, ad = 0.0;
    for (int c = 0; c < k; c++) {
      aSa += static_cast<double>(n_c[c]) * n_c[c] / A[c];
      ad += n_c[c] * d[c];
    }
    for (int c = 0; c < k; c++) d[c] -= (n_c[c] / A[c]) * ad / aSa;
    for (int j = 0; j < J; j++) phi[j] += d[adj.component[j]];
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
    const PgIcarPrior& prior,
    double tau,
    double prior_sigma_U,
    double prior_sigma_alpha
) {
  const double quad_form = prior.quad(phi.begin());

  double sigma = 1.0 / std::sqrt(tau);
  pg_update_scale_pc(quad_form, prior.rank(), prior_sigma_U,
                     prior_sigma_alpha, sigma);
  return 1.0 / (sigma * sigma);
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
    const PgIcarPrior& prior,
    NumericVector& phi_scaled,
    NumericVector& theta,
    double sigma_spatial,
    double rho,
    double scale_factor,
    double beta0,
    double prior_beta_sd,
    NumericVector& u,
    double& removed_mean
) {
  const PgAdjacency& adj = prior.adj;
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
  const double coef = sigma_spatial * sqrt_rho * scale_factor;
  PgInterceptLevel level(beta0, 0.0, prior_beta_sd, coef,
                         phi_scaled.begin(), J);
  PgIcarSweep sweep(prior, phi_scaled.begin());
  for (int j = 0; j < J; j++) {
    double neighbor_sum = 0.0;
    for (int e = adj.row_ptr[j]; e < adj.row_ptr[j + 1]; e++) {
      neighbor_sum += phi_scaled[adj.col_idx[e]];
    }

    // A node's row of the ICAR precision carries its component's multiplier,
    // and an island's augmented row is the unit structured variance the
    // per-component scaling gives it (PgIcarPrior).
    double prior_prec = 0.0, prior_num = 0.0;
    sweep.site(j, phi_scaled[j], neighbor_sum, 1.0, prior_prec, prior_num);
    const double data_prec = sum_omega[j] * coef * coef;

    double post_prec = prior_prec + data_prec;
    double post_mean_num = prior_num + sum_resid[j] * coef;
    level.add(phi_scaled[j], post_prec, post_mean_num);

    const double x_new =
        R::rnorm(post_mean_num / post_prec, std::sqrt(1.0 / post_prec));
    level.moved(phi_scaled[j], x_new);
    sweep.moved(j, phi_scaled[j], x_new);
    phi_scaled[j] = x_new;
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

// Continuous Beta-prior mixing-weight conditional, sampled on logit(rho).
double update_rho_bym2(
    const NumericVector& phi_scaled,
    const NumericVector& theta,
    double sigma_spatial,
    double scale_factor,
    const NumericVector& sum_omega,
    const NumericVector& sum_resid,
    double rho,
    double alpha,
    double beta
) {
  auto log_density = [&](double x) {
    const double r = x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
                              : std::exp(x) / (1.0 + std::exp(x));
    if (!(r > 0.0 && r < 1.0)) return R_NegInf;
    double lp = pg_bym2_rho_log_prior(r, alpha, beta) + std::log(r) + std::log1p(-r);
    const double sr = bym2_sd_structured(r);
    const double si = bym2_sd_unstructured(r);
    for (int j = 0; j < phi_scaled.size(); ++j) {
      const double u = sigma_spatial * (sr * phi_scaled[j] * scale_factor + si * theta[j]);
      lp += sum_resid[j] * u - 0.5 * sum_omega[j] * u * u;
    }
    return lp;
  };
  auto uniform = []() { return R::runif(0.0, 1.0); };
  const double x = slice_univariate(std::log(rho) - std::log1p(-rho), log_density, uniform);
  return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

double pg_bym2_rho_log_prior(double rho, double alpha, double beta) {
  return R::dbeta(rho, alpha, beta, 1);
}
// Update sigma_spatial from its Polya-Gamma full conditional given the
// standardized BYM2 field v_j = sqrt(rho)*phi_scaled_j*scale_factor +
// sqrt(1-rho)*theta_j, where u_j = sigma * v_j. The standardized field is
// fixed in this conditional. The PG likelihood is Gaussian in sigma:
//   exp(sigma * A - 0.5 * sigma^2 * B),  A = sum_j sum_resid_j v_j,
//                                        B = sum_j sum_omega_j v_j^2.
// The PC prior contributes -lambda sigma, giving a positive-truncated
// N((A-lambda)/B, 1/B) conditional.
double update_sigma_spatial_bym2(
    const NumericVector& phi_scaled,
    const NumericVector& theta,
    double rho,
    double scale_factor,
    const NumericVector& sum_omega,
    const NumericVector& sum_resid,
    double prior_sigma_U,
    double prior_sigma_alpha
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

  const double lambda = pc_rate(prior_sigma_U, prior_sigma_alpha);
  if (B == 0.0) return R::rexp(1.0 / lambda);
  return rtruncnorm_pos(A - lambda, B);
}

} // namespace tulpa
