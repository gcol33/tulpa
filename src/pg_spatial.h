// pg_spatial.h
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) and BYM2 priors

#ifndef TULPA_PG_SPATIAL_H
#define TULPA_PG_SPATIAL_H

#include <Rcpp.h>
#include <vector>

#include "bym2_mixing.h"  // bym2_log_rho / bym2_log1m_rho
#include "pg_shared.h"   // PgAdjacency

namespace tulpa {

// ---------------------------------------------------------------------
// ICAR (Intrinsic Conditional Autoregressive) prior
// ---------------------------------------------------------------------

// Update spatial effects with an ICAR prior by a single-site sweep.
//
// Full conditional for phi_j on a connected graph:
//   phi_j | rest ~ N(m_j, v_j)
//   v_j = 1 / (tau * n_j + sum_omega_j)
//   m_j = v_j * (tau * sum_{k ~ j} phi_k + sum_resid_j)
// and on a disconnected one the same with the augmented row of tau Qt
// (PgIcarPrior), which gives each component's level -- an island's value --
// the field's own precision. `phi` carries the previous sweep's field in and
// the new one out; the neighbour sum must read the current value of every
// neighbour, including those the sweep has not reached yet.
//
// After the sweep the k component levels are drawn jointly from their
// conditional under the constraint that the overall field level does not
// move, and the field is centred sum-to-zero. `removed_mean` reports the level
// removed by the centring so the caller can absorb it into the intercept,
// which leaves eta unchanged.
//
// @param kappa y - n/2
// @param omega PG draws
// @param offset X*beta + RE contribution (everything except spatial)
// @param group 1-based spatial unit of each observation
// @param prior The unweighted ICAR prior over the validated adjacency
// @param tau Spatial precision parameter
// @param beta0,prior_beta_sd The intercept and its prior SD: the field's level
//   is the intercept's, so each site conditional carries that prior through the
//   field mean (PgInterceptLevel).
void update_spatial_icar(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group,
    const PgIcarPrior& prior,
    double tau,
    double beta0,
    double prior_beta_sd,
    Rcpp::NumericVector& phi,
    double& removed_mean
);

// Update spatial precision tau with a gamma prior. The centred field's density
// is tau^((J - 1)/2) exp(-tau phi'Qt phi / 2) (PgIcarPrior), so
//   tau | phi ~ Gamma(shape + (J - 1)/2, rate + phi'Qt phi / 2).
double update_tau_icar(
    const Rcpp::NumericVector& phi,
    const PgIcarPrior& prior,
    double prior_shape,
    double prior_rate
);

// ---------------------------------------------------------------------
// BYM2 prior (scaled version of BYM)
// ---------------------------------------------------------------------

// BYM2 decomposes spatial effect as:
// u = sigma * (sqrt(rho) * phi_scaled * scale_factor + sqrt(1-rho) * theta)
// where phi_scaled is scaled ICAR and theta is iid N(0,1)

// Update BYM2 spatial effects.
// Writes the combined effect u for each spatial unit, and updates phi_scaled
// and theta in place.
void update_spatial_bym2(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group,
    // The structured component's ICAR prior, carrying the per-node precision
    // multipliers of the per-component BYM2 scaling beyond `scale_factor`
    // (gcol33/tulpa#902).
    const PgIcarPrior& prior,
    Rcpp::NumericVector& phi_scaled,  // Input/output: structured component
    Rcpp::NumericVector& theta,       // Input/output: unstructured component
    double sigma_spatial,             // Total spatial SD
    double rho,                       // Proportion of variance from structured component
    double scale_factor,              // BYM2 scaling factor (from eigenvalues)
    double beta0,                     // intercept the structured level shares
    double prior_beta_sd,             // its prior SD (PgInterceptLevel)
    Rcpp::NumericVector& u,           // out: combined spatial effect
    double& removed_mean              // out: field level removed by centering phi
);

// The BYM2 mixing weight is drawn on a fixed grid of cell midpoints over
// (0, 1), so its prior is the Beta(alpha, beta) density restricted to those
// nodes and renormalized there.
constexpr int PG_BYM2_RHO_GRID = 20;

inline double pg_bym2_rho_node(int k) {
  return (k + 0.5) / PG_BYM2_RHO_GRID;
}

// Unnormalized Beta(alpha, beta) log kernel at rho.
inline double pg_bym2_rho_log_kernel(double rho, double alpha, double beta) {
  return (alpha - 1.0) * bym2_log_rho(rho) + (beta - 1.0) * bym2_log1m_rho(rho);
}

// Log prior mass of a grid node rho under that restricted Beta.
double pg_bym2_rho_log_prior(double rho, double alpha, double beta);

// Update rho (mixing proportion) with beta prior
// Uses grid search approach
// rho ~ Beta(alpha, beta)
double update_rho_bym2(
    const Rcpp::NumericVector& phi_scaled,
    const Rcpp::NumericVector& theta,
    double sigma_spatial,
    double scale_factor,
    const Rcpp::NumericVector& sum_omega,
    const Rcpp::NumericVector& sum_resid,
    double alpha,
    double beta
);

// Update sigma_spatial from the BYM2 Polya-Gamma full conditional (Gaussian in
// sigma via the standardized field), half-normal prior.
double update_sigma_spatial_bym2(
    const Rcpp::NumericVector& phi_scaled,
    const Rcpp::NumericVector& theta,
    double rho,
    double scale_factor,
    const Rcpp::NumericVector& sum_omega,
    const Rcpp::NumericVector& sum_resid,
    double prior_scale
);

} // namespace tulpa

#endif // TULPA_PG_SPATIAL_H
