// pg_spatial.h
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) and BYM2 priors

#ifndef TULPA_PG_SPATIAL_H
#define TULPA_PG_SPATIAL_H

#include <Rcpp.h>
#include <vector>

#include "pg_shared.h"   // PgAdjacency

namespace tulpa {

// ---------------------------------------------------------------------
// ICAR (Intrinsic Conditional Autoregressive) prior
// ---------------------------------------------------------------------

// Exchangeable prior precision standing in for the ICAR prior on a unit with
// no neighbours. Such a unit is its own graph component: the ICAR prior says
// nothing about its level, so this is the weak proper prior that keeps its
// full conditional finite when the data alone would leave it flat.
constexpr double PG_ICAR_ISOLATED_PREC = 1e-3;

// Update spatial effects with an ICAR prior by a single-site sweep.
//
// Full conditional for phi_j:
//   phi_j | rest ~ N(m_j, v_j)
//   v_j = 1 / (tau * n_j + sum_omega_j)
//   m_j = v_j * (tau * sum_{k ~ j} phi_k + sum_resid_j)
// so `phi` carries the previous sweep's field in and the new one out; the
// neighbour sum must read the current value of every neighbour, including
// those the sweep has not reached yet.
//
// After the sweep the k component levels are drawn jointly from their
// likelihood conditional (the ICAR prior is invariant to a level shift within
// a component) under the constraint that the overall field level does not
// move, and the field is centred sum-to-zero. `removed_mean` reports the level
// removed by the centring so the caller can absorb it into the intercept,
// which leaves eta unchanged.
//
// @param kappa y - n/2
// @param omega PG draws
// @param offset X*beta + RE contribution (everything except spatial)
// @param group 1-based spatial unit of each observation
// @param adj Validated adjacency (CSR + component labels)
// @param tau Spatial precision parameter
void update_spatial_icar(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group,
    const PgAdjacency& adj,
    double tau,
    Rcpp::NumericVector& phi,
    double& removed_mean
);

// Update spatial precision tau with a gamma prior.
// The ICAR pseudo-density is tau^((J - k)/2) exp(-tau phi'Q phi / 2) for k
// graph components (one constant null direction each), so
//   tau | phi ~ Gamma(shape + (J - k)/2, rate + phi'Q phi / 2).
double update_tau_icar(
    const Rcpp::NumericVector& phi,
    const PgAdjacency& adj,
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
    const PgAdjacency& adj,
    Rcpp::NumericVector& phi_scaled,  // Input/output: structured component
    Rcpp::NumericVector& theta,       // Input/output: unstructured component
    double sigma_spatial,             // Total spatial SD
    double rho,                       // Proportion of variance from structured component
    double scale_factor,              // BYM2 scaling factor (from eigenvalues)
    Rcpp::NumericVector& u,           // out: combined spatial effect
    double& removed_mean              // out: field level removed by centering phi
);

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
