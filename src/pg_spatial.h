// pg_spatial.h
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) and BYM2 priors

#ifndef TULPA_PG_SPATIAL_H
#define TULPA_PG_SPATIAL_H

#include <Rcpp.h>
#include <vector>

namespace tulpa {

// ---------------------------------------------------------------------
// ICAR (Intrinsic Conditional Autoregressive) prior
// ---------------------------------------------------------------------

// Update spatial effects with ICAR prior
// phi_i | phi_{-i} ~ N(mean of neighbors, 1/(n_neighbors * tau))
//
// With PG augmentation:
// kappa_i/omega_i = X_beta_i + re_i + phi_i + error
//
// @param kappa y - n/2
// @param omega PG draws
// @param offset X*beta + RE contribution (everything except spatial)
// @param adj_list Adjacency list: adj_list[i] contains indices of neighbors of i
// @param n_neighbors Number of neighbors for each location
// @param tau Spatial precision parameter
// @return Updated spatial effects
Rcpp::NumericVector update_spatial_icar(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group,       // Group index for each observation
    const Rcpp::List& adj_list,             // List of neighbor indices
    const Rcpp::IntegerVector& n_neighbors, // Number of neighbors per location
    double tau,
    double* removed_mean = nullptr          // out: the sum-to-zero mean removed
);

// Update spatial precision tau with gamma prior
// tau ~ Gamma(a, b)
// Posterior: tau | phi ~ Gamma(a + (J-1)/2, b + 0.5 * sum(phi * Q * phi))
// where Q is the precision matrix of ICAR
double update_tau_icar(
    const Rcpp::NumericVector& phi,
    const Rcpp::List& adj_list,
    const Rcpp::IntegerVector& n_neighbors,
    double prior_shape,
    double prior_rate
);

// ---------------------------------------------------------------------
// BYM2 prior (scaled version of BYM)
// ---------------------------------------------------------------------

// BYM2 decomposes spatial effect as:
// u = sigma * (sqrt(rho) * phi_scaled * scale_factor + sqrt(1-rho) * theta)
// where phi_scaled is scaled ICAR and theta is iid N(0,1)
//
// This gives better interpretability and mixing

// Update BYM2 spatial effects
// Returns the combined effect u for each spatial unit
// Also updates phi_scaled and theta in place via references
Rcpp::NumericVector update_spatial_bym2(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group,
    const Rcpp::List& adj_list,
    const Rcpp::IntegerVector& n_neighbors,
    Rcpp::NumericVector& phi_scaled,  // Input/output: structured component
    Rcpp::NumericVector& theta,       // Input/output: unstructured component
    double sigma_spatial,             // Total spatial SD
    double rho,                       // Proportion of variance from structured component
    double scale_factor,              // BYM2 scaling factor (from eigenvalues)
    double* removed_mean = nullptr    // out: field level removed by centering phi
);

// Update sigma_spatial with half-Cauchy prior
double update_sigma_spatial(
    const Rcpp::NumericVector& u,  // Total spatial effect
    double scale
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
// sigma via the standardized field), half-normal prior. Replaces the iid
// half-Cauchy update on the deterministic convolution u.
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
