// hsgp_block_factory.h
// Build a LatentBlock for a Hilbert-space Gaussian-process (HSGP) field
// plugged into the joint multi-arm nested-Laplace driver.
//
// HSGP approximates a stationary GP via the eigendecomposition of the
// Laplacian on a bounded domain D:
//   f(x) = sum_j Phi_j(x) * sqrt(S(omega_j; sigma2, ell)) * beta_j
// where Phi_j are the Dirichlet eigenfunctions on D, omega_j the
// eigenvalues, and S the spectral density of the Matern (or squared-
// exponential) kernel. beta_j ~ N(0, 1) i.i.d.; sigma2 and ell are the
// outer-grid hyperparameters.
//
// HSGP uses DENSE_BASIS semantics. When the basis is wider than the pooled
// training rows, a QR factorization integrates its orthogonal Gaussian
// coordinates exactly. The block then carries one standard-normal coordinate
// per training row; prediction includes the independent orthogonal residual.
//
// Mesh data:
//   * eigenvalues: length-m_total double vector (shared across arms)
//   * phi_flat:    per-arm n_obs_k * m_total row-major matrix (per-arm
//                  basis evaluation at obs locations)
//
// Lifecycle:
//   * `prep(k_grid)` reads (sigma2, ell) from theta_grid, computes
//     sqrt_S[j] for j in 0..m_total and publishes it in a per-cell slot
//     (nl_cell_cache.h). basis_eval and the sparse scatter read their cell's
//     slot, so concurrent outer-grid cells never see each other's state.
//   * The prior on beta is N(0, I) -> diagonal precision; prior_kind = NONE,
//     no off-diagonal pattern entries.
//
// HSGP forces the sparse Newton path (dense scatter does not support
// DENSE_BASIS). add_prior is left empty; the dispatch in solve_at_theta_impl
// (1.4a) MUST route to the sparse path whenever any block has
// contrib_kind != INDEXED_SINGLE.
//
// Axis schema: (sigma2, lengthscale), on the scale the outer grid declares
// and reports them; every caller hands the theta_grid columns through as
// they are. PC priors on sigma2 and log-normal on lengthscale are applied
// OUTSIDE the inner Laplace, as part of the outer-grid hyper-prior, not in
// the block's log_prior. log_prior here is the N(0, I) on beta only.

#ifndef TULPA_HSGP_BLOCK_FACTORY_H
#define TULPA_HSGP_BLOCK_FACTORY_H

#include "latent_block.h"
#include "laplace_re_priors.h"  // center_effects (unused — HSGP doesn't center beta)
#include "nl_cell_cache.h"
#include "sparse_hessian.h"
#include "hsgp_spectral.h"
#include <Rcpp.h>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

void set_hsgp_standard_normal_prior(LatentBlock& block);

LatentBlock make_hsgp_block(
    int                            start,
    int                            m_total,
    const Rcpp::List&              phi_per_arm,            // List of NumericMatrix
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    const Rcpp::NumericVector&     eigenvalues,
    int                            axis_sigma2,
    int                            axis_ell,
    const Rcpp::NumericMatrix&     theta_grid,
    priors::HsgpKernel             kernel,
    Rcpp::Nullable<Rcpp::List>      training_phi = R_NilValue,
    bool                           observation_space = true
);

} // namespace tulpa

#endif // TULPA_HSGP_BLOCK_FACTORY_H
