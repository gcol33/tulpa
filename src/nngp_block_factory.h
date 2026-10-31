// nngp_block_factory.h
// Build a LatentBlock for a Nearest-Neighbor Gaussian Process (NNGP)
// spatial field plugged into the joint multi-arm nested-Laplace driver.
//
// NNGP approximates a Gaussian process via a sparse precision matrix
//   Λ = (I - A)' D⁻¹ (I - A)
// where A is lower-triangular with at most `nn` nonzeros per row (the
// regression coefficients on each focal site's `nn` nearest precedessors
// in the nn_order ordering) and D is diagonal (per-site conditional
// variance). The (A, D) pair is computed per outer-grid cell from
// (sigma2, phi_gp) using batch_nngp_scatter from gpu_nngp_laplace.h.
//
// NNGP uses INDEXED_SINGLE semantics: each obs maps to a single spatial
// unit via the per-arm spatial_idx vector; a caller reading the field through a
// projector replaces that gather (block_projector.h). The data scatter adds the
// usual β/spatial cross + spatial/spatial diagonal (handled by the
// shared scatter_arm_obs_joint_multi_sparse). The prior scatter
// (apply_nngp_full_prior_sparse) adds the off-diagonal coupling
// between each focal site and its `nn` neighbors — that's the
// NN_K-shaped sparsity that distinguishes NNGP from the diagonal-on-w
// approximation.
//
// State per outer-grid cell:
//   * alpha (n_spatial * nn): neighbor regression coefficients, indexed
//     by nn_order index (not raw site index). Rebuilt by prep(k_grid).
//   * cv    (n_spatial):       conditional variance per site (D diagonal).
//
// alpha and cv are owned by shared_ptr captured in the lambda closures so
// the LatentBlock can be copied/moved freely; prep() refreshes them in
// place. The block can run under any tier (Laplace, IMH-Laplace, NUTS)
// because the closures don't rely on R-side state.

#ifndef TULPA_NNGP_BLOCK_FACTORY_H
#define TULPA_NNGP_BLOCK_FACTORY_H

#include "gpu_nngp_laplace.h"
#include "latent_block.h"
#include "nl_cell_cache.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

LatentBlock make_nngp_block(
    int                            start,
    int                            n_spatial,
    const Rcpp::List&              spatial_idx_per_arm,
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    int                            nn,
    int                            cov_type,
    const Rcpp::NumericMatrix&     coords,
    const Rcpp::IntegerMatrix&     nn_idx,
    const Rcpp::NumericMatrix&     nn_dist,
    const Rcpp::IntegerVector&     nn_order,
    int                            axis_sigma2,
    int                            axis_phi_gp,
    const Rcpp::NumericMatrix&     theta_grid
);

} // namespace tulpa

#endif // TULPA_NNGP_BLOCK_FACTORY_H
