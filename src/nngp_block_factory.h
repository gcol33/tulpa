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
// unit via the per-arm spatial_idx vector. The data scatter adds the
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
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

inline LatentBlock make_nngp_block(
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
) {
    if (static_cast<int>(spatial_idx_per_arm.size()) != n_arms ||
        n_obs_per_arm.size() != n_arms) {
        Rcpp::stop("Block %d (type 'nngp'): spatial_idx_per_arm / "
                   "n_obs_per_arm must each have length n_arms (%d).",
                   block_index + 1, n_arms);
    }
    if (coords.nrow() != n_spatial) {
        Rcpp::stop("Block %d (type 'nngp'): nrow(coords) (%d) must equal "
                   "n_spatial (%d).",
                   block_index + 1, static_cast<int>(coords.nrow()),
                   n_spatial);
    }
    if (nn_idx.nrow() != n_spatial || nn_idx.ncol() != nn) {
        Rcpp::stop("Block %d (type 'nngp'): nn_idx must be n_spatial x nn "
                   "(%d x %d), got %d x %d.",
                   block_index + 1, n_spatial, nn,
                   static_cast<int>(nn_idx.nrow()),
                   static_cast<int>(nn_idx.ncol()));
    }
    if (nn_order.size() != n_spatial) {
        Rcpp::stop("Block %d (type 'nngp'): length(nn_order) (%d) must equal "
                   "n_spatial (%d).",
                   block_index + 1, static_cast<int>(nn_order.size()),
                   n_spatial);
    }

    // Materialize per-arm spatial_idx vectors at factory time.
    auto sidx_per_arm =
        std::make_shared<std::vector<Rcpp::IntegerVector>>(n_arms);
    for (int k = 0; k < n_arms; k++) {
        Rcpp::IntegerVector sidx = spatial_idx_per_arm[k];
        int N_k = n_obs_per_arm[k];
        if (static_cast<int>(sidx.size()) != N_k) {
            Rcpp::stop("Block %d (type 'nngp'): spatial_idx_per_arm[[%d]] "
                       "must have length n_obs_per_arm[%d] (expected %d, "
                       "got %d).",
                       block_index + 1, k + 1, k + 1, N_k,
                       static_cast<int>(sidx.size()));
        }
        (*sidx_per_arm)[k] = sidx;
    }

    // Per-cell NNGP state. cm is not used outside batch_nngp_scatter (the
    // prior scatter recomputes the residual from w directly) but the
    // function signature demands it.
    auto alpha = std::make_shared<std::vector<double>>(
        static_cast<size_t>(n_spatial) * nn, 0.0);
    auto cv    = std::make_shared<std::vector<double>>(n_spatial, 0.0);
    auto cm    = std::make_shared<std::vector<double>>(n_spatial, 0.0);

    LatentBlock block;
    block.start = start;
    block.size  = n_spatial;
    block.contrib_kind = BlockContribKind::INDEXED_SINGLE;
    block.prior_kind   = PriorFillKind::NN_K;
    block.d_fac        = [](int) -> double { return 1.0; };

    block.idx = [sidx_per_arm](int i, int k_arm) -> int {
        const Rcpp::IntegerVector& sidx = (*sidx_per_arm)[k_arm];
        if (i < 0 || i >= static_cast<int>(sidx.size())) return -1;
        return static_cast<int>(sidx[i]);
    };

    // Rebuild (alpha, cv) for cell k. batch_nngp_scatter is well-defined on
    // w = 0 (alpha and cv are functions of (coords, nn_idx, nn_dist, sigma2,
    // phi_gp, cov_type) only; cm carries the w-dependence and we ignore it).
    block.prep = [alpha, cv, cm, n_spatial, nn, cov_type, coords, nn_idx,
                   nn_dist, nn_order, axis_sigma2, axis_phi_gp,
                   theta_grid](int k_grid) -> bool {
        double sigma2 = theta_grid(k_grid, axis_sigma2);
        double phi_gp = theta_grid(k_grid, axis_phi_gp);
        if (!(sigma2 > 0.0) || !(phi_gp > 0.0)) return false;
        std::vector<double> w_zero(n_spatial, 0.0);
        bool gpu_used = false;
        batch_nngp_scatter(w_zero, n_spatial, nn, sigma2, phi_gp, cov_type,
                            coords, nn_idx, nn_dist, nn_order,
                            *cm, *cv, gpu_used, alpha.get());
        return true;
    };

    // Pattern: full NN_K-shaped fill, normalized to lower triangle by the
    // builder's init().
    block.add_prior_pattern = [start, n_spatial, nn, nn_idx, nn_order](
        std::vector<std::pair<int,int>>& out
    ) {
        make_nngp_prior_sparsity_pattern(out, nn_idx, nn_order,
                                          n_spatial, nn, start);
    };

    block.add_prior_sparse = [alpha, cv, start, n_spatial, nn, nn_idx,
                               nn_order](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) {
        std::vector<double> w(n_spatial);
        for (int s = 0; s < n_spatial; s++) w[s] = x[start + s];
        apply_nngp_full_prior_sparse(grad, H, w, *alpha, *cv,
                                      nn_idx, nn_order, n_spatial, nn, start);
    };

    block.log_prior = [alpha, cv, start, n_spatial, nn, nn_idx, nn_order](
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) -> double {
        // Conditional Gaussian log-density across the NNGP-ordered chain:
        //   log p(w_focal | w_neighbors) = -0.5*log(2*pi*cv_i)
        //                                  - 0.5*(w_focal - sum_k alpha_k * w_k)^2 / cv_i
        // summed over the n_spatial sites in nn_order. Matches the legacy
        // formula in cpp_nested_laplace_nngp.
        double lp = 0.0;
        const double* alpha_data = (*alpha).data();
        for (int i_nngp = 0; i_nngp < n_spatial; i_nngp++) {
            int obs_focal = nn_order[i_nngp];
            if (obs_focal < 0 || obs_focal >= n_spatial) continue;
            double v_i = (*cv)[obs_focal];
            if (!(v_i > 0.0)) continue;
            double w_focal = x[start + obs_focal];
            double resid = w_focal;
            const double* arow = alpha_data + static_cast<size_t>(i_nngp) * nn;
            for (int k = 0; k < nn; k++) {
                int nnidx_k = nn_idx(i_nngp, k);
                if (nnidx_k <= 0 || nnidx_k > n_spatial) continue;
                int obs_k = nn_order[nnidx_k - 1];
                if (obs_k < 0 || obs_k >= n_spatial) continue;
                resid -= arow[k] * x[start + obs_k];
            }
            lp += -0.5 * std::log(2.0 * M_PI * v_i)
                  - 0.5 * resid * resid / v_i;
        }
        return lp;
    };

    // No centering — NNGP latent has no sum-to-zero constraint (the
    // ordering anchors it via the chain).

    return block;
}

} // namespace tulpa

#endif // TULPA_NNGP_BLOCK_FACTORY_H
