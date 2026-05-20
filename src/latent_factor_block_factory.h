// latent_factor_block_factory.h
// Build a LatentBlock for a F=1 latent factor field shared across arms,
// with per-arm loadings jointly optimized in the latent vector (Stage 1.6a).
//
// Model.
//   eta_i (in arm k_arm) += u[obs_idx_per_arm[k_arm](i)] * lambda[k_arm]
//
// where
//   * u in R^n_latent is the shared factor field
//   * lambda in R^n_arms holds the per-arm loadings
//
// Both u and lambda live in the joint latent vector and are co-optimized
// by the inner Newton solve. The eta contribution is BILINEAR in x — see
// BlockContribKind::BILINEAR_FACTOR in latent_block.h for the scatter and
// eta-accumulator dispatch shape.
//
// Block storage layout:
//   x[start ... start + n_latent)              ->  u_1, ..., u_n_latent
//   x[start + n_latent ... start + size)       ->  lambda_1, ..., lambda_n_arms
// where size = n_latent + n_arms.
//
// Identifiability. A bilinear (u, lambda) product has two degeneracies:
//   (a) overall sign: (u, lambda) and (-u, -lambda) give identical eta.
//   (b) overall scale: (c*u, lambda/c) and (u/c, c*lambda) give identical
//       eta for any c != 0.
// We pin both via tight Gaussian priors that act as soft anchors:
//   * u_1     ~ N(0, eps^2)             (eps = anchor_eps, default 1e-3)
//   * lambda_1 ~ N(1, eps^2)            (same anchor_eps)
// The remaining latents carry the user-facing priors:
//   * u_j     ~ N(0, sigma_u^2)         (sigma_u defaults to 1.0)
//   * lambda_k ~ N(0, sigma_lambda^2)   (sigma_lambda defaults to 1.0)
//
// Soft anchoring (tight prior) instead of hard reparameterization keeps
// the block storage layout uniform across all (u, lambda) slots and lets
// the same BILINEAR_FACTOR scatter handle every arm × every obs without
// dispatching on "is this slot pinned?" — much simpler than a true
// reparam at the cost of two extra well-conditioned diagonal entries.
//
// First-ship scope. F = 1 factor. Multi-factor F > 1 deferred until a
// downstream workload requests it (rotation identifiability on the F x F
// loading matrix needs additional pinning beyond the F = 1 case here).

#ifndef TULPA_LATENT_FACTOR_BLOCK_FACTORY_H
#define TULPA_LATENT_FACTOR_BLOCK_FACTORY_H

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

// Per-arm obs -> factor-field index lookup. Built from R-side
// `obs_idx` (1-based; <=0 means "this obs does not see the factor").
using PerArmIdxFn = std::function<int(int /*i*/, int /*k_arm*/)>;

inline LatentBlock make_latent_factor_block(
    int                                  start,
    int                                  n_latent,
    int                                  n_arms,
    const PerArmIdxFn&                   obs_idx,
    double                               sigma_u,
    double                               sigma_lambda,
    double                               anchor_eps,
    int                                  block_index
) {
    if (n_latent < 2) {
        Rcpp::stop("Block %d (type 'lf'): `n_latent` must be at least 2 "
                   "(got %d).", block_index + 1, n_latent);
    }
    if (n_arms < 1) {
        Rcpp::stop("Block %d (type 'lf'): n_arms must be >= 1 (got %d).",
                   block_index + 1, n_arms);
    }
    if (!(sigma_u > 0.0) || !(sigma_lambda > 0.0) || !(anchor_eps > 0.0)) {
        Rcpp::stop("Block %d (type 'lf'): sigma_u, sigma_lambda, and "
                   "anchor_eps must all be positive (got %g, %g, %g).",
                   block_index + 1, sigma_u, sigma_lambda, anchor_eps);
    }

    const int u_start      = start;
    const int lambda_start = start + n_latent;
    const int size         = n_latent + n_arms;

    // Precision parameters for the prior diagonal entries.
    const double prec_u_anchor    = 1.0 / (anchor_eps * anchor_eps);
    const double prec_u_free      = 1.0 / (sigma_u * sigma_u);
    const double prec_l_anchor    = 1.0 / (anchor_eps * anchor_eps);
    const double prec_l_free      = 1.0 / (sigma_lambda * sigma_lambda);

    LatentBlock block;
    block.start = start;
    block.size  = size;
    block.contrib_kind = BlockContribKind::BILINEAR_FACTOR;
    block.prior_kind   = PriorFillKind::NONE;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale, idx, basis_eval, obs_indices intentionally left empty —
    // scatter dispatches on BILINEAR_FACTOR and reads obs_factor_lambda
    // below.

    block.obs_factor_lambda = [obs_idx, u_start, lambda_start, n_latent](
        int i, int k_arm
    ) -> std::pair<int,int> {
        int l = obs_idx(i, k_arm);
        if (l <= 0 || l > n_latent) return {-1, -1};
        return {u_start + (l - 1), lambda_start + k_arm};
    };

    // Sparse-path prior scatter: diagonal Gaussian on each slot, tight
    // anchor on (u_1, lambda_1). Off-diagonal entries are not used (prior
    // precision is diagonal); pattern builder adds the diagonal via the
    // always-present block-diagonal pass.
    block.add_prior_sparse = [u_start, lambda_start, n_latent, n_arms,
                               prec_u_anchor, prec_u_free,
                               prec_l_anchor, prec_l_free](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) {
        // u_1 anchor: N(0, eps^2). u_j (j > 1): N(0, sigma_u^2).
        for (int j = 0; j < n_latent; j++) {
            int idx = u_start + j;
            double prec = (j == 0) ? prec_u_anchor : prec_u_free;
            grad[idx] -= prec * x[idx];
            H.add(idx, idx, prec);
        }
        // lambda_1 anchor: N(1, eps^2). lambda_k (k > 0): N(0, sigma_lambda^2).
        for (int k = 0; k < n_arms; k++) {
            int idx = lambda_start + k;
            if (k == 0) {
                grad[idx] -= prec_l_anchor * (x[idx] - 1.0);
                H.add(idx, idx, prec_l_anchor);
            } else {
                grad[idx] -= prec_l_free * x[idx];
                H.add(idx, idx, prec_l_free);
            }
        }
    };

    block.log_prior = [u_start, lambda_start, n_latent, n_arms,
                        prec_u_anchor, prec_u_free,
                        prec_l_anchor, prec_l_free](
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) -> double {
        double lp = 0.0;
        // u priors.
        for (int j = 0; j < n_latent; j++) {
            double v = x[u_start + j];
            double prec = (j == 0) ? prec_u_anchor : prec_u_free;
            lp += -0.5 * prec * v * v + 0.5 * std::log(prec) -
                  0.5 * std::log(2.0 * M_PI);
        }
        // lambda priors. lambda_1 centered at 1, others at 0.
        for (int k = 0; k < n_arms; k++) {
            double v = x[lambda_start + k];
            double mu = (k == 0) ? 1.0 : 0.0;
            double prec = (k == 0) ? prec_l_anchor : prec_l_free;
            double diff = v - mu;
            lp += -0.5 * prec * diff * diff + 0.5 * std::log(prec) -
                  0.5 * std::log(2.0 * M_PI);
        }
        return lp;
    };

    // No centering: the soft anchors on u_1 and lambda_1 break both
    // identifiability ambiguities. Newton step on the well-conditioned
    // joint H finds the unique mode.

    return block;
}

} // namespace tulpa

#endif // TULPA_LATENT_FACTOR_BLOCK_FACTORY_H
