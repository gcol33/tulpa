// latent_factor_block_factory.h
// Build a LatentBlock for one latent factor shared across arms, with per-arm
// loadings in the joint latent vector.
//
// Model.
//   eta_i (in arm k_arm) += u[obs_idx_per_arm[k_arm](i)] * lambda[k_arm]
//
// where
//   * u in R^n_latent is the factor's score at each unit, u_l ~ N(0, 1)
//   * lambda holds the loading of each arm k_arm >= first_arm
//
// Block storage layout:
//   x[start ... start + n_latent)        ->  u_1, ..., u_n_latent
//   x[start + n_latent ... start + size) ->  lambda_first_arm, ..., lambda_{n_arms-1}
// where size = n_latent + (n_arms - first_arm).
//
// Identifiability. The product u_l * lambda_k is unchanged by
//   (a) a common rescaling (c u, lambda / c): fixed by the unit variance of
//       the score prior, so the loadings carry the factor's scale;
//   (b) the sign flip (-u, -lambda): fixed by taking the loading of the
//       factor's first arm positive. The prior on that loading is the
//       half-normal 2 N(0, sigma_lambda^2) on (0, Inf), so the reported mode is
//       the one of the two mirror-image modes with lambda_first_arm > 0, and
//       the factor 2 is the mass of the mirror mode;
//   (c) for several factors, a rotation of the score / loading matrices:
//       fixed by the lower-triangular loading matrix, factor f (0-based) not
//       loading on arms 0, ..., f - 1 (`first_arm = f`). Arms below
//       `first_arm` carry no loading slot at all.
// A level shift u -> u + a moves arm k's predictor by a * lambda_k, which the
// arm's intercept absorbs; the proper N(0, 1) prior on u carries that
// direction, so nothing is pinned for it.
//
// The scores are not estimated jointly with the loadings: the inner solve
// integrates them out by a Laplace approximation (latent_factor_marginal.h),
// since the joint mode of a bilinear posterior shrinks the scores and inflates
// the loadings by the score variance it ignores.

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
    int                                  first_arm,
    const PerArmIdxFn&                   obs_idx,
    double                               sigma_lambda,
    int                                  block_index
) {
    if (n_latent < 2) {
        Rcpp::stop("Block %d (type 'lf'): `n_latent` must be at least 2 "
                   "(got %d).", block_index + 1, n_latent);
    }
    if (first_arm < 0 || first_arm >= n_arms) {
        Rcpp::stop("Block %d (type 'lf'): the factor's first arm must be one "
                   "of the %d arms (got %d, 0-based).", block_index + 1,
                   n_arms, first_arm);
    }
    if (!(sigma_lambda > 0.0)) {
        Rcpp::stop("Block %d (type 'lf'): sigma_lambda must be positive "
                   "(got %g).", block_index + 1, sigma_lambda);
    }

    const int u_start      = start;
    const int lambda_start = start + n_latent;
    const int n_loadings   = n_arms - first_arm;
    const int size         = n_latent + n_loadings;
    const double prec_l    = 1.0 / (sigma_lambda * sigma_lambda);

    LatentBlock block;
    block.start = start;
    block.size  = size;
    block.contrib_kind = BlockContribKind::BILINEAR_FACTOR;
    block.prior_kind   = PriorFillKind::NONE;
    block.d_fac        = [](int) -> double { return 1.0; };
    block.factor_n_latent = n_latent;
    block.factor_first_arm = first_arm;

    block.obs_factor_lambda = [obs_idx, u_start, lambda_start, n_latent,
                               n_arms, first_arm](
        int i, int k_arm
    ) -> std::pair<int,int> {
        if (k_arm < first_arm || k_arm >= n_arms) return {-1, -1};
        int l = obs_idx(i, k_arm);
        if (l <= 0 || l > n_latent) return {-1, -1};
        return {u_start + (l - 1), lambda_start + (k_arm - first_arm)};
    };

    // Diagonal Gaussian prior: unit precision on every score, 1 / sigma_lambda^2
    // on every loading.
    block.add_prior_sparse = [u_start, lambda_start, n_latent, n_loadings,
                               prec_l](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) {
        for (int j = 0; j < n_latent; j++) {
            const int idx = u_start + j;
            grad[idx] -= x[idx];
            H.add(idx, idx, 1.0);
        }
        for (int k = 0; k < n_loadings; k++) {
            const int idx = lambda_start + k;
            grad[idx] -= prec_l * x[idx];
            H.add(idx, idx, prec_l);
        }
    };

    // log p(u) + log p(lambda): standard normal scores, N(0, sigma_lambda^2)
    // loadings, the first loading's density doubled on its half-line.
    block.log_prior = [u_start, lambda_start, n_latent, n_loadings, prec_l](
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) -> double {
        const double half_log_2pi = 0.5 * std::log(2.0 * M_PI);
        double lp = 0.0;
        for (int j = 0; j < n_latent; j++) {
            const double v = x[u_start + j];
            lp += -0.5 * v * v - half_log_2pi;
        }
        for (int k = 0; k < n_loadings; k++) {
            const double v = x[lambda_start + k];
            lp += -0.5 * prec_l * v * v + 0.5 * std::log(prec_l) - half_log_2pi;
        }
        return lp + std::log(2.0);
    };

    return block;
}

} // namespace tulpa

#endif // TULPA_LATENT_FACTOR_BLOCK_FACTORY_H
