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
// HSGP uses DENSE_BASIS semantics: every obs touches every basis
// coefficient through Phi[i, j]. The block size is m_total (number of
// basis functions), and the latent vector at [start, start + m_total)
// holds beta_j directly. eta_i += sum_j Phi[i, j] * sqrt_S_j * beta_j.
//
// Mesh data:
//   * eigenvalues: length-m_total double vector (shared across arms)
//   * phi_flat:    per-arm n_obs_k * m_total row-major matrix (per-arm
//                  basis evaluation at obs locations)
//
// Lifecycle:
//   * `prep(k_grid)` reads (log_sigma2, log_ell) from theta_grid, computes
//     sqrt_S[j] for j in 0..m_total and caches it in a shared vector.
//     basis_eval and the sparse scatter consume the cached vector.
//   * The prior on beta is N(0, I) -> diagonal precision; prior_kind = NONE,
//     no off-diagonal pattern entries.
//
// HSGP forces the sparse Newton path (dense scatter does not support
// DENSE_BASIS). add_prior is left empty; the dispatch in solve_at_theta_impl
// (1.4a) MUST route to the sparse path whenever any block has
// contrib_kind != INDEXED_SINGLE.
//
// Axis schema: (log_sigma2, log_lengthscale). PC priors on sigma2 and
// log-normal on lengthscale are applied OUTSIDE the inner Laplace, as
// part of the outer-grid hyper-prior, not in the block's log_prior.
// log_prior here is the N(0, I) on beta only.

#ifndef TULPA_HSGP_BLOCK_FACTORY_H
#define TULPA_HSGP_BLOCK_FACTORY_H

#include "latent_block.h"
#include "laplace_re_priors.h"  // center_effects (unused — HSGP doesn't center beta)
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

inline LatentBlock make_hsgp_block(
    int                            start,
    int                            m_total,
    const Rcpp::List&              phi_per_arm,            // List of NumericMatrix
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    const Rcpp::NumericVector&     eigenvalues,
    int                            axis_log_sigma2,
    int                            axis_log_ell,
    const Rcpp::NumericMatrix&     theta_grid
) {
    if (eigenvalues.size() != m_total) {
        Rcpp::stop("Block %d (type 'hsgp'): length(eigenvalues) must equal "
                   "m_total (%d), got %d.",
                   block_index + 1, m_total,
                   static_cast<int>(eigenvalues.size()));
    }
    if (static_cast<int>(phi_per_arm.size()) != n_arms ||
        n_obs_per_arm.size() != n_arms) {
        Rcpp::stop("Block %d (type 'hsgp'): phi_per_arm / n_obs_per_arm "
                   "must each have length n_arms (%d).",
                   block_index + 1, n_arms);
    }

    // Materialize per-arm phi row-major flat vectors at factory time.
    auto phi_flat_per_arm =
        std::make_shared<std::vector<std::vector<double>>>(n_arms);
    for (int k = 0; k < n_arms; k++) {
        Rcpp::NumericMatrix phi_k = phi_per_arm[k];
        int N_k = n_obs_per_arm[k];
        if (phi_k.nrow() != N_k || phi_k.ncol() != m_total) {
            Rcpp::stop("Block %d (type 'hsgp'): phi_per_arm[[%d]] must be "
                       "n_obs_k x m_total (expected %d x %d, got %d x %d).",
                       block_index + 1, k + 1, N_k, m_total,
                       static_cast<int>(phi_k.nrow()),
                       static_cast<int>(phi_k.ncol()));
        }
        auto& flat = (*phi_flat_per_arm)[k];
        flat.assign(static_cast<size_t>(N_k) * m_total, 0.0);
        for (int i = 0; i < N_k; i++) {
            for (int j = 0; j < m_total; j++) {
                flat[static_cast<size_t>(i) * m_total + j] = phi_k(i, j);
            }
        }
    }

    // Cached eigenvalues (avoid Rcpp wrapper overhead in tight loops).
    auto eig = std::make_shared<std::vector<double>>(
        eigenvalues.begin(), eigenvalues.end());

    // sqrt_S cache — rebuilt by prep() at each outer-grid cell.
    auto sqrt_S_cache = std::make_shared<std::vector<double>>(m_total, 0.0);

    LatentBlock block;
    block.start = start;
    block.size  = m_total;
    block.contrib_kind = BlockContribKind::DENSE_BASIS;
    block.prior_kind   = PriorFillKind::DIAGONAL_LOWRANK;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale left empty — copy not supported.
    // idx / obs_indices left empty — DENSE_BASIS uses basis_eval.

    block.prep = [sqrt_S_cache, eig, m_total,
                   axis_log_sigma2, axis_log_ell, theta_grid](
        int k_grid) -> bool {
        double log_sigma2 = theta_grid(k_grid, axis_log_sigma2);
        double log_ell    = theta_grid(k_grid, axis_log_ell);
        double sigma2 = std::exp(log_sigma2);
        double ell    = std::exp(log_ell);
        if (!(sigma2 > 0.0) || !(ell > 0.0)) return false;
        const double pref = sigma2 * std::sqrt(2.0 * M_PI) * ell;
        const double e_coef = -0.5 * ell * ell;
        auto& cache = *sqrt_S_cache;
        const auto& evs = *eig;
        for (int j = 0; j < m_total; j++) {
            double S_j = pref * std::exp(e_coef * evs[j]);
            cache[j] = std::sqrt(S_j > 0.0 ? S_j : 0.0);
        }
        return true;
    };

    block.basis_eval = [phi_flat_per_arm, sqrt_S_cache, m_total](
        int i, int k_arm, double* out
    ) {
        const auto& flat = (*phi_flat_per_arm)[k_arm];
        const auto& sS   = *sqrt_S_cache;
        const double* row = flat.data() + static_cast<size_t>(i) * m_total;
        for (int j = 0; j < m_total; j++) {
            out[j] = row[j] * sS[j];
        }
    };

    // Prior on beta is N(0, I): diagonal precision. No off-diagonal pattern
    // entries (pattern builder adds the diagonal unconditionally).
    block.add_prior_sparse = [start, m_total](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) {
        for (int j = 0; j < m_total; j++) {
            int idx = start + j;
            grad[idx] -= x[idx];
            H.add(idx, idx, 1.0);
        }
    };

    // add_prior_pattern left empty.
    // add_prior left empty: HSGP forces the sparse path.

    block.log_prior = [start, m_total](
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) -> double {
        double lp = 0.0;
        for (int j = 0; j < m_total; j++) {
            double v = x[start + j];
            lp -= 0.5 * v * v;
        }
        lp -= 0.5 * m_total * std::log(2.0 * M_PI);
        return lp;
    };

    // No centering — beta is already anchored by its N(0, I) prior.

    return block;
}

} // namespace tulpa

#endif // TULPA_HSGP_BLOCK_FACTORY_H
