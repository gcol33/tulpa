// hsgp_mo_block_factory.h
// Build a LatentBlock for a multi-output (K-output) Hilbert-space Gaussian-
// process field — co-regionalization HSGP — plugged into the joint multi-
// arm nested-Laplace driver (Stage 1.7).
//
// Model.
//   K = n_arms correlated latent fields f_1(s), ..., f_K(s) share the same
//   basis Phi and eigenvalues but have cross-output-correlated basis
//   coefficients:
//
//     f_k(s) = sum_m phi_m(s) * sqrt(S_norm(lambda_m, ell)) * beta_{k, m}
//     (beta_{1,m}, ..., beta_{K,m}) ~ N(0, Sigma)    independently per m
//
//   Sigma is the K x K cross-output covariance. For K = 2 (first ship):
//     Sigma = [[sigma_1^2,            rho * sigma_1 * sigma_2],
//              [rho * sigma_1 * sigma_2,  sigma_2^2          ]]
//
//   The marginal variance of f_k matches a standard HSGP with sigma^2 =
//   sigma_k^2; when rho = 0 the block reduces to K independent single-
//   output HSGPs. sqrt(S_norm) carries only the spectral shape (ell), the
//   scale lives entirely in Sigma.
//
// Storage layout — output-major.
//   x[start + k * M + m] = beta_{k, m}    for k in [0, K), m in [0, M)
//   Each output k occupies a contiguous range [k*M, (k+1)*M); basis_eval
//   writes the active arm's range and zeros elsewhere, giving cache-
//   friendly contiguous writes (the sparse scatter then skips the zeros
//   via its `w != 0.0` filter on the K*M-wide buffer).
//
// Prior precision Q = I_M ⊗ Sigma^{-1}.
//   In output-major coordinates the prior couples (k1, m) and (k2, m) at
//   matched basis index m only:
//     Q[(k1*M + m), (k2*M + m)] = Sigma_inv[k1, k2]   for every m
//   That is M independent K x K precision blocks scattered through the
//   K*M sub-vector. add_prior_pattern enumerates the K*(K+1)/2 * M lower-
//   triangle entries; add_prior_sparse scatters the matching values.
//
// Lifecycle.
//   prep(k_grid) reads (sigma_1, sigma_2, rho, ell) — raw, no log
//   transform — from theta_grid and refreshes:
//     * sqrt_S_norm_cache[m] = sqrt((2*pi) * ell^2 * exp(-0.5*ell^2*lambda_m))
//     * Sigma_inv (K x K row-major, closed form for K = 2)
//     * log_det_Sigma scalar
//   Returns false when sigma_1, sigma_2 <= 0, |rho| >= 1, or ell <= 0.
//   The outer-grid driver short-circuits the cell to log_marginal = -inf.
//
// Axes (K = 2 first ship).
//   (sigma_1, sigma_2, rho, ell)  — raw values; |rho| < 1, others > 0.
//
// Pattern over-fill note.
//   The joint H pattern enumerator currently treats every DENSE_BASIS
//   block as a full block x block dense sub-pattern (K*M)^2. For multi-
//   output HSGP only K*M^2 + M*K*(K+1)/2 entries are actually populated
//   by the data + prior. CHOLMOD sees the extra zeros as numerical noise;
//   first-ship correctness is unaffected. Tighter pattern enumeration
//   would route through MULTI_OUTPUT_BASIS (TODO if a downstream workload
//   asks).
//
// First-ship scope.
//   K = 2 only — the C++ factory hard-errors otherwise. The K > 2
//   generalization needs an LKJ-cholesky correlation parameterization on
//   the outer grid plus an Eigen LLT for Sigma_inv; the basis_eval / add_
//   prior_* code paths already generalize once those axes are wired.

#ifndef TULPA_HSGP_MO_BLOCK_FACTORY_H
#define TULPA_HSGP_MO_BLOCK_FACTORY_H

#include "latent_block.h"
#include "nl_cell_cache.h"
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

inline LatentBlock make_hsgp_mo_block(
    int                            start,
    int                            m_total,
    int                            n_arms,            // K (first ship: must == 2)
    const Rcpp::List&              phi_per_arm,        // List of NumericMatrix
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            block_index,
    const Rcpp::NumericVector&     eigenvalues,
    int                            axis_sigma_1,
    int                            axis_sigma_2,
    int                            axis_rho,
    int                            axis_ell,
    const Rcpp::NumericMatrix&     theta_grid
) {
    const int K = n_arms;
    if (K != 2) {
        Rcpp::stop("Block %d (type 'hsgp_mo'): first ship requires "
                   "n_arms == 2 (got %d). Multi-output HSGP with K > 2 "
                   "needs an LKJ-cholesky correlation parameterization "
                   "(deferred follow-up).",
                   block_index + 1, K);
    }
    if (eigenvalues.size() != m_total) {
        Rcpp::stop("Block %d (type 'hsgp_mo'): length(eigenvalues) must "
                   "equal m_total (%d), got %d.",
                   block_index + 1, m_total,
                   static_cast<int>(eigenvalues.size()));
    }
    if (static_cast<int>(phi_per_arm.size()) != K ||
        n_obs_per_arm.size() != K) {
        Rcpp::stop("Block %d (type 'hsgp_mo'): phi_per_arm / n_obs_per_arm "
                   "must each have length n_arms (%d).",
                   block_index + 1, K);
    }

    // Materialize per-arm phi row-major flat vectors at factory time.
    auto phi_flat_per_arm =
        std::make_shared<std::vector<std::vector<double>>>(K);
    for (int k = 0; k < K; k++) {
        Rcpp::NumericMatrix phi_k = phi_per_arm[k];
        int N_k = n_obs_per_arm[k];
        if (phi_k.nrow() != N_k || phi_k.ncol() != m_total) {
            Rcpp::stop("Block %d (type 'hsgp_mo'): phi_per_arm[[%d]] must "
                       "be n_obs_k x m_total (expected %d x %d, got %d x %d).",
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

    auto eig = std::make_shared<std::vector<double>>(
        eigenvalues.begin(), eigenvalues.end());

    // prep-refreshed per-cell state (nl_cell_cache.h): concurrent outer-grid
    // cells each publish their own slot instead of sharing one buffer.
    struct HsgpMoCellState {
        std::vector<double> sqrt_S;
        std::vector<double> Sigma_inv;
        double              log_det_Sigma = 0.0;
    };
    auto cell_cache = std::make_shared<NlCellCache<HsgpMoCellState>>();

    const int size = K * m_total;

    LatentBlock block;
    block.start = start;
    block.size  = size;
    block.contrib_kind = BlockContribKind::DENSE_BASIS;
    block.prior_kind   = PriorFillKind::DIAGONAL_LOWRANK;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale left empty (copy not supported on multi-output HSGP first ship).
    // idx / obs_indices left empty — DENSE_BASIS uses basis_eval.

    block.prep = [cell_cache, eig, m_total, K,
                   axis_sigma_1, axis_sigma_2, axis_rho, axis_ell,
                   theta_grid](int k_grid) -> bool {
        const double sigma_1 = theta_grid(k_grid, axis_sigma_1);
        const double sigma_2 = theta_grid(k_grid, axis_sigma_2);
        const double rho     = theta_grid(k_grid, axis_rho);
        const double ell     = theta_grid(k_grid, axis_ell);
        if (!(sigma_1 > 0.0) || !(sigma_2 > 0.0) || !(ell > 0.0)) return false;
        if (!(rho > -1.0 && rho < 1.0)) return false;
        const double one_minus_rho2 = 1.0 - rho * rho;
        if (!(one_minus_rho2 > 0.0)) return false;

        auto& st = cell_cache->claim();

        // sqrt_S — sigma absorbed into Sigma; basis spectral density carries
        // only the lengthscale.
        const double pref = (2.0 * M_PI) * ell * ell;
        const double e_coef = -0.5 * ell * ell;
        const auto& evs = *eig;
        st.sqrt_S.assign(m_total, 0.0);
        for (int m = 0; m < m_total; m++) {
            const double S_m = pref * std::exp(e_coef * evs[m]);
            st.sqrt_S[m] = std::sqrt(S_m > 0.0 ? S_m : 0.0);
        }

        // Sigma_inv (K = 2 closed form).
        //   det(Sigma) = sigma_1^2 * sigma_2^2 * (1 - rho^2)
        //   Sigma_inv[0,0] = 1 / (sigma_1^2 * (1 - rho^2))
        //   Sigma_inv[1,1] = 1 / (sigma_2^2 * (1 - rho^2))
        //   Sigma_inv[0,1] = Sigma_inv[1,0] = -rho / (sigma_1 * sigma_2 * (1 - rho^2))
        st.Sigma_inv.assign(static_cast<size_t>(K) * K, 0.0);
        auto& Si = st.Sigma_inv;
        const double inv_omr2 = 1.0 / one_minus_rho2;
        Si[0 * K + 0] = inv_omr2 / (sigma_1 * sigma_1);
        Si[1 * K + 1] = inv_omr2 / (sigma_2 * sigma_2);
        Si[0 * K + 1] = -rho * inv_omr2 / (sigma_1 * sigma_2);
        Si[1 * K + 0] = Si[0 * K + 1];

        // log|Sigma| = log(sigma_1^2 * sigma_2^2 * (1 - rho^2))
        //            = 2*log(sigma_1) + 2*log(sigma_2) + log(1 - rho^2)
        st.log_det_Sigma = 2.0 * std::log(sigma_1)
                           + 2.0 * std::log(sigma_2)
                           + std::log(one_minus_rho2);
        cell_cache->publish(k_grid);
        return true;
    };

    block.basis_eval = [phi_flat_per_arm, cell_cache,
                         m_total, size](
        int i, int k_arm, int k_grid, double* out
    ) {
        // Output-major: write zeros over the full K*M buffer (callers do
        // not pre-zero), then fill the active arm's contiguous range.
        for (int j = 0; j < size; j++) out[j] = 0.0;
        const auto& flat = (*phi_flat_per_arm)[k_arm];
        const auto& sS   = cell_cache->find(k_grid).sqrt_S;
        const double* row = flat.data() + static_cast<size_t>(i) * m_total;
        const int off = k_arm * m_total;
        for (int m = 0; m < m_total; m++) {
            out[off + m] = row[m] * sS[m];
        }
    };

    // Batched view for the SYRK / GEMM scatter path (Stage 2.1).
    // Per-arm Phi is N_k x m_total; lands at slot k_arm * m_total inside the
    // K*M block (output-major). Sigma cross-output coupling is handled
    // separately via add_prior_sparse, not here.
    auto n_obs_cache = std::make_shared<std::vector<int>>(
        n_obs_per_arm.begin(), n_obs_per_arm.end());
    block.dense_basis_batch = [phi_flat_per_arm, cell_cache, n_obs_cache,
                                m_total](int k_arm, int k_grid)
        -> LatentBlock::DenseBasisBatch {
        LatentBlock::DenseBasisBatch b;
        b.data              = (*phi_flat_per_arm)[k_arm].data();
        b.sqrt_S            = cell_cache->find(k_grid).sqrt_S.data();
        b.N_k               = (*n_obs_cache)[k_arm];
        b.m_per_arm         = m_total;
        b.m_offset_in_block = k_arm * m_total;
        return b;
    };

    block.add_prior_pattern = [start, m_total, K](
        std::vector<std::pair<int,int>>& out
    ) {
        // M independent K x K precision blocks. Lower triangle entries
        // (r >= c) at (start + k1*M + m, start + k2*M + m) for k2 <= k1.
        for (int m = 0; m < m_total; m++) {
            for (int k1 = 0; k1 < K; k1++) {
                for (int k2 = 0; k2 <= k1; k2++) {
                    int r = start + k1 * m_total + m;
                    int c = start + k2 * m_total + m;
                    out.emplace_back(r, c);
                }
            }
        }
    };

    block.add_prior_sparse = [start, m_total, K, cell_cache](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        const auto& Si = cell_cache->find(k_grid).Sigma_inv;
        for (int m = 0; m < m_total; m++) {
            // grad[k1*M + m] -= sum_{k2} Sigma_inv[k1, k2] * beta_{k2, m}
            for (int k1 = 0; k1 < K; k1++) {
                int idx1 = start + k1 * m_total + m;
                double g = 0.0;
                for (int k2 = 0; k2 < K; k2++) {
                    int idx2 = start + k2 * m_total + m;
                    g += Si[k1 * K + k2] * x[idx2];
                }
                grad[idx1] -= g;
            }
            // H: lower-triangle Sigma_inv block at basis m.
            for (int k1 = 0; k1 < K; k1++) {
                int r = start + k1 * m_total + m;
                for (int k2 = 0; k2 <= k1; k2++) {
                    int c = start + k2 * m_total + m;
                    H.add(r, c, Si[k1 * K + k2]);
                }
            }
        }
    };

    block.log_prior = [start, m_total, K, cell_cache](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        const auto& st = cell_cache->find(k_grid);
        const auto& Si = st.Sigma_inv;
        double quad = 0.0;
        for (int m = 0; m < m_total; m++) {
            for (int k1 = 0; k1 < K; k1++) {
                int idx1 = start + k1 * m_total + m;
                const double v1 = x[idx1];
                for (int k2 = 0; k2 < K; k2++) {
                    int idx2 = start + k2 * m_total + m;
                    quad += Si[k1 * K + k2] * v1 * x[idx2];
                }
            }
        }
        double lp = -0.5 * quad
                    - 0.5 * m_total * st.log_det_Sigma
                    - 0.5 * K * m_total * std::log(2.0 * M_PI);
        return lp;
    };

    // No centering — beta is anchored by the K x K cross-output Gaussian.

    return block;
}

} // namespace tulpa

#endif // TULPA_HSGP_MO_BLOCK_FACTORY_H
