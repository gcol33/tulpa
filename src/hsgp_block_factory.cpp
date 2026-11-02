// hsgp_block_factory.cpp
// Definition of make_hsgp_block, declared in hsgp_block_factory.h.

#include <RcppEigen.h>
#include "hsgp_block_factory.h"
#include "hmc_hsgp_kernels.h"  // spectral_density_se

namespace tulpa {

LatentBlock make_hsgp_block(
    int                            start,
    int                            m_total,
    const Rcpp::List&              phi_per_arm,
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    const Rcpp::NumericVector&     eigenvalues,
    int                            axis_sigma2,
    int                            axis_ell,
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

    // sqrt_S — rebuilt by prep() at each outer-grid cell. Per-cell slot
    // storage: the joint driver runs outer cells concurrently, so a single
    // shared vector would let one cell's prep clobber another cell's
    // in-flight basis_eval read.
    auto sqrt_S_cache = std::make_shared<NlCellCache<std::vector<double>>>();

    LatentBlock block;
    block.start = start;
    block.size  = m_total;
    block.contrib_kind = BlockContribKind::DENSE_BASIS;
    block.prior_kind   = PriorFillKind::DIAGONAL_LOWRANK;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale left empty — copy not supported.
    // idx / obs_indices left empty — DENSE_BASIS uses basis_eval.

    block.prep = [sqrt_S_cache, eig, m_total,
                   axis_sigma2, axis_ell, theta_grid](
        int k_grid) -> bool {
        const double sigma2 = theta_grid(k_grid, axis_sigma2);
        const double ell    = theta_grid(k_grid, axis_ell);
        if (!(sigma2 > 0.0) || !(ell > 0.0)) return false;
        auto& cache = sqrt_S_cache->claim();
        cache.assign(m_total, 0.0);
        const auto& evs = *eig;
        for (int j = 0; j < m_total; j++) {
            const double S_j = tulpa_hsgp::spectral_density_se(evs[j], sigma2, ell);
            cache[j] = std::sqrt(S_j > 0.0 ? S_j : 0.0);
        }
        sqrt_S_cache->publish(k_grid);
        return true;
    };

    block.basis_eval = [phi_flat_per_arm, sqrt_S_cache, m_total](
        int i, int k_arm, int k_grid, double* out
    ) {
        const auto& flat = (*phi_flat_per_arm)[k_arm];
        const auto& sS   = sqrt_S_cache->find(k_grid);
        const double* row = flat.data() + static_cast<size_t>(i) * m_total;
        for (int j = 0; j < m_total; j++) {
            out[j] = row[j] * sS[j];
        }
    };

    // Batched view for the SYRK / GEMM scatter path.
    // Raw Phi (cached at factory time) + cell k_grid's sqrt_S.
    // m_offset_in_block = 0 for single-output HSGP.
    auto n_obs_cache = std::make_shared<std::vector<int>>(
        n_obs_per_arm.begin(), n_obs_per_arm.end());
    block.dense_basis_batch = [phi_flat_per_arm, sqrt_S_cache, n_obs_cache,
                                m_total](int k_arm, int k_grid)
        -> LatentBlock::DenseBasisBatch {
        LatentBlock::DenseBasisBatch b;
        b.data              = (*phi_flat_per_arm)[k_arm].data();
        b.sqrt_S            = sqrt_S_cache->find(k_grid).data();
        b.N_k               = (*n_obs_cache)[k_arm];
        b.m_per_arm         = m_total;
        b.m_offset_in_block = 0;
        return b;
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
    // Dense prior on beta is N(0, I): identity precision, matching
    // add_prior_sparse value-for-value (latent_block.h: both must agree). Used
    // by the dense spec solver (laplace_mode_spec_dense_solve), e.g. the
    // single-point cpp_laplace_fit_hsgp fixed-hyper fit.
    block.add_prior = [start, m_total](
        DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x, int /*k_grid*/
    ) {
        for (int j = 0; j < m_total; j++) {
            int idx = start + j;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };

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
