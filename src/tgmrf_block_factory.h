// tgmrf_block_factory.h
// Build a LatentBlock for a templated user-supplied GMRF (tgmrf) plugged
// into the joint multi-arm nested-Laplace driver. Joint analogue of the
// single-arm "tgmrf" block in nested_laplace_multi.cpp.
//
// The R side precomputes Q(theta_k) at every outer-grid row plus log|Q_k|
// and log p(theta_k). The factory just reads those arrays and assembles the
// callbacks; the C++ side never indexes theta_grid for this block (Q_k is
// fully materialized R-side).
//
// Pattern contract. The SparseHessianBuilder is initialized once at fit-
// time, so the union of per-grid Q sparsity patterns must be representable
// in a single CSC frame. We use the pattern from k=0 as canonical and
// require every other grid point's Q to share it; users with truly
// varying patterns must pad with structural zeros R-side.
//
// Contrib kind: INDEXED_SINGLE (one block-local DOF per obs via obs_idx).
// Prior fill: USER_CSC.
// Copy semantics not supported.

#ifndef TULPA_TGMRF_BLOCK_FACTORY_H
#define TULPA_TGMRF_BLOCK_FACTORY_H

#include "latent_block.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

inline LatentBlock make_tgmrf_block(
    int                                              start,
    int                                              size,
    std::function<int(int /*i*/, int /*k_arm*/)>     obs_idx_fn,
    const Rcpp::List&                                Q_csc_p_per_grid,
    const Rcpp::List&                                Q_csc_i_per_grid,
    const Rcpp::List&                                Q_csc_x_per_grid,
    const Rcpp::NumericVector&                       logdet_Q_per_grid,
    const Rcpp::NumericVector&                       log_pi_theta_per_grid,
    int                                              block_index
) {
    int n_grid_local = Q_csc_p_per_grid.size();
    if (Q_csc_i_per_grid.size() != n_grid_local ||
        Q_csc_x_per_grid.size() != n_grid_local ||
        logdet_Q_per_grid.size() != n_grid_local ||
        log_pi_theta_per_grid.size() != n_grid_local) {
        Rcpp::stop("Block %d (type 'tgmrf'): per-grid arrays must all have "
                   "length %d.", block_index + 1, n_grid_local);
    }

    // Copy CSC triples into C++ vectors so the closures outlive the SEXPs.
    auto Q_p_vec = std::make_shared<std::vector<std::vector<int>>>(n_grid_local);
    auto Q_i_vec = std::make_shared<std::vector<std::vector<int>>>(n_grid_local);
    auto Q_x_vec = std::make_shared<std::vector<std::vector<double>>>(n_grid_local);
    auto logdet_Q     = std::make_shared<std::vector<double>>(n_grid_local);
    auto log_pi_theta = std::make_shared<std::vector<double>>(n_grid_local);

    for (int k = 0; k < n_grid_local; k++) {
        Rcpp::IntegerVector p_k = Q_csc_p_per_grid[k];
        Rcpp::IntegerVector i_k = Q_csc_i_per_grid[k];
        Rcpp::NumericVector x_k = Q_csc_x_per_grid[k];
        if (static_cast<int>(p_k.size()) != size + 1) {
            Rcpp::stop("Block %d (type 'tgmrf'): Q_csc_p_per_grid[[%d]] has "
                       "length %d, expected %d.",
                       block_index + 1, k + 1,
                       static_cast<int>(p_k.size()), size + 1);
        }
        (*Q_p_vec)[k].assign(p_k.begin(), p_k.end());
        (*Q_i_vec)[k].assign(i_k.begin(), i_k.end());
        (*Q_x_vec)[k].assign(x_k.begin(), x_k.end());
        (*logdet_Q)[k]     = logdet_Q_per_grid[k];
        (*log_pi_theta)[k] = log_pi_theta_per_grid[k];
    }

    LatentBlock block;
    block.start = start;
    block.size  = size;
    block.contrib_kind = BlockContribKind::INDEXED_SINGLE;
    block.prior_kind   = PriorFillKind::USER_CSC;
    block.idx          = obs_idx_fn;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale left empty — copy not supported.

    // Dense prior scatter (legacy fallback). Q is stored full (dgCMatrix
    // coerced via generalMatrix); walk every (i, j) once.
    block.add_prior = [start, size, Q_p_vec, Q_i_vec, Q_x_vec](
        DenseVec& grad, DenseMat& H,
        const Rcpp::NumericVector& x, int k
    ) {
        const auto& p_v = (*Q_p_vec)[k];
        const auto& i_v = (*Q_i_vec)[k];
        const auto& x_v = (*Q_x_vec)[k];
        for (int j = 0; j < size; j++) {
            double xj = x[start + j];
            for (int idx = p_v[j]; idx < p_v[j + 1]; idx++) {
                int    i_loc = i_v[idx];
                double q_ij  = x_v[idx];
                H[start + i_loc][start + j] += q_ij;
                grad[start + i_loc] -= q_ij * xj;
            }
        }
    };

    // Sparse twin. Q is stored full; lower-triangle-only H writes via
    // SparseHessianBuilder::add (which normalizes orientation internally).
    // Gradient uses the full Q (every (i, j) once).
    block.add_prior_sparse = [start, size, Q_p_vec, Q_i_vec, Q_x_vec](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k
    ) {
        const auto& p_v = (*Q_p_vec)[k];
        const auto& i_v = (*Q_i_vec)[k];
        const auto& x_v = (*Q_x_vec)[k];
        for (int j = 0; j < size; j++) {
            double xj = x[start + j];
            for (int idx = p_v[j]; idx < p_v[j + 1]; idx++) {
                int    i_loc = i_v[idx];
                double q_ij  = x_v[idx];
                grad[start + i_loc] -= q_ij * xj;
                if (i_loc >= j) {
                    H.add(start + i_loc, start + j, q_ij);
                }
            }
        }
    };

    // Pattern: from grid point 0 (canonical; all grid points must share).
    block.add_prior_pattern = [start, size, Q_p_vec, Q_i_vec](
        std::vector<std::pair<int,int>>& out
    ) {
        const auto& p0 = (*Q_p_vec)[0];
        const auto& i0 = (*Q_i_vec)[0];
        for (int j = 0; j < size; j++) {
            for (int idx = p0[j]; idx < p0[j + 1]; idx++) {
                int i_loc = i0[idx];
                if (i_loc == j) continue;          // diagonal added by builder
                int hi = (i_loc > j) ? i_loc : j;
                int lo = (i_loc > j) ? j : i_loc;
                out.emplace_back(start + hi, start + lo);
            }
        }
    };

    block.log_prior = [start, size, Q_p_vec, Q_i_vec, Q_x_vec,
                        logdet_Q, log_pi_theta](
        const Rcpp::NumericVector& x, int k
    ) -> double {
        const auto& p_v = (*Q_p_vec)[k];
        const auto& i_v = (*Q_i_vec)[k];
        const auto& x_v = (*Q_x_vec)[k];
        double quad = 0.0;
        for (int j = 0; j < size; j++) {
            double xj = x[start + j];
            for (int idx = p_v[j]; idx < p_v[j + 1]; idx++) {
                int    i_loc = i_v[idx];
                double q_ij  = x_v[idx];
                quad += x[start + i_loc] * q_ij * xj;
            }
        }
        return 0.5 * (*logdet_Q)[k]
             - 0.5 * quad
             - 0.5 * size * std::log(2.0 * M_PI)
             + (*log_pi_theta)[k];
    };

    // No center: user owns the parameterisation; centering would require
    // shifting the per-arm intercept by the per-step mean offset, which is
    // incompatible with arbitrary Q structures.

    return block;
}

} // namespace tulpa

#endif // TULPA_TGMRF_BLOCK_FACTORY_H
