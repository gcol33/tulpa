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
// time, so the frame it holds is the UNION of the per-grid Q sparsity
// patterns, assembled by add_prior_pattern below. A grid point whose Q is
// sparser than the union writes into structural zeros, which is what the
// frame is for. Nothing has to be padded R-side.
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
    // The length agreement above holds when every per-grid list is empty, and
    // the pattern callback below reads grid point 0 unconditionally.
    if (n_grid_local < 1) {
        Rcpp::stop("Block %d (type 'tgmrf'): the per-grid arrays are empty; "
                   "at least one outer-grid point is required.",
                   block_index + 1);
    }
    if (size < 0) {
        Rcpp::stop("Block %d (type 'tgmrf'): n_latent (%d) must be "
                   "non-negative.", block_index + 1, size);
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
        // p_k[size] is the entry count CSC declares. Both prior scatters walk
        // i_k / x_k over [p_k[j], p_k[j+1]) and write H[start + i_loc][...] and
        // grad[start + i_loc], where DenseMat::operator[] is a raw row offset:
        // a row index past `size` writes outside the matrix. Check the whole
        // frame here, where the grid point that is wrong can still be named.
        const int nnz = p_k[size];
        if (p_k[0] != 0 || nnz < 0) {
            Rcpp::stop("Block %d (type 'tgmrf'): Q_csc_p_per_grid[[%d]] must "
                       "start at 0 and end at a non-negative entry count "
                       "(got %d and %d).",
                       block_index + 1, k + 1, p_k[0], nnz);
        }
        for (int j = 0; j < size; j++) {
            if (p_k[j + 1] < p_k[j]) {
                Rcpp::stop("Block %d (type 'tgmrf'): Q_csc_p_per_grid[[%d]] is "
                           "not non-decreasing at column %d.",
                           block_index + 1, k + 1, j + 1);
            }
        }
        if (static_cast<int>(i_k.size()) != nnz ||
            static_cast<int>(x_k.size()) != nnz) {
            Rcpp::stop("Block %d (type 'tgmrf'): Q_csc_i_per_grid[[%d]] and "
                       "Q_csc_x_per_grid[[%d]] have lengths %d and %d; both "
                       "must equal p[n_latent] = %d.",
                       block_index + 1, k + 1, k + 1,
                       static_cast<int>(i_k.size()),
                       static_cast<int>(x_k.size()), nnz);
        }
        for (int t = 0; t < nnz; t++) {
            if (i_k[t] == NA_INTEGER || i_k[t] < 0 || i_k[t] >= size) {
                Rcpp::stop("Block %d (type 'tgmrf'): Q_csc_i_per_grid[[%d]][%d] "
                           "is %d; must be a 0-based row index in [0, %d).",
                           block_index + 1, k + 1, t + 1, i_k[t], size);
            }
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

    // Pattern: the UNION over grid points. The builder is initialized once at
    // fit time, so an entry any grid point carries has to be in the frame:
    // SparseHessianBuilder::add DISCARDS an out-of-frame write, so a frame taken
    // from one grid point silently drops whatever the others hold beyond it.
    // Reading grid point 0 alone is not safe even under the R-side convention
    // that the pattern is constant, because an assembled Q loses entries at a
    // parameter value that zeroes them -- an AR1 Q at rho = 0 is diagonal, and
    // Matrix drops an assigned zero from the pattern -- so grid point 0 can be
    // the degenerate one. The union costs structural zeros in the frame, which
    // is what a fixed symbolic frame is for. Duplicates are harmless: the
    // builder's symbolic pass is the one that resolves them.
    block.add_prior_pattern = [n_grid_local, start, size, Q_p_vec, Q_i_vec](
        std::vector<std::pair<int,int>>& out
    ) {
        for (int k = 0; k < n_grid_local; k++) {
            const auto& p_v = (*Q_p_vec)[k];
            const auto& i_v = (*Q_i_vec)[k];
            for (int j = 0; j < size; j++) {
                for (int idx = p_v[j]; idx < p_v[j + 1]; idx++) {
                    int i_loc = i_v[idx];
                    if (i_loc == j) continue;      // diagonal added by builder
                    int hi = (i_loc > j) ? i_loc : j;
                    int lo = (i_loc > j) ? j : i_loc;
                    out.emplace_back(start + hi, start + lo);
                }
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
