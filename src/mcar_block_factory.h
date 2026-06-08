// mcar_block_factory.h
// Separable multivariate CAR (MCAR) block for the joint nested-Laplace driver
// (gcol33/tulpa#89). p areal fields over one graph share a cross-covariance
// Sigma: the joint latent (u_1, ..., u_p) over n cells has precision
//   P = Sigma^-1 (x) Q,            Q = D - W   (intrinsic CAR, rho = 1).
//
// Natural parameterization: the fields ARE the latent (no amplitude in eta).
// Observation i in arm k contributes eta_i += sum_a X_{ia} u_a[cell_i], so the
// block is INDEXED_MULTI -- obs i touches the p slots {a*n + cell_i} with the
// design weights X_{ia} (the intercept column is all-ones, a covariate column
// is the per-row value, exactly the svc_weight values). Sigma enters ONLY the
// prior, integrated over by the outer grid in log-Cholesky coordinates.
//
// Log-determinant tractability: Q is rank n-1 (constant null space per field),
// so P is rank p(n-1) with a p-dim null space spanned by the per-field
// constants. Its pseudo-determinant factorizes,
//   logpdet(Sigma^-1 (x) Q) = (n-1) log|Sigma^-1| + p logpdet(Q),
// so the only Sigma-dependent normalizer is (n-1) log|Sigma^-1| = 2(n-1) sum_i
// (log-Cholesky diagonal), a p x p quantity -- no large generalized determinant.
// The p constant directions are pinned by p sum-to-zero rank-1 penalties (one
// per field), folded by the sparse solver's block-Schur path (gcol33/tulpa#69).

#ifndef TULPA_MCAR_BLOCK_FACTORY_H
#define TULPA_MCAR_BLOCK_FACTORY_H

#include "latent_block.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <vector>
#include <cmath>

namespace tulpa {

// Sum-to-zero penalty precision; matches the ICAR path (laplace_spatial_priors).
constexpr double MCAR_SUM2ZERO_TAU = 1.0;

// Build Sigma^-1 (p x p, row-major) and log|Sigma| from the log-Cholesky
// coordinates of Sigma = L L'. Column-major lower-triangle order (matching
// R's .re_logchol_to_L): for j in 0..p-1, for i in j..p-1, theta holds
// L[i][j], with the diagonal stored as log L[i][i]. Returns log|Sigma| via the
// out-param (= 2 sum_i log L[i][i] = 2 sum of the diagonal coords).
inline void mcar_sigma_inv_from_logchol(
    const double* theta, int p, std::vector<double>& Sinv, double& log_det_Sigma
) {
    std::vector<double> L((std::size_t) p * p, 0.0);
    log_det_Sigma = 0.0;
    int idx = 0;
    for (int j = 0; j < p; ++j) {
        for (int i = j; i < p; ++i) {
            if (i == j) {
                const double ld = theta[idx];
                L[(std::size_t) i * p + j] = std::exp(ld);
                log_det_Sigma += 2.0 * ld;
            } else {
                L[(std::size_t) i * p + j] = theta[idx];
            }
            ++idx;
        }
    }
    // M = L^-1 (lower-triangular inverse by forward substitution).
    std::vector<double> M((std::size_t) p * p, 0.0);
    for (int col = 0; col < p; ++col) {
        M[(std::size_t) col * p + col] = 1.0 / L[(std::size_t) col * p + col];
        for (int i = col + 1; i < p; ++i) {
            double s = 0.0;
            for (int k = col; k < i; ++k)
                s += L[(std::size_t) i * p + k] * M[(std::size_t) k * p + col];
            M[(std::size_t) i * p + col] = -s / L[(std::size_t) i * p + i];
        }
    }
    // Sigma^-1 = M' M (sum over rows of M).
    Sinv.assign((std::size_t) p * p, 0.0);
    for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b) {
            double s = 0.0;
            for (int k = 0; k < p; ++k)
                s += M[(std::size_t) k * p + a] * M[(std::size_t) k * p + b];
            Sinv[(std::size_t) a * p + b] = s;
        }
}

// Q x_b for field b: (Q x_b)[i] = nnbr[i] x_b[i] - sum_{j~i} x_b[j].
inline void mcar_apply_Q(
    const Rcpp::NumericVector& x, int start, int n, int b,
    const Rcpp::IntegerVector& adj_rp, const Rcpp::IntegerVector& adj_ci,
    const Rcpp::IntegerVector& nnbr, std::vector<double>& out
) {
    const int off = start + b * n;
    for (int i = 0; i < n; ++i) {
        double v = static_cast<double>(nnbr[i]) * x[off + i];
        for (int k = adj_rp[i]; k < adj_rp[i + 1]; ++k) {
            const int j = adj_ci[k];
            if (j != i) v -= x[off + j];
        }
        out[i] = v;
    }
}

// Construct the MCAR LatentBlock.
//   start        : latent offset of the p*n field block.
//   n            : n_spatial_units (cells).
//   p            : n_fields.
//   axis0        : first log-Cholesky axis column in theta_grid; the block
//                  reads p(p+1)/2 consecutive columns there.
//   cell_idx     : per-arm 1-based cell index (length n_arms list of vectors).
//   field_weight : per-field, per-arm design column (X_{ia}); outer length p,
//                  inner length n_arms. field_weight[a][k][i] = X_{ia} on arm k.
//   adjacency    : CSR (adj_rp, adj_ci, nnbr), 0-based.
inline LatentBlock make_mcar_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> cell_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    Rcpp::IntegerVector adj_rp, Rcpp::IntegerVector adj_ci,
    Rcpp::IntegerVector nnbr
) {
    const int m = p * (p + 1) / 2;
    (void) m;

    LatentBlock block;
    block.start = start;
    block.size  = p * n;
    block.contrib_kind = BlockContribKind::INDEXED_MULTI;
    block.prior_kind   = PriorFillKind::ADJACENCY;
    block.d_fac = [](int) -> double { return 1.0; };

    // eta_i += sum_a X_{ia} u_a[cell_i]: obs i touches the p field slots
    // {a*n + cell_i} (1-based block-local) with weights X_{ia}.
    block.obs_indices = [cell_idx, field_weight, n, p](
        int i, int k_arm, std::vector<std::pair<int,double>>& out
    ) {
        out.clear();
        const Rcpp::IntegerVector& ci = cell_idx[k_arm];
        if (i < 0 || i >= ci.size()) return;
        const int cell = ci[i];                 // 1-based; 0 => skip
        if (cell < 1 || cell > n) return;
        out.reserve(p);
        for (int a = 0; a < p; ++a)
            out.emplace_back(a * n + cell, field_weight[a][k_arm][i]);
    };

    // Sparse prior scatter: P = Sigma^-1 (x) Q, grad += -P x, plus p sum-to-zero
    // pins (gradient + rank-1, folded by the solver). Sigma^-1 is recomputed
    // from theta_grid(k, .) here (cheap, p x p) so the closure holds no mutable
    // state and is safe across the parallel outer grid.
    block.add_prior_sparse = [start, n, p, axis0, theta_grid,
                               adj_rp, adj_ci, nnbr](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        std::vector<double> Sinv; double log_det_Sigma;
        {
            std::vector<double> th(p * (p + 1) / 2);
            for (int t = 0; t < (int) th.size(); ++t) th[t] = theta_grid(k_grid, axis0 + t);
            mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
        }
        // Gradient: grad[a*n+i] += -sum_b Sinv[a,b] (Q x_b)[i].
        std::vector<std::vector<double>> Qx(p, std::vector<double>(n));
        for (int b = 0; b < p; ++b)
            mcar_apply_Q(x, start, n, b, adj_rp, adj_ci, nnbr, Qx[b]);
        for (int a = 0; a < p; ++a)
            for (int i = 0; i < n; ++i) {
                double g = 0.0;
                for (int b = 0; b < p; ++b) g += Sinv[(std::size_t) a * p + b] * Qx[b][i];
                grad[start + a * n + i] += -g;
            }
        // Hessian P = Sigma^-1 (x) Q, lower triangle (stype = -1).
        for (int a = 0; a < p; ++a) {
            for (int b = 0; b <= a; ++b) {
                const double coef = Sinv[(std::size_t) a * p + b];
                const int oa = start + a * n, ob = start + b * n;
                if (a == b) {
                    for (int i = 0; i < n; ++i) {
                        H.add(oa + i, oa + i, coef * static_cast<double>(nnbr[i]));
                        for (int kk = adj_rp[i]; kk < adj_rp[i + 1]; ++kk) {
                            const int j = adj_ci[kk];
                            if (j < i) H.add(oa + i, oa + j, -coef);  // lower tri of Q
                        }
                    }
                } else {  // a > b: every (i, j) entry is globally lower-triangular
                    for (int i = 0; i < n; ++i) {
                        H.add(oa + i, ob + i, coef * static_cast<double>(nnbr[i]));
                        for (int kk = adj_rp[i]; kk < adj_rp[i + 1]; ++kk) {
                            const int j = adj_ci[kk];
                            if (j != i) H.add(oa + i, ob + j, -coef);
                        }
                    }
                }
            }
        }
        // Per-field sum-to-zero pins (constant null space of Q): exact gradient
        // + rank-1 11' registered for the block-Schur fold (gcol33/tulpa#69).
        for (int a = 0; a < p; ++a) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += x[start + a * n + i];
            for (int i = 0; i < n; ++i) grad[start + a * n + i] -= MCAR_SUM2ZERO_TAU * s;
            H.add_s2z_rank1(start + a * n, n, MCAR_SUM2ZERO_TAU);
        }
    };

    // Sparsity pattern: P's lower-triangle nonzeros (diagonal always present).
    block.add_prior_pattern = [start, n, p, adj_rp, adj_ci](
        std::vector<std::pair<int,int>>& out
    ) {
        for (int a = 0; a < p; ++a) {
            for (int b = 0; b <= a; ++b) {
                const int oa = start + a * n, ob = start + b * n;
                for (int i = 0; i < n; ++i) {
                    out.emplace_back(oa + i, ob + i);     // (a,b) diagonal-in-cell
                    for (int kk = adj_rp[i]; kk < adj_rp[i + 1]; ++kk) {
                        const int j = adj_ci[kk];
                        if (j == i) continue;
                        if (a == b) { if (j < i) out.emplace_back(oa + i, oa + j); }
                        else        { out.emplace_back(oa + i, ob + j); }
                    }
                }
            }
        }
    };

    // log p(u | Sigma): -0.5 u'Pu - 0.5 SUM2ZERO sum_a (sum u_a)^2
    //   + 0.5 (n-1) log|Sigma^-1|  - 0.5 p (n-1) log(2 pi).
    // The constant p logpdet(Q) term (Sigma-independent) is dropped, mirroring
    // log_prior_icar; the Sigma-dependent (n-1) log|Sigma^-1| is exact.
    block.log_prior = [start, n, p, axis0, theta_grid,
                       adj_rp, adj_ci, nnbr](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        std::vector<double> Sinv; double log_det_Sigma;
        {
            std::vector<double> th(p * (p + 1) / 2);
            for (int t = 0; t < (int) th.size(); ++t) th[t] = theta_grid(k_grid, axis0 + t);
            mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
        }
        std::vector<std::vector<double>> Qx(p, std::vector<double>(n));
        for (int b = 0; b < p; ++b)
            mcar_apply_Q(x, start, n, b, adj_rp, adj_ci, nnbr, Qx[b]);
        double quad = 0.0;                         // u'Pu = sum_{a,b} Sinv[a,b] u_a' Q u_b
        for (int a = 0; a < p; ++a)
            for (int b = 0; b < p; ++b) {
                double xa_Qb = 0.0;
                for (int i = 0; i < n; ++i) xa_Qb += x[start + a * n + i] * Qx[b][i];
                quad += Sinv[(std::size_t) a * p + b] * xa_Qb;
            }
        double pin = 0.0;
        for (int a = 0; a < p; ++a) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += x[start + a * n + i];
            pin += s * s;
        }
        const double log_det_Sinv = -log_det_Sigma;
        return -0.5 * quad - 0.5 * MCAR_SUM2ZERO_TAU * pin
               + 0.5 * (n - 1) * log_det_Sinv
               - 0.5 * p * (n - 1) * std::log(2.0 * M_PI);
    };

    // Center each field to sum-to-zero after each Newton step (belt-and-braces
    // with the pins; the per-field constant is unidentified by the prior).
    block.center = [start, n, p](Rcpp::NumericVector& x) -> double {
        for (int a = 0; a < p; ++a) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += x[start + a * n + i];
            const double mean = s / n;
            for (int i = 0; i < n; ++i) x[start + a * n + i] -= mean;
        }
        return 0.0;
    };

    return block;
}

}  // namespace tulpa

#endif  // TULPA_MCAR_BLOCK_FACTORY_H
