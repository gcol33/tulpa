// laplace_temporal_priors.cpp
// Temporal precision helpers for Laplace engines.

#include "laplace_temporal_priors.h"
#include "sparse_hessian.h"

using namespace Rcpp;

namespace tulpa {

void add_rw1_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
) {
    for (int t = 1; t < n_times; t++) {
        int idx = start_idx + t;
        double diff = x[idx] - x[idx - 1];
        grad[idx] -= tau * diff;
        grad[idx - 1] += tau * diff;
        H[idx][idx] += tau;
        H[idx - 1][idx - 1] += tau;
        H[idx][idx - 1] -= tau;
        H[idx - 1][idx] -= tau;
    }
    if (cyclic && n_times > 1) {
        int idx_first = start_idx;
        int idx_last = start_idx + n_times - 1;
        double diff = x[idx_first] - x[idx_last];
        grad[idx_first] -= tau * diff;
        grad[idx_last] += tau * diff;
        H[idx_first][idx_first] += tau;
        H[idx_last][idx_last] += tau;
        H[idx_first][idx_last] -= tau;
        H[idx_last][idx_first] -= tau;
    }
}

void add_rw2_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
) {
    if (n_times < 3) return;
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        double diff2 = x[idx - 1] - 2.0 * x[idx] + x[idx + 1];
        grad[idx - 1] -= tau * diff2;
        grad[idx] += 2.0 * tau * diff2;
        grad[idx + 1] -= tau * diff2;
        H[idx - 1][idx - 1] += tau;
        H[idx][idx] += 4.0 * tau;
        H[idx + 1][idx + 1] += tau;
        H[idx - 1][idx] -= 2.0 * tau;
        H[idx][idx - 1] -= 2.0 * tau;
        H[idx][idx + 1] -= 2.0 * tau;
        H[idx + 1][idx] -= 2.0 * tau;
        H[idx - 1][idx + 1] += tau;
        H[idx + 1][idx - 1] += tau;
    }
}

void add_ar1_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, double rho
) {
    if (n_times < 1) return;
    double tau_marginal = tau * (1.0 - rho * rho);
    int idx0 = start_idx;
    grad[idx0] -= tau_marginal * x[idx0];
    H[idx0][idx0] += tau_marginal;
    for (int t = 1; t < n_times; t++) {
        int idx = start_idx + t;
        int idx_prev = start_idx + t - 1;
        double resid = x[idx] - rho * x[idx_prev];
        grad[idx] -= tau * resid;
        grad[idx_prev] += tau * rho * resid;
        H[idx][idx] += tau;
        H[idx_prev][idx_prev] += tau * rho * rho;
        H[idx][idx_prev] -= tau * rho;
        H[idx_prev][idx] -= tau * rho;
    }
}

// ============================================================
// Sparse twins. Same math; SparseHessianBuilder::add normalizes
// (row, col) to lower triangle internally, so each off-diagonal
// edge is written once.
// ============================================================

void add_rw1_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
) {
    for (int t = 1; t < n_times; t++) {
        int idx = start_idx + t;
        double diff = x[idx] - x[idx - 1];
        grad[idx] -= tau * diff;
        grad[idx - 1] += tau * diff;
        H.add(idx,     idx,     tau);
        H.add(idx - 1, idx - 1, tau);
        H.add(idx,     idx - 1, -tau);
    }
    if (cyclic && n_times > 1) {
        int idx_first = start_idx;
        int idx_last  = start_idx + n_times - 1;
        double diff = x[idx_first] - x[idx_last];
        grad[idx_first] -= tau * diff;
        grad[idx_last]  += tau * diff;
        H.add(idx_first, idx_first, tau);
        H.add(idx_last,  idx_last,  tau);
        H.add(idx_last,  idx_first, -tau);
    }
}

void add_rw2_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start_idx, int n_times, double tau, bool /*cyclic*/
) {
    if (n_times < 3) return;
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        double diff2 = x[idx - 1] - 2.0 * x[idx] + x[idx + 1];
        grad[idx - 1] -= tau * diff2;
        grad[idx]     += 2.0 * tau * diff2;
        grad[idx + 1] -= tau * diff2;
        H.add(idx - 1, idx - 1, tau);
        H.add(idx,     idx,     4.0 * tau);
        H.add(idx + 1, idx + 1, tau);
        H.add(idx,     idx - 1, -2.0 * tau);
        H.add(idx + 1, idx,     -2.0 * tau);
        H.add(idx + 1, idx - 1, tau);
    }
}

void add_ar1_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start_idx, int n_times, double tau, double rho
) {
    if (n_times < 1) return;
    double tau_marginal = tau * (1.0 - rho * rho);
    int idx0 = start_idx;
    grad[idx0] -= tau_marginal * x[idx0];
    H.add(idx0, idx0, tau_marginal);
    for (int t = 1; t < n_times; t++) {
        int idx = start_idx + t;
        int idx_prev = start_idx + t - 1;
        double resid = x[idx] - rho * x[idx_prev];
        grad[idx]      -= tau * resid;
        grad[idx_prev] += tau * rho * resid;
        H.add(idx,      idx,      tau);
        H.add(idx_prev, idx_prev, tau * rho * rho);
        H.add(idx,      idx_prev, -tau * rho);
    }
}

void add_rw1_pattern(
    std::vector<std::pair<int,int>>& out,
    int start_idx, int n_times, bool cyclic
) {
    for (int t = 1; t < n_times; t++) {
        out.emplace_back(start_idx + t, start_idx + t - 1);  // lower-tri (hi, lo)
    }
    if (cyclic && n_times > 1) {
        out.emplace_back(start_idx + n_times - 1, start_idx);
    }
}

void add_rw2_pattern(
    std::vector<std::pair<int,int>>& out,
    int start_idx, int n_times, bool /*cyclic*/
) {
    if (n_times < 3) return;
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        out.emplace_back(idx,     idx - 1);
        out.emplace_back(idx + 1, idx);
        out.emplace_back(idx + 1, idx - 1);
    }
}

void add_ar1_pattern(
    std::vector<std::pair<int,int>>& out,
    int start_idx, int n_times
) {
    for (int t = 1; t < n_times; t++) {
        out.emplace_back(start_idx + t, start_idx + t - 1);
    }
}

} // namespace tulpa
