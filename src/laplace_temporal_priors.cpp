// laplace_temporal_priors.cpp
// Temporal precision helpers for Laplace engines.

#include "laplace_temporal_priors.h"

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

} // namespace tulpa
