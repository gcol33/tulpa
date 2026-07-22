// laplace_temporal_priors.cpp
// Temporal precision helpers for Laplace engines.

#include "laplace_temporal_priors.h"
#include "sparse_hessian.h"
#include "laplace_s2z.h"        // add_s2z_pin* / s2z_pin_quad
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Rcpp;

namespace tulpa {

namespace {

// The G walks of one field, at [start + g*n_times, ...).
template <typename PerGroup>
inline void for_each_temporal_group(int start, int n_groups, int n_times,
                                    PerGroup&& f) {
    for (int g = 0; g < n_groups; g++) f(start + g * n_times);
}

// Normalizer the pin contributes. It fills exactly one direction -- the field's
// global constant -- so rank(Q_aug) = sum_g rank(Q_g) + 1, and the per-group
// log_prior_rw* calls supply the sum_g rank(Q_g) part.
inline double s2z_pin_log_norm(double tau) {
    return 0.5 * std::log(tau / (2.0 * M_PI));
}

} // anonymous namespace

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
    // One second-difference term (x[p] - 2 x[q] + x[r]) into grad and H.
    auto add_term = [&](int p, int q, int r) {
        double diff2 = x[p] - 2.0 * x[q] + x[r];
        grad[p] -= tau * diff2;
        grad[q] += 2.0 * tau * diff2;
        grad[r] -= tau * diff2;
        H[p][p] += tau;         H[q][q] += 4.0 * tau;   H[r][r] += tau;
        H[p][q] -= 2.0 * tau;   H[q][p] -= 2.0 * tau;
        H[q][r] -= 2.0 * tau;   H[r][q] -= 2.0 * tau;
        H[p][r] += tau;         H[r][p] += tau;
    };
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        add_term(idx - 1, idx, idx + 1);
    }
    if (cyclic) {
        // Close the ring: centres at t = n-1 and t = 0 with wrapped neighbours.
        int i0 = start_idx, iL = start_idx + n_times - 1;
        add_term(iL - 1, iL, i0);
        add_term(iL, i0, i0 + 1);
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
    int start_idx, int n_times, double tau, bool cyclic
) {
    if (n_times < 3) return;
    // SparseHessianBuilder::add normalizes (row, col) to the lower triangle, so
    // each unique off-diagonal edge is written once regardless of orientation.
    auto add_term = [&](int p, int q, int r) {
        double diff2 = x[p] - 2.0 * x[q] + x[r];
        grad[p] -= tau * diff2;
        grad[q] += 2.0 * tau * diff2;
        grad[r] -= tau * diff2;
        H.add(p, p, tau);
        H.add(q, q, 4.0 * tau);
        H.add(r, r, tau);
        H.add(p, q, -2.0 * tau);
        H.add(q, r, -2.0 * tau);
        H.add(p, r, tau);
    };
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        add_term(idx - 1, idx, idx + 1);
    }
    if (cyclic) {
        int i0 = start_idx, iL = start_idx + n_times - 1;
        add_term(iL - 1, iL, i0);
        add_term(iL, i0, i0 + 1);
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
    int start_idx, int n_times, bool cyclic
) {
    if (n_times < 3) return;
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        out.emplace_back(idx,     idx - 1);
        out.emplace_back(idx + 1, idx);
        out.emplace_back(idx + 1, idx - 1);
    }
    if (cyclic) {
        // Off-diagonal edges the two ring-closing terms add (as (hi, lo)).
        int i0 = start_idx, iL = start_idx + n_times - 1;
        out.emplace_back(iL,     iL - 1);
        out.emplace_back(iL,     i0);
        out.emplace_back(iL - 1, i0);
        out.emplace_back(i0 + 1, i0);
        out.emplace_back(iL,     i0 + 1);
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

// ============================================================
// Field-level entry points: per-group precision + the one global
// sum-to-zero pin. See the header for why the pin is global.
// ============================================================

void add_rw1_field(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw1_precision(grad, H, x, s, n_times, tau, cyclic);
    });
    add_s2z_pin(grad, H, x, start, n_groups * n_times, tau);
}

void add_rw1_field_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw1_precision_sparse(grad, H, x, s, n_times, tau, cyclic);
    });
    add_s2z_pin_sparse(grad, H, x, start, n_groups * n_times, tau,
                       S2ZStorage::Fold);
}

void add_rw1_field_pattern(
    std::vector<std::pair<int,int>>& out,
    int start, int n_groups, int n_times, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw1_pattern(out, s, n_times, cyclic);
    });
    add_s2z_pin_pattern(out, start, n_groups * n_times, S2ZStorage::Fold);
}

double log_prior_rw1_field(
    const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    double lp = 0.0;
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        lp += log_prior_rw1(x, s, n_times, tau, cyclic);
    });
    return lp + s2z_pin_log_norm(tau)
         - 0.5 * s2z_pin_quad(x, start, n_groups * n_times, tau);
}

void add_rw2_field(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw2_precision(grad, H, x, s, n_times, tau, cyclic);
    });
    add_s2z_pin(grad, H, x, start, n_groups * n_times, tau);
}

void add_rw2_field_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw2_precision_sparse(grad, H, x, s, n_times, tau, cyclic);
    });
    add_s2z_pin_sparse(grad, H, x, start, n_groups * n_times, tau,
                       S2ZStorage::Fold);
}

void add_rw2_field_pattern(
    std::vector<std::pair<int,int>>& out,
    int start, int n_groups, int n_times, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw2_pattern(out, s, n_times, cyclic);
    });
    add_s2z_pin_pattern(out, start, n_groups * n_times, S2ZStorage::Fold);
}

// A non-cyclic RW2 keeps its per-group LINEAR null direction, which the pin
// does not touch: rank(Q_aug) = sum_g rank(Q_g) + 1 either way, and
// log_prior_rw2 already carries the per-group rank that reflects it.
double log_prior_rw2_field(
    const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    double lp = 0.0;
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        lp += log_prior_rw2(x, s, n_times, tau, cyclic);
    });
    return lp + s2z_pin_log_norm(tau)
         - 0.5 * s2z_pin_quad(x, start, n_groups * n_times, tau);
}

} // namespace tulpa
