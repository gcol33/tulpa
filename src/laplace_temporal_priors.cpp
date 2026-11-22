// laplace_temporal_priors.cpp
// Temporal precision helpers for Laplace engines.

#include "laplace_temporal_priors.h"
#include "sparse_hessian.h"
#include "laplace_s2z.h"        // add_s2z_pin* / s2z_pin_quad
#include "laplace_hessian_target.h"
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
    return tulpa_temporal::gmrf_log_norm(1, std::log(tau));
}

// Coefficient k of the order-K difference operator over the window
// (x[p_0], ..., x[p_K]): (-1, 1) for RW1, (1, -2, 1) for RW2.
template <int Order> inline double rw_coef(int k);
template <> inline double rw_coef<1>(int k) { return k == 0 ? -1.0 : 1.0; }
template <> inline double rw_coef<2>(int k) { return k == 1 ? -2.0 : 1.0; }

// One difference term d = sum_k c_k x[p_k] of tau/2 d^2 into grad and H:
// grad[p_k] -= tau c_k d, H[p_k][p_l] += tau c_k c_l.
template <int Order, class Hess>
inline void add_rw_term(DenseVec& grad, Hess& H, const NumericVector& x,
                        const int (&p)[Order + 1], double tau) {
    double diff = rw_coef<Order>(0) * x[p[0]];
    for (int k = 1; k <= Order; k++) diff += rw_coef<Order>(k) * x[p[k]];
    for (int k = 0; k <= Order; k++) {
        grad[p[k]] -= (tau * rw_coef<Order>(k)) * diff;
    }
    for (int k = 0; k <= Order; k++) {
        const double ck = rw_coef<Order>(k);
        hess_add_sym(H, p[k], p[k], tau * (ck * ck));
    }
    for (int k = 0; k <= Order; k++) {
        for (int l = k + 1; l <= Order; l++) {
            hess_add_sym(H, p[k], p[l],
                         tau * (rw_coef<Order>(k) * rw_coef<Order>(l)));
        }
    }
}

// The order-K random-walk precision of one walk of n_times at start_idx: one
// term per window of K + 1 consecutive times, and on a cyclic walk the K
// windows that wrap past the last time.
template <int Order, class Hess>
void add_rw_precision(DenseVec& grad, Hess& H, const NumericVector& x,
                      int start_idx, int n_times, double tau, bool cyclic) {
    constexpr int W = Order + 1;
    if (n_times < W) return;
    int p[W];
    for (int t = 0; t + W <= n_times; t++) {
        for (int k = 0; k < W; k++) p[k] = start_idx + t + k;
        add_rw_term<Order>(grad, H, x, p, tau);
    }
    if (cyclic) {
        for (int t = n_times - W + 1; t < n_times; t++) {
            for (int k = 0; k < W; k++) p[k] = start_idx + (t + k) % n_times;
            add_rw_term<Order>(grad, H, x, p, tau);
        }
    }
}

template <int Order, class Hess>
void add_rw_field_impl(DenseVec& grad, Hess& H, const NumericVector& x,
                       int start, int n_groups, int n_times, double tau,
                       bool cyclic) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        add_rw_precision<Order>(grad, H, x, s, n_times, tau, cyclic);
    });
    add_s2z_pin_to(grad, H, x, start, static_cast<const int*>(nullptr),
                   n_groups * n_times, tau, S2ZStorage::Fold);
}

template <class Hess>
void add_rw_field_dispatch(int order, DenseVec& grad, Hess& H,
                           const NumericVector& x, int start, int n_groups,
                           int n_times, double tau, bool cyclic) {
    if (order == 2) {
        add_rw_field_impl<2>(grad, H, x, start, n_groups, n_times, tau, cyclic);
    } else {
        add_rw_field_impl<1>(grad, H, x, start, n_groups, n_times, tau, cyclic);
    }
}

} // anonymous namespace

void add_ar1_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, double rho
) {
    if (n_times < 1) return;
    double tau_marginal = tau * tulpa_temporal::ar1_one_minus_rho2(rho);
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

void add_ar1_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start_idx, int n_times, double tau, double rho
) {
    if (n_times < 1) return;
    double tau_marginal = tau * tulpa_temporal::ar1_one_minus_rho2(rho);
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

void add_rw_field(
    int order, DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    add_rw_field_dispatch(order, grad, H, x, start, n_groups, n_times, tau,
                          cyclic);
}

void add_rw_field(
    int order, DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
) {
    add_rw_field_dispatch(order, grad, H, x, start, n_groups, n_times, tau,
                          cyclic);
}

void add_rw_field_pattern(
    int order, std::vector<std::pair<int,int>>& out,
    int start, int n_groups, int n_times, bool cyclic
) {
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        if (order == 2) add_rw2_pattern(out, s, n_times, cyclic);
        else            add_rw1_pattern(out, s, n_times, cyclic);
    });
    add_s2z_pin_pattern(out, start, n_groups * n_times, S2ZStorage::Fold);
}

// A non-cyclic RW2 keeps its per-group LINEAR null direction, which the pin
// does not touch: rank(Q_aug) = sum_g rank(Q_g) + 1 either way, and
// log_prior_rw2 already carries the per-group rank that reflects it.
double log_prior_rw_field(
    int order, const NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic,
    double log_pdet
) {
    const TemporalType type = order == 2 ? TemporalType::RW2 : TemporalType::RW1;
    double lp = 0.0;
    for_each_temporal_group(start, n_groups, n_times, [&](int s) {
        lp += log_prior_temporal_span(x, s, n_times, type, tau, 0.0, cyclic);
    });
    return lp + s2z_pin_log_norm(tau)
         - 0.5 * s2z_pin_quad(x, start, n_groups * n_times, tau)
         + 0.5 * n_groups * log_pdet;
}

} // namespace tulpa
