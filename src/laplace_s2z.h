// laplace_s2z.h
// Sum-to-zero identification of an intrinsic field's constant null direction,
// for the Laplace-path kernels.
//
// The convention is inst/include/tulpa/sum_to_zero.h: augment the precision
//
//     Q_aug = Q + (1_c 1_c') / J_c        (scaled by the field precision tau)
//
// so the constant direction carries tau where Q had a zero eigenvalue, and the
// augmented block is full rank on that direction.
//
// On the Laplace path the field is NOT removed from eta. Pinning the constant
// to ~0 during the Newton solve forces the level into the intercept, where the
// intercept prior acts. This is what makes the pin load-bearing for inference
// and not only for the mode: a path that merely centres the mode afterwards
// (latent_block.h `center_intercept`, folding the removed mean into the
// intercept to keep eta invariant) gets the point estimate right and leaves the
// (intercept, field level) direction flat in the Hessian, so every fixed-effect
// standard error read off that Hessian collapses to the fixed-effect prior.
//
// The augmentation's 11' is dense, so how it is stored is a per-block choice
// (S2ZStorage below). Both storages are exact and agree; only the representation
// differs. Where the size predicate decides, the same predicate gates the
// pattern and the scatter so the two cannot disagree.

#ifndef TULPA_LAPLACE_S2Z_H
#define TULPA_LAPLACE_S2Z_H

#include "laplace_types.h"
#include "sparse_hessian.h"
#include "tulpa/sum_to_zero.h"   // s2z_aug_coef / s2z_aug_rank
#include <Rcpp.h>
#include <cstdlib>
#include <utility>
#include <vector>

namespace tulpa {

// Field size at or below which the augmentation's 11' is stored exactly rather
// than folded in at solve time.
constexpr int S2Z_DENSIFY_MAX = 256;

inline int s2z_densify_max() {
    // Power-user / test override of the densify-vs-rank-1 cutoff. A value of 0
    // forces the rank-1 (Woodbury) path on every intrinsic field; a huge value
    // forces densify. The same env var is seen by the pattern and the scatter
    // so they never disagree within a fit.
    const char* e = std::getenv("TULPA_S2Z_DENSIFY_MAX");
    if (e && *e) {
        const int v = std::atoi(e);
        if (v >= 0) return v;
    }
    return S2Z_DENSIFY_MAX;
}

inline bool s2z_densify(int n_units) {
    return n_units <= s2z_densify_max();
}

// How a block stores the augmentation's rank-1 11'.
//
//   Auto -- write it into the sparse pattern while the span is small enough to
//           afford it (s2z_densify), and register it for the solver to fold in
//           above that. For a block whose structural pattern is already dense
//           within a component, storing it costs little.
//   Fold -- always register it. For a BANDED block -- an RW chain, whose
//           pattern is tridiagonal (RW1) or pentadiagonal (RW2) -- storing 11'
//           would turn O(n) entries into O(n^2) and contradict the block's
//           declared ADJACENCY fill kind.
//
// The two agree: verified on a 400-node RW1 field, where the fixed-effect
// standard errors match to five significant figures across the cutoff.
enum class S2ZStorage { Auto, Fold };

// Sum over a component's nodes. `idx` is the field-local node list (nullptr for
// the contiguous run [start, start + n)); node k is at s2z_node(start, idx, k).
inline double s2z_span_sum(const Rcpp::NumericVector& x, int start,
                           const int* idx, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[s2z_node(start, idx, i)];
    return s;
}
inline double s2z_span_sum(const Rcpp::NumericVector& x, int start, int n) {
    return s2z_span_sum(x, start, static_cast<const int*>(nullptr), n);
}

// tau * (sum_{i in component} x_i)^2 / n -- the augmentation's contribution to
// the quadratic form. Enters a log-prior as -0.5 * this.
inline double s2z_pin_quad(const Rcpp::NumericVector& x, int start,
                           const int* idx, int n, double tau) {
    if (n <= 0) return 0.0;
    const double s = s2z_span_sum(x, start, idx, n);
    return s2z_aug_coef(tau, n) * s * s;
}
inline double s2z_pin_quad(const Rcpp::NumericVector& x, int start, int n,
                           double tau) {
    return s2z_pin_quad(x, start, static_cast<const int*>(nullptr), n, tau);
}

// Dense scatter of the pin over a component: exact rank-1 (tau/n) 11' on the
// component's nodes (contiguous, or the arbitrary `idx` set).
inline void add_s2z_pin(DenseVec& grad, DenseMat& H,
                        const Rcpp::NumericVector& x,
                        int start, const int* idx, int n, double tau) {
    if (n <= 0) return;
    const double lambda = s2z_aug_coef(tau, n);
    const double s = s2z_span_sum(x, start, idx, n);
    for (int i = 0; i < n; i++) {
        const int a = s2z_node(start, idx, i);
        grad[a] -= lambda * s;
        for (int j = 0; j < n; j++) H[a][s2z_node(start, idx, j)] += lambda;
    }
}
inline void add_s2z_pin(DenseVec& grad, DenseMat& H,
                        const Rcpp::NumericVector& x,
                        int start, int n, double tau) {
    add_s2z_pin(grad, H, x, start, static_cast<const int*>(nullptr), n, tau);
}

// Sparse twin. The exact gradient is always added; the 11' Hessian is stored
// when the component densifies and registered for the solver to fold in
// otherwise. The rank-1 fold carries the component's node list, so a
// non-contiguous component (a genuine disconnected map) folds exactly as a
// contiguous one.
inline void add_s2z_pin_sparse(DenseVec& grad, SparseHessianBuilder& H,
                               const Rcpp::NumericVector& x,
                               int start, const int* idx, int n, double tau,
                               S2ZStorage storage = S2ZStorage::Auto) {
    if (n <= 0) return;
    const double lambda = s2z_aug_coef(tau, n);
    const double s = s2z_span_sum(x, start, idx, n);
    for (int i = 0; i < n; i++) grad[s2z_node(start, idx, i)] -= lambda * s;
    if (storage == S2ZStorage::Auto && s2z_densify(n)) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                H.add(s2z_node(start, idx, i), s2z_node(start, idx, j), lambda);
    } else {
        H.add_s2z_rank1(start, n, idx, lambda);
    }
}
inline void add_s2z_pin_sparse(DenseVec& grad, SparseHessianBuilder& H,
                               const Rcpp::NumericVector& x,
                               int start, int n, double tau,
                               S2ZStorage storage = S2ZStorage::Auto) {
    add_s2z_pin_sparse(grad, H, x, start, static_cast<const int*>(nullptr), n,
                       tau, storage);
}

// Lower-triangle entries the stored 11' needs. Empty when the component is
// folded at solve time. Callers append this alongside their own structural
// pattern; where the block densifies it subsumes that structure, and the
// builder's entry map collapses the overlap. The storage argument must match
// the one the scatter was given, or the pattern and the values disagree.
inline void add_s2z_pin_pattern(std::vector<std::pair<int,int>>& out,
                                int start, const int* idx, int n,
                                S2ZStorage storage = S2ZStorage::Auto) {
    if (n <= 0 || storage == S2ZStorage::Fold || !s2z_densify(n)) return;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            out.emplace_back(s2z_node(start, idx, i), s2z_node(start, idx, j));
}
inline void add_s2z_pin_pattern(std::vector<std::pair<int,int>>& out,
                                int start, int n,
                                S2ZStorage storage = S2ZStorage::Auto) {
    add_s2z_pin_pattern(out, start, static_cast<const int*>(nullptr), n, storage);
}

}  // namespace tulpa

#endif  // TULPA_LAPLACE_S2Z_H
