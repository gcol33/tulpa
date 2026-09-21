// lkj_chol.h
// The partial-correlation Cholesky map and the LKJ density on it, templated
// over the scalar so the engine's random-effect prior and a consumer
// package's differentiate one implementation. tulpa's src/lkj_chol_helpers.h
// includes this header and adds the double-only gradient helpers.
//
// Parameterization
// ----------------
// For an n-dimensional correlated random-effect term, the Cholesky factor L of
// the correlation matrix R = L L^T is built from `n*(n-1)/2` unconstrained raw
// values through the canonical partial correlations z = tanh(raw):
//
//   L[0, 0] = 1
//   L[i, j] = z[i, j] * sqrt(s[i, j]),   s[i, j] = prod_{k<j} (1 - z[i, k]^2)
//   L[i, i] = sqrt(s[i, i])
//
// Each row then has unit norm for ANY raw vector: s[i, j] is a product of
// factors in (0, 1], so the map is onto the whole cone of correlation Cholesky
// factors and back. Writing L[i, j] = tanh(raw) directly instead bounds nothing
// but the individual entries -- row 2 with raw = (atanh(0.8), atanh(0.8)) has
// row sum of squares 1.28 -- so at n >= 3 that map covers a strict subset of the
// cone and the sampler meets its complement as a wall. n = 2 is the one case
// where the two maps coincide, which is why the defect starts at n = 3.
//
// s is carried as a running PRODUCT rather than as 1 - sum_{k<j} L[i, k]^2. The
// two are algebraically equal; the product is strictly positive by construction
// where the difference cancels.
//
// Random effects use the non-centered form
//   u_eff[g, c] = sigma[c] * (L z)[g, c],   z[g, c] ~ N(0, 1).
//
// Priors
// ------
// - Half-Cauchy(0, scale) on each sigma_c (handled outside this header).
// - LKJ(eta) on R, written in raw coordinates as
//     log p(L)   = sum_k (eta - 1 + (n - k - 1)/2) * 2 * log(L[k, k])
//     log|dL/draw| = sum_{i>j} [ log(1 - z[i,j]^2) + 0.5 * log(s[i, j]) ]
//   The L[k,k] exponent is the complete Stan lkj_corr_cholesky_lpdf: it already
//   folds the correlation -> Cholesky Jacobian sum_k (n-k) log L[k,k] into
//   det(R)^(eta-1). Adding that Jacobian a second time tilts the effective
//   prior to LKJ(eta + 0.5) on a 2x2 block. The 0.5 * log(s) term is the
//   scaling half of the transform above and is absent from a direct-tanh map.
//

#ifndef TULPA_LKJ_CHOL_H
#define TULPA_LKJ_CHOL_H

#include <cstddef>
#include "ad_scalar_math.h"

namespace tulpa {

// Index of the (i, j) strict-lower entry in the row-major raw vector:
// (1,0), (2,0), (2,1), (3,0), ...
inline int lkj_raw_row_base(int i) { return i * (i - 1) / 2; }

// Build lower-triangular L from raw params. L_flat is row-major n x n and must
// be zero-initialized by the caller; only the lower triangle (incl. diagonal)
// is written. Strict-upper entries are not touched.
//
// If log_jac != nullptr, *log_jac is INCREMENTED by the FULL log-Jacobian of
// raw -> L, i.e. the tanh term and the scaling term together.
//
// Succeeds for every raw vector, which is the point of the parameterization:
// there is no in-support / out-of-support split for a caller to disagree about.
// Templated over the scalar so the AD paths differentiate the same build the
// double paths evaluate.
template <typename T>
inline void build_L_from_raw(const T* raw, int n, T* L_flat,
                             T* log_jac = nullptr) {
    if (n <= 0) return;
    L_flat[0] = T(1.0);
    int idx = 0;
    for (int i = 1; i < n; i++) {
        T s = T(1.0);                       // s[i, j], updated as j advances
        for (int j = 0; j < i; j++) {
            const T z = math::safe_tanh(raw[idx]);
            const T one_m_z2 = T(1.0) - z * z;
            L_flat[(std::size_t)i * n + j] = z * math::safe_sqrt(s);
            if (log_jac) {
                *log_jac = *log_jac
                    + math::safe_log(math::safe_max(one_m_z2, T(1e-300)))
                    + T(0.5) * math::safe_log(math::safe_max(s, T(1e-300)));
            }
            s = s * one_m_z2;
            idx++;
        }
        L_flat[(std::size_t)i * n + i] = math::safe_sqrt(s);
    }
}

// LKJ(eta) log-density of the correlation R = L L', written on its Cholesky
// factor L (row-major n x n, as build_L_from_raw writes it). The exponent
// (2 eta - 2 + (n - k - 1)) on log L[k, k] is the complete Stan
// lkj_corr_cholesky_lpdf: det(R)^(eta - 1) plus the exact correlation ->
// Cholesky Jacobian sum_k (n - k) log L[k, k]. Combined with build_L_from_raw's
// raw -> L Jacobian it is the full change of variables to raw space; adding a
// second Cholesky -> correlation Jacobian would tilt the effective prior to
// LKJ(eta + 0.5) on a 2x2 block. The terms are added to `log_post` in order,
// so a caller accumulating a larger log-density keeps its summation order.
template <typename T>
inline void lkj_cholesky_log_density_add(const T* L_flat, int n, const T& eta,
                                         T& log_post) {
    for (int k = 0; k < n; k++) {
        T L_kk = L_flat[k * n + k];
        log_post = log_post + (eta - T(1.0)
                   + T((n - k - 1) / 2.0)) * T(2.0) * math::safe_log(L_kk);
    }
}

}  // namespace tulpa

#endif  // TULPA_LKJ_CHOL_H
