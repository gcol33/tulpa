// lkj_chol_helpers.h
// Single source of truth for the LKJ-Cholesky machinery shared by the Laplace
// spec solver, HMC, the fast-path samplers (mclmc/pathfinder/DA/SMC) and the
// generic Gibbs kernel.
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
// Gradient convention: all "_add" / "_grad_add" helpers are ADDITIVE -- they
// increment the caller's gradient buffer rather than overwriting it. Callers
// that need overwrite semantics should zero the relevant slots first.

#ifndef TULPA_LKJ_CHOL_HELPERS_H
#define TULPA_LKJ_CHOL_HELPERS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "autodiff_utils.h"   // safe_tanh / safe_log / safe_sqrt / safe_max

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

// Inverse of build_L_from_raw: recover the raw values from a correlation
// Cholesky factor (row-major n x n, unit-norm rows). raw_out is filled with
// n*(n-1)/2 entries in the same row-major strict-lower order.
//
// z[i, j] = L[i, j] / sqrt(s[i, j]) with s carried the same way as the forward
// build, so the round trip is exact up to the atanh domain clamp.
inline void raw_from_L(const double* L_flat, int n, double* raw_out) {
    int idx = 0;
    for (int i = 1; i < n; i++) {
        double s = 1.0;
        for (int j = 0; j < i; j++) {
            double z = (s > 0.0) ? L_flat[(std::size_t)i * n + j] / std::sqrt(s)
                                 : 0.0;
            // Negated comparisons so a NaN entry lands on a boundary instead
            // of passing both tests and reaching atanh.
            if (!(z > -0.999999)) z = -0.999999;
            if (!(z <  0.999999)) z =  0.999999;
            raw_out[idx++] = std::atanh(z);
            s *= (1.0 - z * z);
        }
    }
}

// Log-density contribution from L diagonals: the complete Stan
// lkj_corr_cholesky_lpdf on R = L L^T (the L[k,k] exponent already includes the
// correlation -> Cholesky Jacobian). Excludes the raw -> L Jacobian (handled by
// build_L_from_raw via its log_jac out-param).
inline double lkj_log_prior_density(const double* L_flat, int n, double eta) {
    double lp = 0.0;
    for (int k = 0; k < n; k++) {
        double L_kk = L_flat[(std::size_t)k * n + k];
        lp += (eta - 1.0 + (n - k - 1) / 2.0) * 2.0 * std::log(L_kk);
    }
    return lp;
}

// Additive gradient on raw from the LKJ density plus the raw -> L Jacobian
// (i.e. everything in the prior except the half-Cauchy on sigma).
//
// Both pieces reduce to sums of log(1 - z_m^2) in this parameterization, which
// is what collapses the gradient to one term per raw entry:
//
//   row i's LKJ term    = coef_i * sum_{k<i} log(1 - z_k^2),  coef_i as above
//   row i's Jacobian    = sum_{j<i} [ log(1 - z_j^2) + 0.5 * sum_{k<j} log(1 - z_k^2) ]
//
// Differentiating both at raw_m (column m of row i) with
// d log(1 - z_m^2) / d raw_m = -2 z_m gives
//
//   -z_m * [ 2 + (i - 1 - m) + 2 * coef_i ] = -z_m * (n + 2 eta - m - 2),
//
// which is independent of the row.
inline void lkj_log_prior_grad_add(const double* raw, int n, double eta,
                                   double* grad_raw) {
    for (int i = 1; i < n; i++) {
        const int base = lkj_raw_row_base(i);
        for (int m = 0; m < i; m++) {
            const double z = std::tanh(raw[base + m]);
            grad_raw[base + m] += -z * (n + 2.0 * eta - m - 2.0);
        }
    }
}

// u_eff[g, c] = sigma[c] * (L z)[g, c], with z and u_eff stored row-major
// of shape (n_groups, n).
inline void compute_u_eff(const double* L_flat, int n,
                          const double* sigma, const double* z,
                          int n_groups, double* u_eff) {
    for (int g = 0; g < n_groups; g++) {
        for (int c = 0; c < n; c++) {
            double Lz_c = 0.0;
            for (int k = 0; k <= c; k++) {
                Lz_c += L_flat[(std::size_t)c * n + k] * z[g * n + k];
            }
            u_eff[g * n + c] = sigma[c] * Lz_c;
        }
    }
}

// Likelihood chain rule for the non-centered LKJ-Cholesky parameterization.
//
// Inputs:
//   L_flat, sigma, z, raw, u_eff: as built by build_L_from_raw / compute_u_eff
//   glik: row-major n_groups x n, accumulated dLL/d(u_eff[g, c])
//
// Adds (additively) to:
//   grad_z[g*n + c]   - dLL/dz[g, k] = sum_{c>=k} glik[g, c] * sigma[c] * L[c, k]
//   grad_log_sigma[c] - dLL/dlog_sigma[c] = sum_g glik[g, c] * u_eff[g, c]
//   grad_raw[idx]     - dLL/draw
//
// The raw channel uses the two derivatives of the transform. Within row i, with
// G_ij = sigma_i * S_i[j] the derivative in L treating its entries as free:
//
//   dL[i,m]/draw_m = sqrt(s_m) * (1 - z_m^2)
//   dL[i,j]/draw_m = -z_m * L[i,j]   for every j > m, the diagonal included,
//
// because raw_m enters those entries only through the shared factor sqrt(s_j),
// and d sqrt(s_j)/draw_m = -z_m sqrt(s_j). Summing the second over j > m is one
// suffix sum per row.
inline void chol_nc_chain_rule_add(const double* L_flat, int n,
                                   const double* sigma, const double* z,
                                   const double* raw, const double* u_eff,
                                   int n_groups, const double* glik,
                                   double* grad_z, double* grad_log_sigma,
                                   double* grad_raw) {
    for (int g = 0; g < n_groups; g++) {
        for (int k = 0; k < n; k++) {
            double gz = 0.0;
            for (int c = k; c < n; c++) {
                gz += glik[g * n + c] * sigma[c] * L_flat[(std::size_t)c * n + k];
            }
            grad_z[g * n + k] += gz;
        }
    }
    for (int c = 0; c < n; c++) {
        double gs = 0.0;
        for (int g = 0; g < n_groups; g++) {
            gs += glik[g * n + c] * u_eff[g * n + c];
        }
        grad_log_sigma[c] += gs;
    }
    std::vector<double> S_i, suffix;
    for (int i = 1; i < n; i++) {
        S_i.assign(i + 1, 0.0);
        for (int k = 0; k <= i; k++) {
            for (int g = 0; g < n_groups; g++) {
                S_i[k] += glik[g * n + i] * z[g * n + k];
            }
        }
        // suffix[j] = sum_{j' >= j} S_i[j'] * L[i, j'], over j' up to i.
        suffix.assign(i + 2, 0.0);
        for (int j = i; j >= 0; j--) {
            suffix[j] = suffix[j + 1] + S_i[j] * L_flat[(std::size_t)i * n + j];
        }
        const int base = lkj_raw_row_base(i);
        double s = 1.0;
        for (int m = 0; m < i; m++) {
            const double z_m = std::tanh(raw[base + m]);
            const double one_m_z2 = 1.0 - z_m * z_m;
            grad_raw[base + m] += sigma[i] *
                (S_i[m] * std::sqrt(s) * one_m_z2 - z_m * suffix[m + 1]);
            s *= one_m_z2;
        }
    }
}

// R = L L^T in row-major n x n.
inline void correlation_from_L(const double* L_flat, int n, double* R_flat) {
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            double r = 0.0;
            int kmax = std::min(ii, jj);
            for (int k = 0; k <= kmax; k++) {
                r += L_flat[(std::size_t)ii * n + k] * L_flat[(std::size_t)jj * n + k];
            }
            R_flat[(std::size_t)ii * n + jj] = r;
        }
    }
}

}  // namespace tulpa

#endif  // TULPA_LKJ_CHOL_HELPERS_H
