// lkj_chol_helpers.h
// Single source of truth for the LKJ-Cholesky machinery shared by the Laplace
// spec solver, HMC, the fast-path samplers (mclmc/pathfinder/DA/SMC) and the
// generic Gibbs kernel.
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

#include "autodiff_utils.h"
#include "tulpa/lkj_chol.h"  // parameterization, build_L_from_raw, LKJ density

namespace tulpa {

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
