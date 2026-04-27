// lkj_chol_helpers.h
// Single source of truth for the tanh-parameterized LKJ-Cholesky machinery
// shared by HMC, the fast-path samplers (mclmc/pathfinder/DA/SMC), and the
// generic Gibbs kernel.
//
// Parameterization
// ----------------
// For an n_coefs-dimensional correlated random-effect term, the strict-lower
// entries of the Cholesky factor L of the correlation matrix R = L L^T are
// parameterized by `n*(n-1)/2` unconstrained raw values:
//
//   L[i, j] = tanh(raw[idx])         for j < i, idx in row-major strict-lower
//   L[i, i] = sqrt(1 - sum_{j<i} L[i, j]^2)
//
// Rows of L have unit norm by construction, so R = L L^T is a valid
// correlation matrix.
//
// Random effects use the non-centered form
//   u_eff[g, c] = sigma[c] * (L z)[g, c],   z[g, c] ~ N(0, 1).
//
// Priors
// ------
// - Half-Cauchy(0, scale) on each sigma_c (handled outside this header).
// - LKJ(eta) on R, written in raw-Cholesky coordinates as
//     log p(L) = sum_k (eta - 1 + (n - k - 1)/2) * 2 * log(L[k, k])
//                 + sum_{k>=1} (n - k) * log(L[k, k])     (L -> correlation Jacobian)
//                 + sum_{i>j} log(sech^2(raw[i, j]))      (tanh Jacobian).
//
// Gradient convention: all "_add" / "_grad_add" helpers are ADDITIVE — they
// increment the caller's gradient buffer rather than overwriting it. Callers
// that need overwrite semantics should zero the relevant slots first.

#ifndef TULPA_LKJ_CHOL_HELPERS_H
#define TULPA_LKJ_CHOL_HELPERS_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace tulpa {

// Build lower-triangular L from raw chol params via tanh + row-norm-1.
// L_flat is row-major n x n; only the lower triangle (incl. diagonal) is set.
// Strict-upper entries are not touched.
//
// If log_jac_tanh != nullptr, *log_jac_tanh is INCREMENTED by
//   sum_{i>j} log(sech^2(raw[i, j])).
//
// Returns false if any row's squared diagonal would be < 1e-10 (numerical
// constraint violation); L_flat may be partially written in that case.
inline bool build_L_from_raw(const double* raw, int n,
                             double* L_flat,
                             double* log_jac_tanh = nullptr) {
    int idx = 0;
    for (int i = 0; i < n; i++) {
        double row_sum_sq = 0.0;
        for (int j = 0; j < i; j++) {
            double l_ij = std::tanh(raw[idx]);
            L_flat[i * n + j] = l_ij;
            row_sum_sq += l_ij * l_ij;
            if (log_jac_tanh) {
                double sech2 = 1.0 - l_ij * l_ij;
                *log_jac_tanh += std::log(std::max(1e-300, sech2));
            }
            idx++;
        }
        double diag_sq = 1.0 - row_sum_sq;
        if (diag_sq < 1e-10) return false;
        L_flat[i * n + i] = std::sqrt(diag_sq);
    }
    return true;
}

// Log-density contribution from L diagonals: LKJ(eta) on R = L L^T plus the
// L -> correlation Jacobian. Excludes the tanh Jacobian (handled by
// build_L_from_raw via its log_jac_tanh out-param).
inline double lkj_log_prior_density(const double* L_flat, int n, double eta) {
    double lp = 0.0;
    for (int k = 0; k < n; k++) {
        double L_kk = L_flat[k * n + k];
        lp += (eta - 1.0 + (n - k - 1) / 2.0) * 2.0 * std::log(L_kk);
    }
    for (int k = 1; k < n; k++) {
        double L_kk = L_flat[k * n + k];
        lp += (n - k) * std::log(L_kk);
    }
    return lp;
}

// Additive gradient on raw_chol from the LKJ + L->correlation Jacobian + tanh
// Jacobian (i.e., everything in the prior except the half-Cauchy on sigma).
inline void lkj_log_prior_grad_add(const double* raw, const double* L_flat, int n,
                                   double eta, double* grad_raw) {
    // tanh-Jacobian gradient: d/draw of log(sech^2(raw)) = -2 tanh(raw)
    int chol_idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            grad_raw[chol_idx] += -2.0 * std::tanh(raw[chol_idx]);
            chol_idx++;
        }
    }
    // LKJ + L->R Jacobian via L_kk chain rule:
    //   d log L[k,k] / d L[k,j] = -L[k,j] / L[k,k]^2,    d L[k,j] / d raw = sech^2(raw)
    for (int i = 1; i < n; i++) {
        double L_ii = L_flat[i * n + i];
        double coef = (eta - 1.0 + (n - i - 1) / 2.0) * 2.0 + (n - i);
        int chol_base = i * (i - 1) / 2;
        for (int j = 0; j < i; j++) {
            double L_ij = L_flat[i * n + j];
            double l_val = std::tanh(raw[chol_base + j]);
            double sech2 = 1.0 - l_val * l_val;
            grad_raw[chol_base + j] += coef * (-L_ij / (L_ii * L_ii)) * sech2;
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
                Lz_c += L_flat[c * n + k] * z[g * n + k];
            }
            u_eff[g * n + c] = sigma[c] * Lz_c;
        }
    }
}

// Likelihood chain rule for the non-centered tanh-Cholesky LKJ parameterization.
//
// Inputs:
//   L_flat, sigma, z, raw, u_eff: as built by build_L_from_raw / compute_u_eff
//   glik: row-major n_groups x n, accumulated dLL/d(u_eff[g, c])
//
// Adds (additively) to:
//   grad_z[g*n + c]            - dLL/dz[g, k] = sum_{c>=k} glik[g, c] * sigma[c] * L[c, k]
//   grad_log_sigma[c]          - dLL/dlog_sigma[c] = sum_g glik[g, c] * u_eff[g, c]
//   grad_raw[idx]              - dLL/draw via dLL/dL[ii, jj] * sech^2(raw)
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
                gz += glik[g * n + c] * sigma[c] * L_flat[c * n + k];
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
    for (int ii = 1; ii < n; ii++) {
        double L_ii = L_flat[ii * n + ii];
        std::vector<double> S_i(ii + 1, 0.0);
        for (int k = 0; k <= ii; k++) {
            for (int g = 0; g < n_groups; g++) {
                S_i[k] += glik[g * n + ii] * z[g * n + k];
            }
        }
        int chol_base = ii * (ii - 1) / 2;
        for (int jj = 0; jj < ii; jj++) {
            double L_ij = L_flat[ii * n + jj];
            double grad_L_ij = sigma[ii] * (S_i[jj] - S_i[ii] * L_ij / L_ii);
            double l_val = std::tanh(raw[chol_base + jj]);
            double sech2 = 1.0 - l_val * l_val;
            grad_raw[chol_base + jj] += grad_L_ij * sech2;
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
                r += L_flat[ii * n + k] * L_flat[jj * n + k];
            }
            R_flat[ii * n + jj] = r;
        }
    }
}

}  // namespace tulpa

#endif  // TULPA_LKJ_CHOL_HELPERS_H
