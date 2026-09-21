// re_term_prior.h
// The prior of one random-effect term and the effect values it implies, for
// the engine's random-effect block and for a consumer package that carries a
// random effect of its own among its extra parameters. One implementation, so
// the two put the same density on the same parameter vector.
//
// A term has `n_groups` groups, each with `n_coefs` coefficients (an intercept
// and / or slopes), and reads from `params`:
//
//   * log_sigma_idx[c]  the log SD of coefficient c, with a half-Cauchy(0,
//                       sigma_scale) prior on the SD and the log Jacobian;
//   * chol_start        the first of the n_coefs (n_coefs - 1) / 2 raw values of
//                       the correlation's Cholesky factor (build_L_from_raw),
//                       with an LKJ(2) prior; -1 for an uncorrelated term;
//   * re_start          the n_groups * n_coefs latent values, group-major.
//
// A correlated term is non-centered, re[g, c] = sigma[c] (L z[g])[c] with
// z ~ N(0, I). An uncorrelated term is non-centered (re = sigma z) when
// `noncentered`, and otherwise samples re ~ N(0, sigma^2) directly.

#ifndef TULPA_RE_TERM_PRIOR_H
#define TULPA_RE_TERM_PRIOR_H

#include <vector>
#include "ad_scalar_math.h"
#include "lkj_chol.h"

namespace tulpa {
namespace priors {

// Is the term correlated: a Cholesky block and more than one coefficient.
inline bool re_term_correlated(int chol_start, int n_coefs) {
    return chol_start >= 0 && n_coefs > 1;
}

// The term's SDs, sigmas[c] = exp(params[log_sigma_idx[c]]), and for a
// correlated term its correlation Cholesky factor into `L_flat` (row-major
// n_coefs x n_coefs, zero-initialized by the caller), adding the raw -> L log
// Jacobian to `*log_jac` when it is non-null.
template <typename T>
inline void re_term_scales(const T* params, const int* log_sigma_idx,
                           int n_coefs, int chol_start, T* sigmas, T* L_flat,
                           T* log_jac = nullptr) {
    for (int c = 0; c < n_coefs; c++) {
        sigmas[c] = math::safe_exp(params[log_sigma_idx[c]]);
    }
    if (re_term_correlated(chol_start, n_coefs)) {
        build_L_from_raw(params + chol_start, n_coefs, L_flat, log_jac);
    }
}

// The non-centered effect of group g, out[c] = sigma[c] (L z[g])[c], or
// sigma[c] z[g, c] for an uncorrelated term, from the scales re_term_scales()
// returned.
template <typename T>
inline void re_term_group_effect(const T* params, const T* sigmas,
                                 const T* L_flat, int n_coefs, bool correlated,
                                 int re_start, int g, T* out) {
    if (correlated) {
        for (int c = 0; c < n_coefs; c++) {
            T Lz_c = T(0.0);
            for (int k = 0; k <= c; k++) {
                Lz_c = Lz_c + L_flat[c * n_coefs + k]
                       * params[re_start + g * n_coefs + k];
            }
            out[c] = sigmas[c] * Lz_c;
        }
    } else {
        for (int c = 0; c < n_coefs; c++) {
            out[c] = sigmas[c] * params[re_start + g * n_coefs + c];
        }
    }
}

// The term's prior, added to `log_post` term by term so a caller summing
// several terms keeps its summation order, and its effect values into
// `re_vals` (n_groups * n_coefs, group-major).
template <typename T>
inline void re_term_log_prior_add(const T* params, const int* log_sigma_idx,
                                  int n_coefs, int chol_start, int re_start,
                                  int n_groups, bool noncentered,
                                  double sigma_scale, T& log_post,
                                  T* re_vals) {
    const bool correlated = re_term_correlated(chol_start, n_coefs);
    std::vector<T> sigmas(n_coefs);
    std::vector<T> L_flat(correlated ? n_coefs * n_coefs : 0, T(0.0));
    // Every raw vector is in support, so there is no -Inf wall for the
    // sampler to meet.
    T log_jac = T(0.0);
    re_term_scales(params, log_sigma_idx, n_coefs, chol_start, sigmas.data(),
                   L_flat.data(), &log_jac);

    for (int c = 0; c < n_coefs; c++) {
        log_post = log_post
                   + math::log_prior_half_cauchy(params[log_sigma_idx[c]],
                                                 sigma_scale);
    }
    if (correlated) {
        log_post = log_post + log_jac;
        lkj_cholesky_log_density_add(L_flat.data(), n_coefs, T(2.0), log_post);
    }

    if (correlated || noncentered) {
        for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
                T z_gc = params[re_start + g * n_coefs + c];
                log_post = log_post - T(0.5) * z_gc * z_gc;
            }
            re_term_group_effect(params, sigmas.data(), L_flat.data(), n_coefs,
                                 correlated, re_start, g,
                                 re_vals + g * n_coefs);
        }
    } else {
        for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
                T re_val = params[re_start + g * n_coefs + c];
                T tau_c = T(1.0) / (sigmas[c] * sigmas[c] + T(1e-10));
                log_post = log_post - T(0.5) * tau_c * re_val * re_val;
                log_post = log_post + T(0.5) * math::safe_log(tau_c);
                re_vals[g * n_coefs + c] = re_val;
            }
        }
    }
}

}  // namespace priors
}  // namespace tulpa

#endif  // TULPA_RE_TERM_PRIOR_H
