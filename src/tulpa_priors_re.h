// tulpa_priors_re.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_RE_H
#define TULPA_PRIORS_RE_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "lkj_chol_helpers.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 1. Random effects prior (multi-term, slopes, correlated slopes)
// ============================================================================

template<typename T>
T compute_re_prior(const std::vector<T>& params, const ModelData& data,
                   const ParamLayout& layout,
                   std::vector<T>& re_vals, std::vector<int>& re_term_offsets)
{
    T log_post = T(0.0);

    if (layout.has_re) {
        int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

        // Per-term dimensions, resolved once. The offset loop below needs them
        // to size re_vals and the density loop needs the same two numbers, so
        // they are stored rather than recomputed from the same ternaries.
        std::vector<int> n_groups_per_term(n_terms);
        std::vector<int> n_coefs_per_term(n_terms);
        int total_re_vals = 0;
        re_term_offsets.resize(n_terms);
        for (int t = 0; t < n_terms; t++) {
            re_term_offsets[t] = total_re_vals;
            n_groups_per_term[t] = (n_terms > 1 || data.n_re_terms > 0)
                                   ? data.re_n_groups_multi[t] : data.n_re_groups;
            n_coefs_per_term[t] = layout.has_re_slopes
                                  ? layout.re_n_coefs_multi[t] : 1;
            total_re_vals += n_groups_per_term[t] * n_coefs_per_term[t];
        }
        re_vals.resize(total_re_vals, T(0.0));

        for (int t = 0; t < n_terms; t++) {
            int n_groups_t = n_groups_per_term[t];
            int n_coefs_t = n_coefs_per_term[t];
            bool is_correlated = layout.has_re_slopes &&
                                 !layout.re_correlated_multi.empty() &&
                                 layout.re_correlated_multi[t];

            // Extract sigma parameters for this term
            std::vector<T> sigmas_t(n_coefs_t);
            for (int c = 0; c < n_coefs_t; c++) {
                int log_sigma_idx;
                if (layout.has_re_slopes) {
                    log_sigma_idx = layout.log_sigma_re_slopes[t][c];
                } else if (n_terms > 1) {
                    log_sigma_idx = layout.log_sigma_re_multi[t];
                } else {
                    log_sigma_idx = layout.log_sigma_re_idx;
                }
                T log_sigma = params[log_sigma_idx];
                sigmas_t[c] = safe_exp(log_sigma);

                // Half-Cauchy prior on sigma
                log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);
            }

            // Correlated slopes: partial-correlation Cholesky + LKJ prior
            std::vector<T> L_flat_t;
            if (is_correlated && n_coefs_t > 1) {
                int chol_start = layout.chol_re_start_multi[t];
                L_flat_t.resize(n_coefs_t * n_coefs_t, T(0.0));

                // The build and its log-Jacobian are tulpa::build_L_from_raw:
                // one implementation, differentiated here and evaluated by the
                // Laplace spec solver, so the two backends put the same Sigma
                // on the same raw vector. Every raw vector is in support, so
                // there is no -Inf wall for the sampler to meet.
                T log_jac = T(0.0);
                tulpa::build_L_from_raw(params.data() + chol_start, n_coefs_t,
                                        L_flat_t.data(), &log_jac);
                log_post = log_post + log_jac;

                // LKJ(eta=2) prior on R = L L', expressed directly on the
                // Cholesky factor. The exponent (2*eta - 2 + (n - k - 1)) on
                // log L_kk is the complete Stan lkj_corr_cholesky_lpdf: it is
                // det(R)^(eta-1) PLUS the exact correlation -> Cholesky Jacobian
                // sum_k (n-k) log L_kk. Combined with the raw -> L Jacobian
                // above it is the full change of variables to raw space; adding
                // a second Cholesky -> correlation Jacobian here would tilt the
                // effective prior to LKJ(eta + 0.5) on a 2x2 block.
                T eta_lkj = T(2.0);
                for (int k = 0; k < n_coefs_t; k++) {
                    T L_kk = L_flat_t[k * n_coefs_t + k];
                    log_post = log_post + (eta_lkj - T(1.0)
                               + T((n_coefs_t - k - 1) / 2.0)) * T(2.0) * safe_log(L_kk);
                }
            }

            // Get RE start index
            int re_start_t = (n_terms > 1 || layout.has_re_slopes)
                             ? layout.re_start_multi[t] : layout.re_start;
            int off = re_term_offsets[t];

            if (is_correlated && n_coefs_t > 1) {
                // NC correlated: z ~ N(0,I), re = diag(sigma) * L * z
                for (int g = 0; g < n_groups_t; g++) {
                    for (int c = 0; c < n_coefs_t; c++) {
                        T z_gc = params[re_start_t + g * n_coefs_t + c];
                        log_post = log_post - T(0.5) * z_gc * z_gc;

                        // re[c] = sigma[c] * (L[c,:] . z[g])
                        T Lz_c = T(0.0);
                        for (int k = 0; k <= c; k++) {
                            Lz_c = Lz_c + L_flat_t[c * n_coefs_t + k]
                                   * params[re_start_t + g * n_coefs_t + k];
                        }
                        re_vals[off + g * n_coefs_t + c] = sigmas_t[c] * Lz_c;
                    }
                }
            } else if (data.re_parameterization == 1) {
                // NC uncorrelated: z ~ N(0,I), re = sigma * z
                for (int g = 0; g < n_groups_t; g++) {
                    for (int c = 0; c < n_coefs_t; c++) {
                        T z_gc = params[re_start_t + g * n_coefs_t + c];
                        log_post = log_post - T(0.5) * z_gc * z_gc;
                        re_vals[off + g * n_coefs_t + c] = sigmas_t[c] * z_gc;
                    }
                }
            } else {
                // Centered: re ~ N(0, sigma^2)
                for (int g = 0; g < n_groups_t; g++) {
                    for (int c = 0; c < n_coefs_t; c++) {
                        T re_val = params[re_start_t + g * n_coefs_t + c];
                        T tau_c = T(1.0) / (sigmas_t[c] * sigmas_t[c] + T(1e-10));
                        log_post = log_post - T(0.5) * tau_c * re_val * re_val;
                        log_post = log_post + T(0.5) * safe_log(tau_c);
                        re_vals[off + g * n_coefs_t + c] = re_val;
                    }
                }
            }
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_RE_H
