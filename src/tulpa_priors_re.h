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
#include "tulpa/re_term_prior.h"

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

            // Log-SD position of each coefficient; the raw Cholesky block of a
            // correlated term; the term's latent block.
            std::vector<int> log_sigma_idx(n_coefs_t);
            for (int c = 0; c < n_coefs_t; c++) {
                if (layout.has_re_slopes) {
                    log_sigma_idx[c] = layout.log_sigma_re_slopes[t][c];
                } else if (n_terms > 1) {
                    log_sigma_idx[c] = layout.log_sigma_re_multi[t];
                } else {
                    log_sigma_idx[c] = layout.log_sigma_re_idx;
                }
            }
            int chol_start = (is_correlated && n_coefs_t > 1)
                             ? layout.chol_re_start_multi[t] : -1;
            int re_start_t = (n_terms > 1 || layout.has_re_slopes)
                             ? layout.re_start_multi[t] : layout.re_start;

            re_term_log_prior_add(params.data(), log_sigma_idx.data(), n_coefs_t,
                                  chol_start, re_start_t, n_groups_t,
                                  data.re_parameterization == 1,
                                  data.sigma_re_scale, log_post,
                                  re_vals.data() + re_term_offsets[t]);
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_RE_H
