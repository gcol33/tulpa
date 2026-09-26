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
// 0. Where a random-effect term sits in the parameter vector
// ============================================================================

// The shape of term t and the parameter positions it reads: its group and
// coefficient counts, the log-SD of each coefficient, the raw Cholesky block of
// a correlated term (-1 otherwise) and its latent block. The one reading of the
// layout that compute_re_prior() and the ESS backend's random-effect moves
// (ess_sampler.h) share, so the two cannot come to disagree about which
// coordinate is which.
struct ReTermView {
    int n_groups = 0;
    int n_coefs = 1;
    int chol_start = -1;
    int re_start = -1;
    std::vector<int> log_sigma_idx;
};

inline int re_n_terms(const ModelData& data) {
    return (data.n_re_terms > 0) ? data.n_re_terms : 1;
}

inline ReTermView re_term_view(const ModelData& data, const ParamLayout& layout,
                               int t) {
    const int n_terms = re_n_terms(data);
    ReTermView v;
    v.n_groups = (n_terms > 1 || data.n_re_terms > 0)
                 ? data.re_n_groups_multi[t] : data.n_re_groups;
    v.n_coefs = layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
    const bool is_correlated = layout.has_re_slopes &&
                               !layout.re_correlated_multi.empty() &&
                               layout.re_correlated_multi[t];
    v.log_sigma_idx.resize(v.n_coefs);
    for (int c = 0; c < v.n_coefs; c++) {
        if (layout.has_re_slopes) {
            v.log_sigma_idx[c] = layout.log_sigma_re_slopes[t][c];
        } else if (n_terms > 1) {
            v.log_sigma_idx[c] = layout.log_sigma_re_multi[t];
        } else {
            v.log_sigma_idx[c] = layout.log_sigma_re_idx;
        }
    }
    v.chol_start = (is_correlated && v.n_coefs > 1)
                   ? layout.chol_re_start_multi[t] : -1;
    v.re_start = (n_terms > 1 || layout.has_re_slopes)
                 ? layout.re_start_multi[t] : layout.re_start;
    return v;
}

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
        const int n_terms = re_n_terms(data);

        // Per-term layout, resolved once: the offset loop below needs the
        // dimensions to size re_vals, and the density loop needs all of it.
        std::vector<ReTermView> terms(n_terms);
        int total_re_vals = 0;
        re_term_offsets.resize(n_terms);
        for (int t = 0; t < n_terms; t++) {
            terms[t] = re_term_view(data, layout, t);
            re_term_offsets[t] = total_re_vals;
            total_re_vals += terms[t].n_groups * terms[t].n_coefs;
        }
        re_vals.resize(total_re_vals, T(0.0));

        for (int t = 0; t < n_terms; t++) {
            const ReTermView& v = terms[t];
            re_term_log_prior_add(params.data(), v.log_sigma_idx.data(),
                                  v.n_coefs, v.chol_start, v.re_start,
                                  v.n_groups, data.re_parameterization == 1,
                                  data.sigma_re_scale, log_post,
                                  re_vals.data() + re_term_offsets[t]);
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_RE_H
