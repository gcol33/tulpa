// re_structure.h
// Single source of truth for marshalling the multi-term random-effect layout of
// a ModelData from the R-facing per-term inputs (group indices, group counts,
// optional Z designs, coefficient counts, correlated flags). Shared by the
// multi-term Laplace fit (cpp_laplace_fit_multi_re) and the ModelData-sampler
// builder (build_sampler_model_inputs), so the structural marshalling lives in
// one place. This fills only the *structural* fields + totals; the per-term
// hyperparameter values (the sigma packing the Laplace path conditions on, or
// the sampled parameters the kernels draw) are the caller's concern.

#ifndef TULPA_RE_STRUCTURE_H
#define TULPA_RE_STRUCTURE_H

#include "tulpa/model_data.h"
#include <Rcpp.h>
#include <vector>

namespace tulpa {

// Populate data's multi-term RE structural fields for K = re_ngroups.size()
// terms. re_idx_list[t] is the 1-based per-obs group index (length N) for term
// t; ncoefs[t] the coefficients per group (1 = intercept only); correlated[t]
// whether the term's Sigma is full (only meaningful when ncoefs[t] > 1).
// re_Z_list (optional) supplies each term's RE design: an all-ones first column
// marks the implicit group intercept (lme4 `(1 + x | g)`), so the slopes are
// columns 1.., while a non-ones first column is a no-intercept design
// (`(0 + x | g)`). A null / absent Z is intercept-only.
//
// Mirrors the multi-term marshalling cpp_laplace_fit_multi_re used inline; the
// only difference is that `correlated` is supplied by the caller (the Laplace
// path derives it from the packed-Sigma length, the sampler from the formula)
// rather than inferred here.
inline void populate_re_structure(
    ModelData& data, int N,
    const Rcpp::List& re_idx_list,
    const std::vector<int>& re_ngroups,
    const std::vector<int>& ncoefs,
    const Rcpp::Nullable<Rcpp::List>& re_Z_list,
    const std::vector<bool>& correlated
) {
    const int K = (int)re_ngroups.size();

    data.n_re_terms = K;
    data.re_n_groups_multi.assign(K, 0);
    data.re_n_coefs.assign(K, 1);
    data.re_n_slopes.assign(K, 0);
    data.re_has_intercept.assign(K, 1);
    data.re_correlated.assign(K, false);
    data.re_n_chol.assign(K, 0);
    data.re_offsets.assign(K, 0);
    data.re_slope_matrices.assign(K, std::vector<double>());
    data.re_group_multi_flat.assign((size_t)N * K, 0);

    const bool have_Z = re_Z_list.isNotNull();
    Rcpp::List zl = have_Z ? Rcpp::as<Rcpp::List>(re_Z_list) : Rcpp::List();

    int total_re_groups = 0, total_re_params = 0;
    int total_sigma_params = 0, total_chol_params = 0;
    bool any_slopes = false, any_correlated = false;

    for (int t = 0; t < K; t++) {
        const int n_g = re_ngroups[t];
        const int q   = ncoefs[t];
        Rcpp::IntegerVector gi = Rcpp::as<Rcpp::IntegerVector>(re_idx_list[t]);
        if ((int)gi.size() != N) {
            Rcpp::stop("re_idx_list[[%d]] has length %d but N = %d.",
                       t + 1, (int)gi.size(), N);
        }
        const bool corr = (q > 1) && correlated[t];

        // RE design -> intercept flag + slope matrix. An all-ones first column
        // marks the implicit intercept (lme4 (1 + x | g)); a non-ones first
        // column is a no-intercept design ((0 + x | g)).
        bool has_int = true;
        int  n_slopes = 0;
        if (have_Z && !Rf_isNull(zl[t])) {
            Rcpp::NumericMatrix Z = Rcpp::as<Rcpp::NumericMatrix>(zl[t]);
            if (Z.nrow() != N) {
                Rcpp::stop("re_Z_list[[%d]] has %d rows but N = %d.",
                           t + 1, Z.nrow(), N);
            }
            has_int = true;
            for (int i = 0; i < N; i++) { if (Z(i, 0) != 1.0) { has_int = false; break; } }
            n_slopes = has_int ? (Z.ncol() - 1) : Z.ncol();
            if (n_slopes > 0) {
                data.re_slope_matrices[t].assign((size_t)N * n_slopes, 0.0);
                for (int i = 0; i < N; i++)
                    for (int s = 0; s < n_slopes; s++)
                        data.re_slope_matrices[t][(size_t)i * n_slopes + s]
                            = Z(i, has_int ? s + 1 : s);
            }
        }

        data.re_n_groups_multi[t] = n_g;
        data.re_n_coefs[t]        = q;
        data.re_n_slopes[t]       = n_slopes;
        data.re_has_intercept[t]  = has_int ? 1 : 0;
        data.re_correlated[t]     = corr;
        data.re_n_chol[t]         = corr ? q * (q - 1) / 2 : 0;
        data.re_offsets[t]        = total_re_groups;
        for (int i = 0; i < N; i++) data.re_group_multi_flat[(size_t)i * K + t] = gi[i];

        if (n_slopes > 0) any_slopes = true;
        if (corr)         any_correlated = true;
        total_re_groups    += n_g;
        total_re_params    += n_g * q;
        total_sigma_params += q;
        total_chol_params  += data.re_n_chol[t];
    }

    data.total_re_groups          = total_re_groups;
    data.total_re_params          = total_re_params;
    data.total_sigma_params       = total_sigma_params;
    data.total_chol_params        = total_chol_params;
    data.has_re_slopes            = any_slopes;
    data.has_re_correlated_slopes = any_correlated;
}

} // namespace tulpa

#endif // TULPA_RE_STRUCTURE_H
