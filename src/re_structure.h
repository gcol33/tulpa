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
#include <string>
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
void populate_re_structure(
    ModelData& data, int N,
    const Rcpp::List& re_idx_list,
    const std::vector<int>& re_ngroups,
    const std::vector<int>& ncoefs,
    const Rcpp::Nullable<Rcpp::List>& re_Z_list,
    const std::vector<bool>& correlated
);

} // namespace tulpa

#endif // TULPA_RE_STRUCTURE_H
