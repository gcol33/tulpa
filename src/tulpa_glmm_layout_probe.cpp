// tulpa_glmm_layout_probe.cpp
// ----------------------------------------------------------------------------
// Report the parameter layout `cpp_tulpa_sample_glmm` will lay out for a given
// model, without sampling it.
//
// This exists so a warm start can be assembled in R by INDEX rather than by
// reconstructing the sampler's column-naming convention. The sampler's
// parameter vector is `compute_param_layout(data)`, and its column labels are
// `sampler_param_names()` derived from that same layout; a warm-start builder
// that instead re-derived where `log_sigma_re` for term 2 lands would be a
// second copy of the layout rule, free to drift from the first. Asking the
// engine is the single source of truth.
//
// Every index is returned 1-based, and an absent block is reported as NULL
// rather than as a sentinel, so an R caller cannot silently index on -1.
// ----------------------------------------------------------------------------

#include <Rcpp.h>
#include <string>
#include <vector>

#include "laplace_spec_fit.h"      // as_offset_vec
#include "sampler_model_data.h"    // build_sampler_model_inputs / sampler_param_names

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

// -1 (absent) -> R NULL; otherwise the 1-based index.
inline SEXP idx1(int i) {
    if (i < 0) return R_NilValue;
    return Rcpp::wrap(i + 1);
}

// Bounds are stored half-open [start, end) 0-based; report the 1-based inclusive
// span an R caller would use, or NULL when the block is absent or empty.
inline SEXP span1(int start, int end) {
    if (start < 0 || end <= start) return R_NilValue;
    return Rcpp::IntegerVector::create(start + 1, end);
}

inline Rcpp::IntegerVector idx1_vec(const std::vector<int>& v) {
    Rcpp::IntegerVector out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = (v[i] < 0) ? NA_INTEGER : (v[i] + 1);
    }
    return out;
}

}  // namespace


// [[Rcpp::export]]
Rcpp::List cpp_tulpa_glmm_layout(
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    double sigma_beta = 10.0,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    Rcpp::Nullable<Rcpp::CharacterVector> fixed_names = R_NilValue,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi_spec = R_NilValue
) {
    const int N = y.size();

    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, sigma_beta, offset, sigma_re_scale,
        re_spec, spatial_spec, temporal_spec, svc_spec, tvc_spec, zi_spec);

    const ParamLayout& L = in.layout;

    // Per-term random-effect blocks. Reported per term rather than flattened so
    // the caller places a term's deviations without recomputing group * coef
    // strides -- the layout already knows them.
    Rcpp::List re_terms;
    if (L.has_re) {
        const int n_terms = (int)L.re_start_multi.size();
        for (int t = 0; t < n_terms; t++) {
            const int nc = (t < (int)L.re_n_coefs_multi.size())
                ? L.re_n_coefs_multi[t] : 1;
            const int span = L.re_end_multi[t] - L.re_start_multi[t];
            const bool cor = (t < (int)L.re_correlated_multi.size())
                ? (bool)L.re_correlated_multi[t] : false;

            // log_sigma_re is one index per coefficient when the term carries
            // slopes, and a single index otherwise.
            SEXP log_sigma;
            if (L.has_re_slopes && t < (int)L.log_sigma_re_slopes.size()) {
                log_sigma = idx1_vec(L.log_sigma_re_slopes[t]);
            } else if (t < (int)L.log_sigma_re_multi.size()) {
                log_sigma = idx1(L.log_sigma_re_multi[t]);
            } else {
                log_sigma = R_NilValue;
            }

            re_terms.push_back(Rcpp::List::create(
                Rcpp::Named("re") = span1(L.re_start_multi[t], L.re_end_multi[t]),
                Rcpp::Named("n_coefs") = nc,
                Rcpp::Named("n_groups") = (nc > 0) ? (span / nc) : span,
                Rcpp::Named("correlated") = cor,
                Rcpp::Named("log_sigma_re") = log_sigma,
                Rcpp::Named("chol") = (t < (int)L.chol_re_start_multi.size())
                    ? span1(L.chol_re_start_multi[t], L.chol_re_end_multi[t])
                    : R_NilValue));
        }
    }

    // Fixed-effect blocks, one per process (the ZI predictor is a second
    // process on the Laplace path but a separate beta_zi block here).
    Rcpp::List beta_blocks;
    for (size_t kp = 0; kp < L.process_beta_start.size(); kp++) {
        beta_blocks.push_back(span1(
            L.process_beta_start[kp],
            L.process_beta_start[kp] + L.process_beta_count[kp]));
    }

    // Structure a warm-start builder cannot supply from a plain Laplace/EB fit:
    // each carries hyperparameters that fit never estimated. Reported so the R
    // layer can refuse precisely rather than guess.
    Rcpp::CharacterVector unsupported;
    if (L.has_spatial)  unsupported.push_back("spatial");
    if (L.has_temporal) unsupported.push_back("temporal");
    if (L.is_gp)        unsupported.push_back("gp");
    if (L.is_hsgp)      unsupported.push_back("hsgp");

    return Rcpp::List::create(
        Rcpp::Named("total_params") = L.total_params,
        Rcpp::Named("param_names") =
            tulpa::sampler_param_names(in.data, in.layout, fixed_names),
        Rcpp::Named("beta") = beta_blocks,
        Rcpp::Named("beta_zi") = L.has_zi
            ? span1(L.beta_zi_start, L.beta_zi_end) : R_NilValue,
        Rcpp::Named("re_terms") = re_terms,
        Rcpp::Named("extra") = (L.n_extra_params > 0)
            ? span1(L.extra_offset, L.extra_offset + L.n_extra_params)
            : R_NilValue,
        Rcpp::Named("unsupported") = unsupported);
}
