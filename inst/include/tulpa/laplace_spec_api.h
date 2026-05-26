// laplace_spec_api.h
// Cross-package LikelihoodSpec-driven Laplace API for model packages.
//
// This header exposes the LikelihoodSpec-driven Laplace path: it reads the
// per-observation log-lik and IRLS weights from data->likelihood_spec, so a
// model package can route arbitrary families (binomial, poisson, beta,
// tweedie, ordered_beta, truncated/hurdle, ratio, ...) through Laplace
// without touching tulpa's internals.
//
// Contract:
//   - data->n_processes >= 1 (multi-process supported; ratio likelihoods
//     use n_processes == 2, integrated occupancy uses 1 + n_sources, etc.)
//   - layout->process_beta_start[k] / .process_beta_count[k] populated
//     for every k in [0, n_processes).
//   - Per-process additive offsets are picked up automatically from
//     data->processes[k].offset when non-empty (length must equal N).
//   - Random effects: K = data->n_re_terms terms, q_t = re_n_coefs[t]
//     coefficients per group per term (q_t == 1 for `(1|g)`, q_t > 1
//     for slopes). The latent contribution at obs i is
//         sum_t  z_{t,i}^T b_{t, g_t(i)}
//     where, for an intercept-carrying block, z_{t,i,0} = 1 and slopes
//     z_{t,i,c} (c = 1..q_t-1) are read from
//     data->re_slope_matrices[t][i*n_slopes + (c-1)]. A slope-only block
//     (`(0 + x | g)`, re_has_intercept[t] == 0) has no z = 1 column: every
//     coef c = 0..q_t-1 is a slope read at column c. Per-term
//     prior covariance Σ_t is uncorrelated (`(x||g)` -> diagonal) or
//     correlated (`(x|g)` -> tanh-Cholesky-parameterized in
//     params[chol_re_start_multi[t]..chol_re_end_multi[t])).
//     The legacy single-term layout (layout->re_start / re_end /
//     log_sigma_re_idx) is honoured as the K == 1, q_0 == 1 case.
//     Per-process sharing is uniform across terms via data->sharing.re.
//   - data->likelihood_spec points to a tulpa::LikelihoodSpec whose
//     ll_double and eta_weights_fn are non-null. eta_weights_fn must
//     fill grad_eta[k] = d log_lik_i / d eta_i_k and neg_hess_eta as a
//     row-major n_processes x n_processes block.
//
// Spatial / temporal latent fields are still follow-on work; this entry
// integrates only over fixed effects + RE blocks and treats every other
// hyperparameter slot as pinned at its input value.

#ifndef TULPA_LAPLACE_SPEC_API_H
#define TULPA_LAPLACE_SPEC_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "laplace_api.h"   // for LaplaceShimResult
#include "nuts_api.h"      // check_abi_version

namespace tulpa {

// Function signature for tulpa_laplace_spec_dense.
//   data, layout    : same shape as the NUTS shim (n_processes >= 1).
//   params_inout    : full parameter vector. Hyperparameter slots
//                     (sigma_re, dispersion, ...) supply their fixed values
//                     on entry; latent slots may carry a warm start. On exit
//                     the latent slots are overwritten with the mode;
//                     hyperparameter slots are untouched. Length = n_params.
//   re_group        : per-obs 1-based RE group index for the LEGACY
//                     single-term path (length data->N when layout->has_re
//                     and the caller has not populated
//                     data->re_group_multi_flat). Multi-term callers must
//                     populate data->re_group_multi_flat themselves
//                     (length data->N * data->n_re_terms) and may pass
//                     nullptr / 0 here.
//   max_iter, tol, n_threads : Newton controls.
//   result_out      : caller-allocated. Filled with mode (== params_inout
//                     latent slice in [beta_0..beta_{np-1}, re] order),
//                     log_det_Q, log_marginal, n_iter, converged.
//                     result_out->n_x = sum(beta_count) + n_re_groups.
typedef void (*LaplaceSpecDenseFn)(
    const ModelData* data,
    const ParamLayout* layout,
    double* params_inout,
    int n_params,
    const int* re_group,
    int n_re_group,
    int max_iter,
    double tol,
    int n_threads,
    LaplaceShimResult* result_out
);

inline LaplaceSpecDenseFn get_laplace_spec_dense_fn() {
    static LaplaceSpecDenseFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceSpecDenseFn)R_GetCCallable(
            "tulpa", "tulpa_laplace_spec_dense");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_SPEC_API_H
