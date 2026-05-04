// laplace_spec_api.h
// Cross-package LikelihoodSpec-driven Laplace API for model packages.
//
// The family-enum Laplace shims in laplace_api.h dispatch to the six
// hard-coded families wired in src/laplace_family_link.h. This header
// exposes a complementary path that reads the per-observation log-lik
// and IRLS weights from data->likelihood_spec instead, so a model
// package can route arbitrary families (beta, tweedie, ordered_beta,
// truncated/hurdle, ratio, ...) through Laplace without touching tulpa.
//
// Contract:
//   - data->n_processes >= 1 (multi-process supported; ratio likelihoods
//     use n_processes == 2, integrated occupancy uses 1 + n_sources, etc.)
//   - layout->process_beta_start[k] / .process_beta_count[k] populated
//     for every k in [0, n_processes).
//   - Per-process additive offsets are picked up automatically from
//     data->processes[k].offset when non-empty (length must equal N).
//   - At most one iid RE term (layout->has_re + .re_start / .re_end).
//     The RE shares into process k iff data->sharing.re[k] is true.
//   - data->likelihood_spec points to a tulpa::LikelihoodSpec whose
//     ll_double and eta_weights_fn are non-null. eta_weights_fn must
//     fill grad_eta[k] = d log_lik_i / d eta_i_k and neg_hess_eta as a
//     row-major n_processes x n_processes block.
//
// Random-slope / spatial / temporal variants are follow-on work; this
// entry will reject them with a clear error so callers get a deterministic
// signal instead of silent miscomputation.

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
//   re_group        : per-obs 1-based RE group index. Length = data->N if
//                     layout->has_re, otherwise pass nullptr / len 0.
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
