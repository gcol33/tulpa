// laplace_spec_api.h
// Cross-package LikelihoodSpec-driven Laplace API for model packages.
//
// The family-enum Laplace shims in laplace_api.h dispatch to the six
// hard-coded families wired in src/laplace_family_link.h. This header
// exposes a complementary path that reads the per-observation log-lik
// and IRLS weights from data->likelihood_spec instead, so a model
// package can route arbitrary families (beta, tweedie, ordered_beta,
// truncated/hurdle, ...) through Laplace without touching tulpa.
//
// Contract (first cut):
//   - data->n_processes == 1
//   - layout->process_beta_start[0] / .process_beta_count[0] populated
//   - At most one iid RE term (layout->has_re + .re_start / .re_end)
//   - data->likelihood_spec points to a tulpa::LikelihoodSpec whose
//     ll_double and eta_weights_fn are non-null
//
// Multi-process / random-slope / spatial / temporal variants are
// follow-on work; this entry will reject them with a clear error so
// callers get a deterministic signal instead of silent miscomputation.

#ifndef TULPA_LAPLACE_SPEC_API_H
#define TULPA_LAPLACE_SPEC_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "laplace_api.h"   // for LaplaceShimResult
#include "nuts_api.h"      // check_abi_version

namespace tulpa {

// Function signature for tulpa_laplace_spec_dense.
//   data, layout    : same shape as the NUTS shim (n_processes == 1).
//   params_inout    : full parameter vector. Hyperparameter slots
//                     (sigma_re, dispersion, ...) supply their fixed values
//                     on entry; latent slots may carry a warm start. On exit
//                     the latent slots are overwritten with the mode;
//                     hyperparameter slots are untouched. Length = n_params.
//   re_group        : per-obs 1-based RE group index. Length = data->N if
//                     layout->has_re, otherwise pass nullptr / len 0.
//   max_iter, tol, n_threads : Newton controls.
//   result_out      : caller-allocated. Filled with mode (== params_inout
//                     latent slice), log_det_Q, log_marginal, n_iter,
//                     converged. result_out->n_x = layout latent count.
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
