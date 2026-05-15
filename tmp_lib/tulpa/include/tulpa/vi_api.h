// vi_api.h
// Cross-package Variational Inference API (mean-field / low-rank / full-rank).
//
// Single shim: the internal dispatcher picks the variant from D and config.
// Set variant != AUTO in VIShimConfig to force a specific one.

#ifndef TULPA_VI_API_H
#define TULPA_VI_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "nuts_api.h"  // check_abi_version

namespace tulpa {

// VI variant enum — mirrors src/vi_types.h's tulpa::vi::VIVariant.
// Plain int storage so it's ABI-stable across DLLs.
enum VIVariantCode : int {
    VI_AUTO      = 0,
    VI_MEANFIELD = 1,
    VI_LOWRANK   = 2,
    VI_FULLRANK  = 3
};

// POD config that the shim translates into the internal VIConfig.
struct VIShimConfig {
    int variant;            // VIVariantCode
    int max_iter;
    int mc_samples;
    double tol_grad;
    double tol_rel_elbo;
    int patience;
    double adam_alpha;
    double adam_beta1;
    double adam_beta2;
    double adam_eps;
    int rank;               // -1 = auto
    int use_laplace_init;   // 0/1
    int fullrank_threshold;
    int lowrank_threshold;
    int verbose;            // 0/1
    int print_every;
    unsigned int seed;
};

// Result.
//   mu:         [D]
//   Sigma:      [D * D] row-major, always filled
//   L_factor:   [D * rank] for low-rank, [D * D] for full-rank, nullptr for meanfield
//   d_diag:     [D] for low-rank, nullptr otherwise
//   elbo_history: [iterations] (caller-owned)
struct VIShimResult {
    int variant_used;       // VIVariantCode
    int D;
    int rank_used;
    int iterations;
    int converged;
    double final_elbo;
    double psis_k;
    double* mu;
    double* Sigma;
    double* L_factor;
    double* d_diag;
    double* elbo_history;

    void free_buffers() {
        if (mu)           { delete[] mu;           mu = nullptr; }
        if (Sigma)        { delete[] Sigma;        Sigma = nullptr; }
        if (L_factor)     { delete[] L_factor;     L_factor = nullptr; }
        if (d_diag)       { delete[] d_diag;       d_diag = nullptr; }
        if (elbo_history) { delete[] elbo_history; elbo_history = nullptr; }
    }
};

// init_mu may be nullptr, or a [D] vector to seed the variational mean.
typedef void (*VIFitFn)(
    const ModelData* data,
    const ParamLayout* layout,
    int D,
    const VIShimConfig* config,
    const double* init_mu,
    int n_init_mu,
    VIShimResult* result_out
);

inline VIFitFn get_vi_fit_fn() {
    static VIFitFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (VIFitFn)R_GetCCallable("tulpa", "tulpa_fit_vi");
    }
    return fn;
}

// Fill a VIShimConfig with the same defaults the internal struct uses.
inline VIShimConfig default_vi_config() {
    VIShimConfig c;
    c.variant = VI_AUTO;
    c.max_iter = 10000;
    c.mc_samples = 10;
    c.tol_grad = 1e-4;
    c.tol_rel_elbo = 0.01;
    c.patience = 50;
    c.adam_alpha = 0.01;
    c.adam_beta1 = 0.9;
    c.adam_beta2 = 0.999;
    c.adam_eps = 1e-8;
    c.rank = -1;
    c.use_laplace_init = 1;
    c.fullrank_threshold = 200;
    c.lowrank_threshold = 2000;
    c.verbose = 1;
    c.print_every = 100;
    c.seed = 0;
    return c;
}

} // namespace tulpa

#endif // TULPA_VI_API_H
