// ess_api.h
// Cross-package Elliptical Slice Sampling API.
//
// Wraps tulpa_ess::run_ess_sampler. Set joint_sigma_re = 1 in ESSShimConfig
// to enable a joint (log_sigma_re, re) Metropolis move on top of ESS — the
// alternating updates mix poorly for Poisson + RE because the centered
// parameterization makes log_sigma_re and re strongly anti-correlated.

#ifndef TULPA_ESS_API_H
#define TULPA_ESS_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "nuts_api.h"  // check_abi_version

namespace tulpa {

struct ESSShimConfig {
    int n_iter;
    int n_warmup;
    int n_thin;
    int verbose;
    int print_every;
    unsigned int seed;
    int use_cholesky;        // 0/1
    int adapt_during_warmup; // 0/1
    int adapt_interval;
    int joint_sigma_re;      // 0/1 — joint-sample (log_sigma_re, z) for Poisson + RE
};

inline ESSShimConfig default_ess_config() {
    ESSShimConfig c;
    c.n_iter = 2000;
    c.n_warmup = 1000;
    c.n_thin = 1;
    c.verbose = 1;
    c.print_every = 100;
    c.seed = 12345;
    c.use_cholesky = 1;
    c.adapt_during_warmup = 1;
    c.adapt_interval = 100;
    c.joint_sigma_re = 0;
    return c;
}

// Result. samples is [n_save * n_params] row-major.
struct ESSShimResult {
    int n_save;
    int n_params;
    int n_slice_evals;
    double avg_slice_evals;
    int success;          // 0/1
    double* samples;
    double* log_lik;
    char error_msg[256];

    void free_buffers() {
        if (samples) { delete[] samples; samples = nullptr; }
        if (log_lik) { delete[] log_lik; log_lik = nullptr; }
    }
};

typedef void (*ESSFn)(
    const double* init_params,
    int n_params,
    const ModelData* data,
    const ParamLayout* layout,
    const ESSShimConfig* config,
    ESSShimResult* result_out
);

inline ESSFn get_ess_fn() {
    static ESSFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (ESSFn)R_GetCCallable("tulpa", "tulpa_run_ess_sampler");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_ESS_API_H
