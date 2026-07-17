// hmc_gradient_dispatch.h
// Gradient-mode dispatch policy for the HMC backend.
//
// Phase D simplification: the only supported entry path
// is the generic LikelihoodSpec interface (`n_processes > 0` plus a
// non-null `data.likelihood_spec`). Legacy ratio (n_processes == 0)
// dispatch was removed along with the entry points that produced that
// ModelData shape; downstream packages route through `tulpa::LikelihoodSpec`.
//
// Included from hmc_gradient_dispatch.cpp inside namespace tulpa_hmc.

#ifndef TULPA_HMC_GRADIENT_DISPATCH_H
#define TULPA_HMC_GRADIENT_DISPATCH_H

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    (void)layout;  // dispatcher is layout-agnostic after Phase D
    if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
        Rcpp::stop("tulpa: ModelData has n_processes == 0 — the legacy ratio "
                   "dispatch path was removed in Phase D of the tulpaRatio "
                   "migration (gcol33/tulpa#15). Downstream packages must "
                   "populate `n_processes > 0` and `data.likelihood_spec` "
                   "via the generic LikelihoodSpec interface.");
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    // Hand-coded full gradient hook (FullGradFn): the model package ships a
    // tuned gradient that subsumes log-prior + log-likelihood. When set, it
    // wins regardless of the requested mode unless the user explicitly asks
    // for NUMERICAL (used for runtime gradient verification). The signature
    // matches GradientFn exactly, so it plugs straight into the dispatcher
    // with no wrapper.
    if (spec->gradient_fn != nullptr && mode != GradientMode::NUMERICAL) {
        return reinterpret_cast<GradientFn>(spec->gradient_fn);
    }

    // Arena AD path: requires the model package to ship an arena-AD log-lik
    // and, if extra_prior is set, also an arena-AD prior variant. When
    // extra_prior exists without an arena variant we cannot fold the prior
    // gradient into the backward pass; fall back to central differences so
    // the prior gradient is correct.
    if (spec->ll_arena != nullptr &&
        (spec->extra_prior == nullptr || spec->extra_prior_arena != nullptr)) {
        return &compute_gradient_generic_arena;
    }

    return &compute_gradient_generic_numerical;
}

// Prior-only counterpart used by the multiple-time-stepping leaf. The hand-coded
// full-gradient hook (spec->gradient_fn) subsumes prior + likelihood and cannot
// be split, so it is never chosen here: prefer the arena-AD prior path (same
// availability rule as the full arena path), else central differences on the
// prior-only log-post. Correctness matches resolve_gradient_fn -- both fold in
// the optional model-package prior -- so grad_full - grad_prior is exactly the
// observation-likelihood gradient.
GradientFn resolve_prior_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    (void)layout;
    if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
        Rcpp::stop("tulpa: ModelData has n_processes == 0 — the generic "
                   "LikelihoodSpec interface is required for the prior-only "
                   "gradient path (multiple-time-stepping integrator).");
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    if (mode != GradientMode::NUMERICAL &&
        spec->ll_arena != nullptr &&
        (spec->extra_prior == nullptr || spec->extra_prior_arena != nullptr)) {
        return &compute_gradient_prior_arena;
    }

    return &compute_gradient_prior_numerical;
}

#endif // TULPA_HMC_GRADIENT_DISPATCH_H
