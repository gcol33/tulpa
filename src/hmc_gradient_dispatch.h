// hmc_gradient_dispatch.h
// Gradient-mode dispatch policy for the HMC backend. This is the whole of it:
// the dispatcher is layout-agnostic, so no `layout.has_*` flag selects a
// kernel and there are no per-structure predicates to consult.
//
// The only supported entry path is the generic LikelihoodSpec interface
// (`n_processes > 0` plus a non-null `data.likelihood_spec`); downstream
// packages route through `tulpa::LikelihoodSpec`.
//
// resolve_gradient_fn returns one of three gradient sources, in this order:
//
//   1. spec->gradient_fn        the model package's hand-coded full gradient,
//                               unless the caller asked for NUMERICAL
//   2. compute_gradient_generic_arena     arena reverse-mode AD, when the
//                               package ships ll_arena (and an arena variant of
//                               extra_prior, if it sets one)
//   3. compute_gradient_generic_numerical central differences against
//                               compute_log_post_generic_spec_double
//
// resolve_prior_gradient_fn mirrors it for the prior-only ("fast") force of the
// multiple-time-stepping integrator, minus arm 1: a full gradient cannot be
// split into prior and likelihood parts.
//
// Adding a gradient source means adding an arm here. There is no separate
// predicate header and no specificity ordering to insert into.
//
// Included from hmc_gradient_dispatch.cpp inside namespace tulpa_hmc.

#ifndef TULPA_HMC_GRADIENT_DISPATCH_H
#define TULPA_HMC_GRADIENT_DISPATCH_H

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    (void)layout;  // no layout flag selects a gradient source
    if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
        Rcpp::stop("tulpa: ModelData has n_processes == 0. Downstream packages "
                   "must populate `n_processes > 0` and `data.likelihood_spec` "
                   "via the generic LikelihoodSpec interface.");
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    // Three distinct gradient sources exist: NUMERICAL (finite differences),
    // the arena reverse-mode AD path, and a model package's hand-coded gradient.
    // The AUTODIFF_FWD (forward) and AUTODIFF_TAPE (tape) modes are parsed for
    // API stability but ship no separate kernel, so they resolve here to the
    // same arena path as AUTODIFF_ARENA -- an explicit alias, not a silent
    // substitution (see nuts_api.h). Only NUMERICAL selects a different source.

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
