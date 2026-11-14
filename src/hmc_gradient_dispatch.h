// hmc_gradient_dispatch.h
// Gradient-mode dispatch for model likelihoods and engine-owned field targets.
//
// The only supported entry path is the generic LikelihoodSpec interface
// (`n_processes > 0` plus a non-null `data.likelihood_spec`); downstream
// packages route through `tulpa::LikelihoodSpec`.
//
// Numerical mode uses central differences. Other modes preserve transformed
// field coordinates and collapsed targets, then select a resident device
// likelihood, a consumer full gradient, arena reverse mode, or central
// differences in that order. Device and arena routes require an arena prior
// whenever the consumer supplies an extra prior.
//
// resolve_prior_gradient_fn supplies the prior-only force for the
// multiple-time-stepping integrator. Full gradients include the likelihood,
// so the prior force uses its own arena or numerical evaluation.
//
// Adding a gradient source means adding an arm here. There is no separate
// predicate header and no specificity ordering to insert into.
//
// NOT A HEADER. This is a textual fragment: it DEFINES two functions with
// external linkage and is included, exactly once, from
// hmc_gradient_dispatch.cpp inside `namespace tulpa_hmc`. Including it from a
// second translation unit gives two definitions of each and fails at link
// time, so there is no include guard to suggest otherwise -- a guard is
// per-translation-unit and would not prevent that anyway. Declarations for
// callers live in hmc_sampler_decls.h.

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    if (mode == GradientMode::NUMERICAL)
        return &compute_gradient_generic_numerical;
    if (field_coordinates && !field_coordinates->empty())
        return &compute_full_gradient_field_coordinates;
    if (has_collapsed_spatial(layout)) return &compute_gradient_collapsed_spatial;
    if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
        Rcpp::stop("tulpa: ModelData has n_processes == 0. Downstream packages "
                   "must populate `n_processes > 0` and `data.likelihood_spec` "
                   "via the generic LikelihoodSpec interface.");
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    tulpa::DeviceLikelihood kernel;
    tulpa::device::Status device_status;
    auto policy=tulpa::device::current_policy();
    if ((spec->extra_prior==nullptr || spec->extra_prior_arena!=nullptr) &&
        tulpa::device::select_likelihood(data,layout,kernel,*policy,device_status))
        return &compute_gradient_generic_device;
    if (!device_status) {
        policy->fail(device_status);
        throw std::runtime_error(device_status.message);
    }
    policy->record("likelihood",false);

    // Hand-coded full gradient hook (FullGradFn): the model package ships a
    // tuned gradient that subsumes log-prior + log-likelihood. It applies
    // when device likelihood evaluation is inactive. The signature
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
    if (mode == GradientMode::NUMERICAL)
        return &compute_gradient_prior_numerical;
    if (field_coordinates && !field_coordinates->empty())
        return &compute_prior_gradient_field_coordinates;
    if (has_collapsed_spatial(layout)) return &compute_prior_gradient_collapsed_spatial;
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
