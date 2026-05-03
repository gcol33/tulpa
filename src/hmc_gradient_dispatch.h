// hmc_gradient_dispatch.h
// Gradient-mode dispatch policy for the HMC backend.
//
// Included from hmc_gradient_dispatch.cpp inside namespace tulpa_hmc, which
// is the single reviewable home for the dispatch policy. Per-kernel support
// contracts live in hmc_gradient_dispatch_predicates.h.

#ifndef TULPA_HMC_GRADIENT_DISPATCH_H
#define TULPA_HMC_GRADIENT_DISPATCH_H

#include "hmc_gradient_dispatch_predicates.h"

// Dispatch order is part of the HMC performance contract: honor explicit user
// modes first, use the fastest safe hand-coded path for AUTO/H, and fall back
// to arena autodiff or numerical gradients for model combinations whose
// hand-coded implementations do not cover the full posterior.

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    // Generic multi-process models: route through generic gradient
    if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
        const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
        // Use arena AD if model provides it. A double-only extra_prior cannot
        // be differentiated by arena AD, so keep that case on the numerical
        // path until model-specific extra priors get templated callbacks.
        if (spec->ll_arena != nullptr && spec->extra_prior == nullptr) {
            return &compute_gradient_generic_arena;
        }
        return &compute_gradient_generic_numerical;
    }

    // Collapsed ICAR/BYM2: spatial inner state (phi*, theta*) is marginalized
    // and not in the param vector. compute_log_post_impl<T> can't autodiff
    // through the inner Newton solve, so AUTODIFF modes would either return
    // wrong gradients or crash on params[spatial_start=-1]. Redirect
    // explicit AUTODIFF requests to the numerical reference; the H-mode
    // analytical handler below remains the canonical fast path.
    const bool is_collapsed = layout.is_icar_collapsed || layout.is_bym2_collapsed;
    if (is_collapsed && (mode == GradientMode::AUTODIFF_TAPE ||
                          mode == GradientMode::AUTODIFF_ARENA ||
                          mode == GradientMode::AUTODIFF_FWD)) {
        return &compute_gradient_numerical;
    }

    // Explicit mode overrides
    if (mode == GradientMode::NUMERICAL)
        return &compute_gradient_numerical;
    if (mode == GradientMode::AUTODIFF_TAPE)
        return &compute_gradient_autodiff;
    if (mode == GradientMode::AUTODIFF_ARENA)
        return &compute_gradient_arena;
    if (mode == GradientMode::AUTODIFF_FWD)
        return &compute_gradient_forward;

    // AUTO or HANDCODED: use fastest available (H > A_r > A > N)
    if (can_use_analytical_gradient(data, layout)) {
        return &compute_gradient_analytical;
    }

    // ZI/OI supported in analytical (all non-exotic configs) and composite (exotic combos).
    // Specialized H-mode functions do not handle ZI; skip them for ZI/OI models.
    // ZOIB has a pre-existing gradient bug in the H-mode ZOIB residual code.
    if (data.zi_type == tulpa_zi::ZIType::ZOIB)
        return &compute_gradient_arena;
    if (layout.has_zi || layout.has_oi)
        return &compute_gradient_composite;

    // Crossed intercept-only RE is not handled by composite for exotic configs.
    if (has_exotic_feature(layout) && !layout.has_re_slopes && data.n_re_terms > 1)
        return &compute_gradient_arena;

    // TVC + latent is not covered by either specialized function.
    if (layout.has_tvc && layout.has_latent)
        return &compute_gradient_arena;

    // ST interaction + AR1 temporal type: H and composite use IID fallback, while
    // compute_log_post/log_post_impl return 0 prior for that case.
    if (layout.has_spatiotemporal &&
        data.spatiotemporal_data.temporal_type == TemporalType::AR1)
        return &compute_gradient_arena;

    // Specialized H-mode kernels (each predicate documents its support contract;
    // see hmc_gradient_dispatch_predicates.h).
    if (can_use_hsgp_handcoded(data, layout))           return &compute_gradient_hsgp;
    if (can_use_gp_collapsed(data, layout))             return &compute_gradient_gp_collapsed;
    if (can_use_icar_collapsed(layout))                 return &compute_gradient_icar_collapsed;
    if (can_use_gp_handcoded(data, layout))             return &compute_gradient_gp_handcoded;
    if (can_use_msgp_hsgp(data, layout))                return &compute_gradient_msgp_hsgp;
    if (can_use_msgp_plus_temporal(data, layout))       return &compute_gradient_msgp_plus_temporal_handcoded;
    if (can_use_msgp_handcoded(data, layout))           return &compute_gradient_msgp_handcoded;
    if (can_use_gp_plus_temporal(layout))               return &compute_gradient_gp_plus_temporal_handcoded;
    if (can_use_svc_hsgp_handcoded(data, layout))       return &compute_gradient_svc_hsgp_handcoded;
    if (can_use_svc_handcoded(data, layout))            return &compute_gradient_svc_handcoded;
    if (can_use_tvc_handcoded(data, layout))            return &compute_gradient_tvc_handcoded;
    if (can_use_spatiotemporal_handcoded(data, layout)) return &compute_gradient_spatiotemporal_handcoded;
    if (can_use_temporal_gp_handcoded(data, layout))    return &compute_gradient_temporal_gp_handcoded;
    if (can_use_ms_temporal_handcoded(layout))          return &compute_gradient_ms_temporal_handcoded;
    if (can_use_latent_handcoded(data, layout))         return &compute_gradient_latent_handcoded;

    // Composite H-mode: catch-all for exotic multi-feature combinations.
    return &compute_gradient_composite;
}

#endif // TULPA_HMC_GRADIENT_DISPATCH_H
