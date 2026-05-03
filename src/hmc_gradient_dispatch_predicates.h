// hmc_gradient_dispatch_predicates.h
// Named guards documenting which feature combinations each specialized
// hand-coded gradient kernel supports.
//
// Included from hmc_gradient_dispatch.h inside namespace tulpa_hmc.
//
// Each predicate states the kernel's support contract as positive
// activation conditions plus the exclusion of features the kernel does
// not handle. Keeping these one-per-kernel makes it clear what changes
// when a new latent-structure feature is added: every kernel that does
// not yet support the new feature must add it to its exclusion list.

#ifndef TULPA_HMC_GRADIENT_DISPATCH_PREDICATES_H
#define TULPA_HMC_GRADIENT_DISPATCH_PREDICATES_H

// ----------------------------------------------------------------------
// Cross-kernel exotic-feature aggregate
// Used by the crossed-RE early exit. "Exotic" means a feature that
// pushes dispatch off the analytical fast path even when the rest of
// the model is otherwise well-behaved.
// ----------------------------------------------------------------------
inline bool has_exotic_feature(const ParamLayout& layout) {
    return layout.is_temporal_gp || layout.has_tvc ||
           layout.has_multiscale_temporal || layout.is_gp ||
           layout.is_multiscale_gp || layout.has_svc ||
           layout.has_latent;
}

// ----------------------------------------------------------------------
// HSGP spatial: HSGP basis with no other latent structure.
// ----------------------------------------------------------------------
inline bool can_use_hsgp_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.is_hsgp && data.has_hsgp &&
           !layout.has_spatiotemporal &&
           !layout.has_latent && !layout.has_svc && !layout.has_re_slopes &&
           !layout.has_multiscale_temporal &&
           !layout.is_temporal_gp && !layout.has_tvc;
}

// ----------------------------------------------------------------------
// Collapsed GP: GP with inner state marginalized.
// ----------------------------------------------------------------------
inline bool can_use_gp_collapsed(const ModelData& data, const ParamLayout& layout) {
    return layout.is_gp_collapsed && data.has_gp && data.gp_collapsed;
}

// ----------------------------------------------------------------------
// Collapsed ICAR/BYM2: spatial inner state marginalized.
// ----------------------------------------------------------------------
inline bool can_use_icar_collapsed(const ParamLayout& layout) {
    return layout.is_icar_collapsed || layout.is_bym2_collapsed;
}

// ----------------------------------------------------------------------
// GP spatial: full GP with no other latent structure.
// ----------------------------------------------------------------------
inline bool can_use_gp_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.is_gp && data.has_gp &&
           !layout.has_temporal && !layout.has_re_slopes &&
           !layout.is_temporal_gp && !layout.has_tvc &&
           !layout.has_multiscale_temporal &&
           !layout.has_latent && !layout.has_svc &&
           !layout.has_spatiotemporal;
}

// ----------------------------------------------------------------------
// Multi-scale GP (HSGP variant): no temporal or other exotic features.
// ----------------------------------------------------------------------
inline bool can_use_msgp_hsgp(const ModelData& data, const ParamLayout& layout) {
    return layout.is_multiscale_gp && data.has_multiscale_gp && data.msgp_is_hsgp &&
           !layout.has_re_slopes &&
           !layout.has_latent && !layout.has_spatiotemporal && !layout.has_svc &&
           !layout.is_temporal_gp && !layout.has_tvc &&
           !layout.has_multiscale_temporal &&
           !layout.has_temporal;
}

// ----------------------------------------------------------------------
// Multi-scale GP plus AR1/IID temporal: composite-style kernel.
// ----------------------------------------------------------------------
inline bool can_use_msgp_plus_temporal(const ModelData& data, const ParamLayout& layout) {
    return layout.is_multiscale_gp && data.has_multiscale_gp &&
           layout.has_temporal && !layout.has_re_slopes &&
           !layout.is_temporal_gp && !layout.has_multiscale_temporal &&
           !layout.has_tvc;
}

// ----------------------------------------------------------------------
// Multi-scale GP, no temporal interaction.
// ----------------------------------------------------------------------
inline bool can_use_msgp_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.is_multiscale_gp && data.has_multiscale_gp &&
           !layout.has_re_slopes &&
           !layout.is_temporal_gp && !layout.has_multiscale_temporal &&
           !layout.has_tvc;
}

// ----------------------------------------------------------------------
// GP spatial plus AR1/IID temporal.
// ----------------------------------------------------------------------
inline bool can_use_gp_plus_temporal(const ParamLayout& layout) {
    return layout.is_gp && layout.has_temporal &&
           !layout.is_temporal_gp && !layout.has_re_slopes &&
           !layout.has_tvc && !layout.has_multiscale_temporal &&
           !layout.has_latent && !layout.has_svc && !layout.has_spatiotemporal;
}

// ----------------------------------------------------------------------
// SVC (HSGP basis): no temporal or interacting features.
// ----------------------------------------------------------------------
inline bool can_use_svc_hsgp_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.has_svc && data.has_svc && data.svc_is_hsgp &&
           !layout.has_spatiotemporal &&
           !layout.has_latent && !layout.has_tvc &&
           !layout.is_temporal_gp && !layout.has_multiscale_temporal &&
           !layout.has_temporal && !layout.has_re_slopes;
}

// ----------------------------------------------------------------------
// SVC (full GP basis): no temporal or interacting features.
// ----------------------------------------------------------------------
inline bool can_use_svc_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.has_svc && data.has_svc &&
           !layout.has_temporal && !layout.has_spatiotemporal &&
           !layout.has_latent && !layout.has_tvc && !layout.has_re_slopes &&
           !layout.is_temporal_gp && !layout.has_multiscale_temporal;
}

// ----------------------------------------------------------------------
// TVC: no spatial or interacting features.
// ----------------------------------------------------------------------
inline bool can_use_tvc_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.has_tvc && data.has_tvc &&
           !layout.has_spatial && !layout.has_latent &&
           !layout.is_hsgp && !layout.is_gp && !layout.is_multiscale_gp &&
           !layout.has_svc && !layout.has_re_slopes;
}

// ----------------------------------------------------------------------
// Spatiotemporal interaction: structured ST with non-AR1 temporal.
// ----------------------------------------------------------------------
inline bool can_use_spatiotemporal_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.has_spatiotemporal && !layout.is_st_gp &&
           !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
           !layout.has_latent &&
           data.spatiotemporal_data.type != STType::NONE &&
           layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0 &&
           data.spatiotemporal_data.temporal_type != TemporalType::AR1;
}

// ----------------------------------------------------------------------
// Temporal GP (exponential covariance only).
// ----------------------------------------------------------------------
inline bool can_use_temporal_gp_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.is_temporal_gp && layout.has_temporal &&
           !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
           !layout.has_spatial && !layout.has_latent && !layout.has_svc &&
           !layout.has_re_slopes &&
           data.temporal_gp_data.cov_type == tulpa_temporal_gp::TemporalCovType::EXPONENTIAL;
}

// ----------------------------------------------------------------------
// Multi-scale temporal: no spatial or interacting features.
// ----------------------------------------------------------------------
inline bool can_use_ms_temporal_handcoded(const ParamLayout& layout) {
    return layout.has_multiscale_temporal &&
           !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
           !layout.has_svc && !layout.has_tvc &&
           !layout.has_latent && !layout.has_spatiotemporal &&
           !layout.has_spatial && !layout.has_re_slopes;
}

// ----------------------------------------------------------------------
// Latent factors: no spatial/temporal/SVC and no random slopes.
// ----------------------------------------------------------------------
inline bool can_use_latent_handcoded(const ModelData& data, const ParamLayout& layout) {
    return layout.has_latent && data.latent_n_factors > 0 &&
           !layout.has_spatial && !layout.has_temporal && !layout.has_svc &&
           !layout.is_hsgp && !layout.is_gp && !layout.is_multiscale_gp &&
           !layout.has_re_slopes;
}

#endif // TULPA_HMC_GRADIENT_DISPATCH_PREDICATES_H
