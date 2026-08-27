// log_post_generic_impl.h
// Generic multi-process log-posterior implementation for LikelihoodSpec models.
//
// Included from log_post_impl.h while namespace tulpa is open. The legacy
// log-posterior still owns the shared math/prior definitions this code uses.

#ifndef TULPA_LOG_POST_GENERIC_IMPL_H
#define TULPA_LOG_POST_GENERIC_IMPL_H

#include <Rcpp.h>  // Rcpp::stop for offset-length validation
#include <type_traits>

// NOTE: this file is included from log_post_impl.h while
// `namespace tulpa { ... }` is already open, so any nested
// `#include "<header_that_opens_namespace_tulpa>"` would create
// `tulpa::tulpa::` symbols and break qualified lookup of
// tulpa::LikelihoodSpec. The joint-NUTS NC transform declarations live
// in spde_nc_apply.h and are pulled in by log_post_impl.h at the top
// (outside any namespace), so they are visible here without a re-include.

constexpr int MAX_PROCESSES = 8;

template<typename T>
using LikelihoodFnT = T(*)(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& logit_oi,
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data
);

template<typename T>
struct GenericLogPostState {
    std::vector<const T*> beta;
    std::vector<T> re_vals;
    std::vector<int> re_term_offsets;
    const T* phi_spatial = nullptr;
    // phi centred per connected component. The augmented ICAR prior gives each
    // component's constant direction the field's own precision, so it has to be
    // removed here rather than left for the prior to pin.
    std::vector<T> phi_spatial_centered;
    T tau_spatial = T(1.0);
    T sigma_s_bym2 = T(1.0);
    T sigma_u_bym2 = T(1.0);
    const T* theta_bym2 = nullptr;
    std::vector<T> gp_w;
    std::vector<T> ms_gp_effect;
    std::vector<T> hsgp_f;
    std::vector<T> spde_w;
    std::vector<T> phi_temporal;
    // Mean removed from the temporal field on its way into eta, and whether to
    // remove it. Only the GLOBAL constant is aliased with the intercept: a
    // per-group walk level shifts eta only for that group's observations, which
    // the data identifies, so it is augmented in the prior but left in eta.
    T phi_temporal_mean = T(0.0);
    bool phi_temporal_center_all = false;
    T tau_temporal = T(1.0);
    T rho_ar1 = T(0.5);
    T sigma2_temporal_gp = T(1.0);
    T phi_temporal_gp = T(1.0);
    std::vector<T> ms_temporal_eta;
    std::vector<T> svc_eta;
    std::vector<T> tvc_eta;
    std::vector<T> latent_eta;
    std::vector<T> st_delta;
    std::vector<T> beta_zi;
    std::vector<T> beta_oi;
    std::vector<std::vector<T>> eta_fixed;
};

// One dispatch for every non-centered block's field reconstruction.
//
// Each block ships one implementation per scalar type the generic log
// posterior is instantiated with, and the choice is made at compile time. The
// four blocks below (GP, multiscale GP, SPDE, SVC) differ only in which three
// functions they name, so the ladder itself -- and the message an unsupported
// type gets -- lives here rather than being copied once per block.
//
// `on_fwd` is null for a block with no forward-mode kernel: the generic path
// samples via arena and verifies via numerical double, so most blocks need
// none, and SPDE's closed-form tangent is the exception rather than the rule.
template<typename T>
static inline void dispatch_nc_transform(
    const char* block,
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<T>& out,
    void (*on_double)(const std::vector<double>&, const ModelData&,
                      const ParamLayout&, std::vector<double>&),
    void (*on_arena)(const std::vector<arena::Var>&, const ModelData&,
                     const ParamLayout&, std::vector<arena::Var>&),
    void (*on_fwd)(const std::vector< ::fwd::Dual>&, const ModelData&,
                   const ParamLayout&, std::vector< ::fwd::Dual>&) = nullptr
) {
    if constexpr (std::is_same_v<T, double>) {
        on_double(params, data, layout, out);
    } else if constexpr (std::is_same_v<T, arena::Var>) {
        on_arena(params, data, layout, out);
    } else if constexpr (std::is_same_v<T, ::fwd::Dual>) {
        if (on_fwd == nullptr) {
            Rcpp::stop("%s non-centered: forward-mode AD is not supported (the "
                       "generic path samples via arena and verifies via "
                       "numerical double).", block);
        }
        on_fwd(params, data, layout, out);
    } else {
        Rcpp::stop("%s non-centered: AD type not supported.", block);
    }
}

template<typename T>
static inline void add_to_shared_processes(
    T* eta,
    const std::vector<bool>& sharing,
    int n_processes,
    const T& effect
) {
    for (int k = 0; k < n_processes; k++) {
        if (sharing[k]) eta[k] = eta[k] + effect;
    }
}

template<typename T>
static T initialize_generic_state(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    GenericLogPostState<T>& state
) {
    T log_post = T(0.0);
    const int np = data.n_processes;

    state.beta.resize(np);
    const double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int k = 0; k < np; k++) {
        state.beta[k] = &params[layout.process_beta_start[k]];
        for (int j = 0; j < layout.process_beta_count[k]; j++) {
            log_post = log_post + log_prior_normal(
                params[layout.process_beta_start[k] + j], tau_beta);
        }
    }

    if (layout.has_re) {
        log_post = log_post + priors::compute_re_prior(
            params, data, layout, state.re_vals, state.re_term_offsets);
    }

    if (layout.has_spatial) {
        log_post = log_post + priors::compute_spatial_icar_bym2_prior(
            params, data, layout, state.phi_spatial, state.tau_spatial,
            state.sigma_s_bym2, state.sigma_u_bym2, state.theta_bym2);
        // CAR_proper is full rank, so its level is identified by the prior and
        // must not be centred away; only the intrinsic branches are.
        if (state.phi_spatial != nullptr && !layout.is_car_proper) {
            // One direction, by design: see the rationale above
            // icar_center_field. The component count is deliberately not an
            // argument, so this call site does not read as though it were.
            priors::icar_center_field(
                state.phi_spatial, data.n_spatial_units,
                state.phi_spatial_centered);
        }
    }

    if (layout.is_gp && data.has_gp) {
        log_post = log_post + priors::compute_gp_spatial_prior(
            params, data, layout, state.gp_w);

        // Non-centered NNGP: compute_gp_spatial_prior added the N(0, I) z prior
        // and left state.gp_w empty; reconstruct the field w = f(z, sigma2,
        // phi) here so the eta accumulator multiplies w (not z) through the
        // spatial map. Dispatch is statically resolved on the AD type:
        //   double     : forward-only.
        //   arena::Var : reverse-mode AD via custom_backward.
        // The generic path samples via arena and verifies via numerical double
        // (AUTODIFF_FWD aliases to arena -- no forward-mode kernel), so no
        // fwd::Dual branch is needed.
        if (data.gp_parameterization == 1) {
            dispatch_nc_transform<T>("GP", params, data, layout, state.gp_w,
                                     &apply_gp_nc_transform_double,
                                     &apply_gp_nc_transform_arena);
        }
    }

    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        log_post = log_post + priors::compute_multiscale_gp_prior(
            params, data, layout, state.ms_gp_effect);

        // Non-centered NNGP-MSGP: compute_multiscale_gp_prior added both
        // scales' N(0, I) z priors and left state.ms_gp_effect empty;
        // reconstruct each scale's field w = f(z, sigma2, phi) and combine
        // them to the per-observation effect here, mirroring the GP block's
        // dispatch above. Dispatch is statically resolved on the AD type,
        // same as GP/SVC.
        if (data.msgp_parameterization == 1 && !data.msgp_is_hsgp) {
            dispatch_nc_transform<T>("Multiscale GP", params, data, layout,
                                     state.ms_gp_effect,
                                     &apply_msgp_nc_transform_double,
                                     &apply_msgp_nc_transform_arena);
        }
    }

    if (layout.is_hsgp && data.has_hsgp) {
        log_post = log_post + priors::compute_hsgp_spatial_prior(
            params, data, layout, state.hsgp_f);
    }

    if (layout.is_spde && data.has_spde) {
        log_post = log_post + priors::compute_spde_prior(
            params, data, layout, state.spde_w);
        // PC prior on (range, sigma), joint-NUTS mode only (stub returns 0
        // when joint_hypers == false). Lives outside compute_spde_prior so
        // the hyper density is computed regardless of whether the latent
        // block contributes a centered or non-centered prior.
        log_post = log_post + priors::compute_spde_hyper_prior<T>(
            params, data, layout);

        // Non-centered modes (joint-NUTS or fixed-hyper): compute the
        // field w = L^{-T} z from the z block sitting in
        // params[spde_w_start..spde_w_end). compute_spde_prior wrote
        // -0.5 sum(z^2) to log_post and left spde_w empty; the eta
        // accumulator below multiplies w through the projection A, so we
        // need state.spde_w populated with w (not z) before the obs loop.
        // Joint: the z->w Jacobian cancels log|Q(theta)|/2 through the
        // (a.ii) adjoint. Fixed-hyper: both are constant and dropped.
        //
        // Dispatch is statically resolved on the AD type. Three flavours:
        //   double      : forward-only.
        //   arena::Var  : reverse-mode AD via custom_backward.
        //   fwd::Dual   : forward-mode AD via the closed-form tangent.
        if (data.spde_data.joint_hypers || data.spde_data.nc_fixed) {
            dispatch_nc_transform<T>("SPDE joint-NUTS", params, data, layout,
                                     state.spde_w,
                                     &apply_spde_nc_transform_double,
                                     &apply_spde_nc_transform_arena,
                                     &apply_spde_nc_transform_fwd);
        }
    }

    if (layout.has_temporal) {
        const int n_temporal = layout.temporal_end - layout.temporal_start;
        state.phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            state.phi_temporal[t] = params[layout.temporal_start + t];
        }
        log_post = log_post + priors::compute_temporal_prior(
            params, data, layout, state.phi_temporal, state.tau_temporal,
            state.rho_ar1, state.sigma2_temporal_gp, state.phi_temporal_gp);
        // Only the intrinsic walks are centred; AR1 and the temporal GP are
        // proper and identify their own level.
        state.phi_temporal_center_all =
            (data.temporal_type == tulpa_temporal::TemporalType::RW1 ||
             data.temporal_type == tulpa_temporal::TemporalType::RW2) &&
            !state.phi_temporal.empty();
        if (state.phi_temporal_center_all) {
            state.phi_temporal_mean = tulpa::s2z_component_mean(
                state.phi_temporal.data(), 0,
                (int)state.phi_temporal.size());
        }
    }

    if (layout.has_multiscale_temporal && data.has_multiscale_temporal) {
        log_post = log_post + priors::compute_multiscale_temporal_prior(
            params, data, layout, state.ms_temporal_eta);
    }

    if (layout.has_svc && data.has_svc) {
        log_post = log_post + priors::compute_svc_prior(
            params, data, layout, state.svc_eta);

        // Non-centered NNGP: compute_svc_prior added each term's N(0, I) z
        // prior and left state.svc_eta empty; reconstruct every term's field
        // w_j = f(z_j, sigma2_j, phi_j), apply the sum-to-zero penalty on the
        // reconstructed w (not z -- the penalty pins the field's own level,
        // same as the centered path), and compute svc_eta from it. SVC has no
        // per-field equivalent of the GP block's plain "leave gp_w for a
        // downstream lookup" because compute_svc_eta / the sum-to-zero
        // penalty are deterministic post-processing SVC needs done once, up
        // front, in T -- not per-observation -- so they run here rather than
        // inside compute_svc_prior. Dispatch is statically resolved on T, same
        // as the GP block above.
        if (data.svc_parameterization == 1 && !data.svc_is_hsgp &&
            data.svc_data.n_svc > 0) {
            std::vector<T> svc_w_flat;
            dispatch_nc_transform<T>("SVC", params, data, layout, svc_w_flat,
                                     &apply_svc_nc_transform_double,
                                     &apply_svc_nc_transform_arena);
            // Identify each term's global level by CENTERING the reconstructed
            // field on its way into eta. On a reconstructed w = L z a penalty
            // on the sum becomes -0.5 * lambda * (v'z)^2 with v = L'1, a rank-1
            // term whose stiffness rides the Vecchia cascade: d(sum w)/dz_i is
            // large for early-ordered z and small for late ones, so the
            // direction is both stiff and wildly anisotropic against the
            // unit-variance prior on z, and a diagonal mass matrix cannot
            // precondition it. Centring imposes the same identification and
            // leaves the prior on z at N(0, I); see tulpa/sum_to_zero.h.
            tulpa_svc_ad::svc_center_eta(svc_w_flat, data.svc_data,
                                         state.svc_eta);
        }
    }

    if (layout.has_tvc && data.has_tvc) {
        log_post = log_post + priors::compute_tvc_prior(
            params, data, layout, state.tvc_eta);
    }

    if (layout.has_latent && data.has_latent) {
        log_post = log_post + priors::compute_latent_prior(
            params, data, layout, state.latent_eta);
    }

    if (layout.has_spatiotemporal && data.has_spatiotemporal) {
        log_post = log_post + priors::compute_st_prior(
            params, data, layout, state.st_delta);
    }

    return log_post + priors::compute_zi_oi_prior(
        params, data, layout, state.beta_zi, state.beta_oi);
}

template<typename T>
static void precompute_generic_fixed_eta(
    const ModelData& data,
    GenericLogPostState<T>& state
) {
    const int np = data.n_processes;
    state.eta_fixed.resize(np);

    for (int k = 0; k < np; k++) {
        const auto& proc = data.processes[k];
        state.eta_fixed[k].assign(data.N, T(0.0));

        if (proc.p > 0) {
            if constexpr (std::is_same_v<T, double>) {
                tulpa_linalg::matvec(proc.X_flat.data(), state.beta[k],
                                     state.eta_fixed[k].data(), data.N, proc.p);
            } else {
                for (int i = 0; i < data.N; i++) {
                    T eta_i = T(0.0);
                    const double* row = &proc.X_flat[i * proc.p];
                    for (int j = 0; j < proc.p; j++) {
                        eta_i = eta_i + T(row[j]) * state.beta[k][j];
                    }
                    state.eta_fixed[k][i] = eta_i;
                }
            }
        }

        // Optional per-process offset on the linear predictor. Empty means
        // "no offset" (treated as zero for every observation). When present
        // it must have length data.N — checked here so a malformed length
        // raises a deterministic error instead of overrunning eta_fixed.
        if (!proc.offset.empty()) {
            if ((int)proc.offset.size() != data.N) {
                Rcpp::stop("ProcessData[%d]: offset length (%d) must equal "
                           "ModelData::N (%d)",
                           k, (int)proc.offset.size(), data.N);
            }
            for (int i = 0; i < data.N; i++) {
                state.eta_fixed[k][i] = state.eta_fixed[k][i] + T(proc.offset[i]);
            }
        }
    }
}

template<typename T>
static T generic_re_effect(
    int i,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (!layout.has_re || state.re_vals.empty()) return T(0.0);

    const int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
    T effect = T(0.0);

    if (layout.has_re_slopes && !data.re_group_multi_flat.empty()) {
        for (int t = 0; t < n_terms; t++) {
            const int flat_idx = i * n_terms + t;
            if (flat_idx >= (int)data.re_group_multi_flat.size()) continue;

            const int group_idx = data.re_group_multi_flat[flat_idx];
            if (group_idx <= 0) continue;

            const int g = group_idx - 1;
            const int n_coefs_t = layout.re_n_coefs_multi[t];
            const int off = state.re_term_offsets[t];
            // Implicit group intercept (coef 0) present unless this is a
            // slope-only block (lme4 `(0 + x | g)`). When absent, every coef
            // is a slope and there is no z = 1 column.
            const bool has_int = re_term_has_intercept(data, t);
            const int n_slopes = n_coefs_t - (has_int ? 1 : 0);
            const int coef0 = has_int ? 1 : 0;  // first slope's coef index
            T term_effect = has_int ? state.re_vals[off + g * n_coefs_t] : T(0.0);

            if (n_slopes > 0 && t < (int)data.re_slope_matrices.size() &&
                !data.re_slope_matrices[t].empty()) {
                for (int s = 0; s < n_slopes; s++) {
                    const double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                    term_effect = term_effect
                        + state.re_vals[off + g * n_coefs_t + coef0 + s] * T(x_slope);
                }
            }
            effect = effect + term_effect;
        }
    } else if (n_terms > 1 && !data.re_group_multi_flat.empty()) {
        for (int t = 0; t < n_terms; t++) {
            const int flat_idx = i * n_terms + t;
            if (flat_idx >= (int)data.re_group_multi_flat.size()) continue;

            const int group_idx = data.re_group_multi_flat[flat_idx];
            if (group_idx > 0) {
                effect = effect + state.re_vals[state.re_term_offsets[t] + group_idx - 1];
            }
        }
    } else if (!data.re_group.empty() && data.re_group[i] > 0) {
        effect = state.re_vals[data.re_group[i] - 1];
    }

    return effect;
}

template<typename T>
static void add_generic_spatial_effect(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        const int s = data.spatial_group[i] - 1;
        // Intrinsic branches read the centred field; CAR_proper is full rank and
        // keeps its own identified level.
        const T* phi = state.phi_spatial_centered.empty()
            ? state.phi_spatial : state.phi_spatial_centered.data();
        const T effect = layout.is_bym2
            ? state.sigma_s_bym2 * phi[s] * T(data.bym2_scale_factor)
                + state.sigma_u_bym2 * state.theta_bym2[s]
            : phi[s];
        add_to_shared_processes(eta, data.sharing.spatial, data.n_processes, effect);
    }

    if (layout.is_gp && !state.gp_w.empty()) {
        add_to_shared_processes(
            eta, data.sharing.spatial, data.n_processes,
            state.gp_w[data.gp_data.obs_to_loc[i]]);
    }

    if (layout.is_multiscale_gp && !state.ms_gp_effect.empty()) {
        add_to_shared_processes(
            eta, data.sharing.spatial, data.n_processes, state.ms_gp_effect[i]);
    }

    if (layout.is_hsgp && !state.hsgp_f.empty()) {
        add_to_shared_processes(
            eta, data.sharing.spatial, data.n_processes, state.hsgp_f[i]);
    }

    if (layout.is_spde && !state.spde_w.empty()) {
        // eta_i += sum_j A_ij * w_j. The projection A is sparse: each obs
        // is a convex combination of ~3 triangle-vertex weights stored in
        // a_rows[i]. Empty a_rows[i] means the observation falls outside
        // the mesh and contributes no spatial effect.
        const auto& row = data.spde_data.a_rows[i];
        if (!row.empty()) {
            T effect = T(0.0);
            for (const auto& ae : row) {
                effect = effect + T(ae.weight) * state.spde_w[ae.mesh_idx];
            }
            add_to_shared_processes(
                eta, data.sharing.spatial, data.n_processes, effect);
        }
    }
}

template<typename T>
static void add_generic_temporal_effect(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (layout.has_temporal && !data.temporal_time_idx.empty() &&
        i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0 &&
        !state.phi_temporal.empty()) {
        const int t = data.temporal_time_idx[i] - 1;
        const int g = (!data.temporal_group_idx.empty() &&
                       i < (int)data.temporal_group_idx.size())
            ? (data.temporal_group_idx[i] - 1) : 0;
        const int idx_t = g * data.n_times + t;
        if (idx_t >= 0 && idx_t < (int)state.phi_temporal.size()) {
            // Intrinsic walks (RW1/RW2) are centred on the way in; their
            // augmented constant carries tau, so leaving it in eta would free
            // the level. AR1 is proper and keeps its own.
            const T eff = state.phi_temporal_center_all
                ? state.phi_temporal[idx_t] - state.phi_temporal_mean
                : state.phi_temporal[idx_t];
            add_to_shared_processes(
                eta, data.sharing.temporal, data.n_processes, eff);
        }
    }

    if (layout.has_multiscale_temporal && !state.ms_temporal_eta.empty()) {
        add_to_shared_processes(
            eta, data.sharing.temporal, data.n_processes, state.ms_temporal_eta[i]);
    }
}

template<typename T>
static void add_generic_named_effects(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    const T re = generic_re_effect(i, data, layout, state);
    if (layout.has_re && !state.re_vals.empty()) {
        add_to_shared_processes(eta, data.sharing.re, data.n_processes, re);
    }

    add_generic_spatial_effect(i, eta, data, layout, state);
    add_generic_temporal_effect(i, eta, data, layout, state);

    if (layout.has_svc && !state.svc_eta.empty() && i < (int)state.svc_eta.size()) {
        add_to_shared_processes(eta, data.sharing.svc, data.n_processes, state.svc_eta[i]);
    }

    if (layout.has_tvc && !state.tvc_eta.empty()) {
        add_to_shared_processes(eta, data.sharing.tvc, data.n_processes, state.tvc_eta[i]);
    }

    if (layout.has_latent && !state.latent_eta.empty() &&
        i < (int)state.latent_eta.size()) {
        add_to_shared_processes(
            eta, data.sharing.latent, data.n_processes, state.latent_eta[i]);
    }
}

template<typename T>
static void add_generic_st_effect(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (!layout.has_spatiotemporal || state.st_delta.empty()) return;

    // Both index conventions are 1-based, and a 0 means the row carries no
    // spatiotemporal effect. Anything outside the field's own extent leaves
    // the row at zero rather than reading past the delta vector.
    T effect = T(0.0);
    if (data.st_is_hsgp) {
        const std::vector<int>& t_index = data.spatiotemporal_data.t_idx;
        if (i >= (int)t_index.size()) return;
        const int t_st = t_index[i] - 1;
        const int T_st = data.spatiotemporal_data.n_times;
        if (t_st < 0 || t_st >= T_st) return;
        const int M = data.st_hsgp_data.m_total;
        const std::size_t phi_off = (std::size_t)i * (std::size_t)M;
        if (phi_off + (std::size_t)M > data.st_hsgp_data.phi_flat.size()) return;
        if (M > 0 &&
            (std::size_t)(M - 1) * (std::size_t)T_st + (std::size_t)t_st
                >= state.st_delta.size()) {
            return;
        }
        for (int j = 0; j < M; j++) {
            effect = effect
                + T(data.st_hsgp_data.phi_flat[phi_off + (std::size_t)j])
                * state.st_delta[(std::size_t)j * (std::size_t)T_st + (std::size_t)t_st];
        }
    } else {
        const std::vector<int>& st_flat = data.spatiotemporal_data.st_flat;
        if (i >= (int)st_flat.size()) return;
        const int st_idx = st_flat[i];
        if (st_idx > 0 && st_idx <= (int)state.st_delta.size()) {
            effect = state.st_delta[st_idx - 1];
        }
    }
    add_to_shared_processes(eta, data.sharing.st, data.n_processes, effect);
}

template<typename T>
static void generic_zi_oi_logits(
    int i,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state,
    T& logit_zi,
    T& logit_oi
) {
    logit_zi = T(0.0);
    logit_oi = T(0.0);

    if (layout.has_zi && data.p_zi > 0) {
        for (int j = 0; j < data.p_zi; j++) {
            logit_zi = logit_zi + T(data.X_zi_flat[i * data.p_zi + j]) * state.beta_zi[j];
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        for (int j = 0; j < data.p_oi; j++) {
            logit_oi = logit_oi + T(data.X_oi_flat[i * data.p_oi + j]) * state.beta_oi[j];
        }
    }
}

// Linear predictor of observation i, every latent component included: the
// precomputed fixed part plus the named effects and the spatiotemporal one.
// `eta` is written, not accumulated, and must hold data.n_processes entries.
//
// The observation loop below and the eta-space working-weight pass
// (hmc_mass_st_gmrf.cpp) both read eta at the same point of the model, so they
// read it from here. A second assembly drifts silently: it produces a finite
// eta whichever component it forgets.
template<typename T>
static inline void generic_eta_at(
    int i,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state,
    T* eta
) {
    for (int k = 0; k < data.n_processes; k++) {
        eta[k] = state.eta_fixed[k][i];
    }
    add_generic_named_effects(i, eta, data, layout, state);
    add_generic_st_effect(i, eta, data, layout, state);
}

template<typename T>
static T compute_generic_likelihood_sum(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LikelihoodFnT<T> likelihood_fn,
    const void* model_response_data,
    const GenericLogPostState<T>& state
) {
    T log_lik = T(0.0);

    for (int i = 0; i < data.N; i++) {
        T eta[MAX_PROCESSES];
        generic_eta_at(i, data, layout, state, eta);

        T logit_zi;
        T logit_oi;
        generic_zi_oi_logits(i, data, layout, state, logit_zi, logit_oi);

        log_lik = log_lik + likelihood_fn(
            i, eta, logit_zi, logit_oi, params, data, layout, model_response_data);
    }

    return log_lik;
}

template<typename T>
T compute_log_post_generic(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LikelihoodFnT<T> likelihood_fn,
    const void* model_response_data,
    bool skip_obs_loop = false,
    bool skip_prior = false
) {
    // The process count sizes the per-observation eta buffer below, so a model
    // wider than the buffer is a model this build cannot evaluate rather than
    // an improbable point: -Inf would reject every proposal, leaving a chain
    // that never moves and no diagnostic saying why.
    if (data.n_processes > MAX_PROCESSES) {
        Rcpp::stop("tulpa: ModelData declares %d processes; the generic "
                   "log-posterior evaluates at most MAX_PROCESSES = %d.",
                   data.n_processes, MAX_PROCESSES);
    }

    GenericLogPostState<T> state;
    T log_post = initialize_generic_state(params, data, layout, state);
    // The prior pass also BUILDS the state the observation loop reads, so
    // `skip_prior` drops its value rather than skipping the pass. Recovering
    // the log-likelihood as (full - prior-only) instead would cost a second
    // full evaluation and cancel the prior in floating point, losing the
    // precision a direct accumulation keeps wherever the prior is large next
    // to the likelihood.
    if (skip_prior) log_post = T(0.0);
    precompute_generic_fixed_eta(data, state);

    if (!skip_obs_loop) {
        log_post = log_post + compute_generic_likelihood_sum(
            params, data, layout, likelihood_fn, model_response_data, state);
    }

    return log_post;
}

double compute_log_post_generic_spec_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop = false,
    bool skip_prior = false);

extern template double compute_log_post_generic<double>(
    const std::vector<double>&,
    const ModelData&,
    const ParamLayout&,
    LikelihoodFnT<double>,
    const void*,
    bool,
    bool);

#endif  // TULPA_LOG_POST_GENERIC_IMPL_H
