// tulpa_glmm_eta_draws.cpp
// ----------------------------------------------------------------------------
// The in-sample linear predictor at each posterior draw of a ModelData sampler
// fit, read through the same assembly the sampler's observation loop reads.
//
// A draw row is the full parameter vector `compute_param_layout(data)` lays out
// -- fixed effects, random effects, every latent field, the hyperparameters --
// so eta at observation i is `generic_eta_at()` at that vector: the fixed part
// with its offset, the random effects, the areal / temporal / continuous fields
// and any varying coefficients, after each block's non-centered reconstruction
// and centring. Rebuilding the same sum in R from the draw columns is a second
// assembly, and one that forgets a component still returns a finite eta.
//
// The model is built from exactly the arguments the sampler received, so the
// layout the draws were laid out on and the layout read here are one layout;
// a column count that disagrees is an error rather than a misread.
// ----------------------------------------------------------------------------

#include <Rcpp.h>
#include "fixed_effect_prior.h"
#include <string>
#include <vector>

#include "hmc_sampler.h"           // ModelData / ParamLayout
#include "laplace_spec_fit.h"      // as_offset_vec
#include "log_post_impl.h"         // GenericLogPostState, generic_eta_at
#include "sampler_model_data.h"    // build_sampler_model_inputs
#include "sampler_log_prob.h"      // sampler_log_prob_rows
#include "hmc_gp.h"                // make_msgp_nc_view_{local,regional}
#include "nngp_nc_term_apply.h"    // apply_nngp_nc_term_double

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_tulpa_glmm_eta_draws(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi_spec = R_NilValue
) {
    const double sigma_beta_value =
        tulpa::read_sigma_beta(sigma_beta);
    const int N = y.size();

    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta_value, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);

    const tulpa_hmc::ModelData& data = in.data;
    const tulpa_hmc::ParamLayout& layout = in.layout;
    const int D = layout.total_params;
    if (draws.ncol() != D) {
        Rcpp::stop("cpp_tulpa_glmm_eta_draws: the draws carry %d columns but the "
                   "model lays out %d parameters.", draws.ncol(), D);
    }
    if (data.n_processes != 1) {
        Rcpp::stop("cpp_tulpa_glmm_eta_draws: a built-in family fit carries one "
                   "linear predictor; this model declares %d.",
                   data.n_processes);
    }

    const int S = draws.nrow();
    Rcpp::NumericMatrix out(S, N);
    std::vector<double> params(D);
    double eta = 0.0;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) params[j] = draws(s, j);
        tulpa::GenericLogPostState<double> state;
        tulpa::initialize_generic_state(params, data, layout, state);
        tulpa::precompute_generic_fixed_eta(data, state);
        for (int i = 0; i < N; i++) {
            tulpa::generic_eta_at(i, data, layout, state, &eta);
            out(s, i) = eta;
        }
    }
    return out;
}

// The per-draw log posterior of a ModelData sampler fit, from the same helper
// every non-NUTS backend reports its `log_prob` through, on the model built from
// the arguments the sampler received. Evaluated on a NUTS fit's draws it is
// what that fit's own recorded log_prob has to reproduce.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_tulpa_glmm_log_prob_draws(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi_spec = R_NilValue
) {
    const double sigma_beta_value =
        tulpa::read_sigma_beta(sigma_beta);
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta_value, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);
    return tulpa::sampler_log_prob_rows(draws, in.data, in.layout);
}

// The latent fields of a ModelData sampler fit at their own support, at each
// posterior draw: what every observation of a unit, a time or a location
// receives, so a reader at new rows gathers or maps them rather than
// reassembling them.
//
//   * `spatial_unit` (S x n_units): the areal field's contribution at each
//     unit (`generic_spatial_unit_effect()`: centred per component on the
//     intrinsic branches, the BYM2 combination at the draw's scales);
//   * `temporal` (S x n_groups * n_times, group-major): the temporal field's
//     contribution at each block position (`generic_temporal_effect()`), after
//     a non-centered temporal GP's forward transform;
//   * `gp_w` (S x n_loc) with `gp_sigma2`, `gp_phi`: the NNGP field at the
//     fitted locations, reconstructed from z on the non-centered path;
//   * `hsgp_z` (S x M) with `hsgp_sigma2`, `hsgp_lengthscale`: the N(0, 1)
//     basis coefficients, the field being Phi diag(sqrt(S)) z;
//   * `ms_local`, `ms_regional` with `ms_sigma2_local`, `ms_phi_local`,
//     `ms_sigma2_regional`, `ms_phi_regional`: each scale of a multi-scale
//     field, the field at the fitted locations (NNGP; reconstructed on the
//     non-centered path) or the basis coefficients (HSGP, `ms_is_hsgp`), the
//     `phi` entries then being lengthscales.
//
// An absent field is a zero-column matrix and empty scales.
// [[Rcpp::export]]
Rcpp::List cpp_sampler_field_draws(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi_spec = R_NilValue
) {
    const double sigma_beta_value = tulpa::read_sigma_beta(sigma_beta);
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta_value, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);
    const tulpa_hmc::ModelData& data = in.data;
    const tulpa_hmc::ParamLayout& layout = in.layout;
    const int D = layout.total_params;
    if (draws.ncol() != D) {
        Rcpp::stop("cpp_sampler_field_draws: the draws carry %d columns but "
                   "the model lays out %d parameters.", draws.ncol(), D);
    }
    const int S = draws.nrow();

    const bool areal = layout.has_spatial && layout.spatial_start >= 0;
    const int n_units = areal ? data.n_spatial_units : 0;
    const bool temporal = layout.has_temporal && layout.temporal_start >= 0;
    const int n_temp = temporal ? layout.temporal_end - layout.temporal_start : 0;
    const bool gp = layout.is_gp && data.has_gp;
    const int n_gp = gp ? layout.gp_w_end - layout.gp_w_start : 0;
    const bool hsgp = layout.is_hsgp && data.has_hsgp;
    const int n_hsgp = hsgp ? layout.hsgp_beta_end - layout.hsgp_beta_start : 0;
    const bool ms = layout.is_multiscale_gp && data.has_multiscale_gp;
    const int n_ms = ms ? layout.gp_local_end - layout.gp_local_start : 0;
    const bool ms_nc = ms && data.msgp_parameterization == 1 &&
                       !data.msgp_is_hsgp;
    const bool mst = layout.has_multiscale_temporal &&
                     data.has_multiscale_temporal;
    const int n_mst = mst ? data.multiscale_temporal_data.n_times : 0;

    Rcpp::NumericMatrix spatial_unit(S, n_units), temporal_out(S, n_temp),
        gp_w(S, n_gp), hsgp_z(S, n_hsgp), ms_local(S, n_ms),
        ms_regional(S, n_ms), ms_temporal(S, n_mst);
    Rcpp::NumericVector gp_s2(gp ? S : 0), gp_ph(gp ? S : 0),
        hs_s2(hsgp ? S : 0), hs_ls(hsgp ? S : 0),
        ms_s2l(ms ? S : 0), ms_phl(ms ? S : 0),
        ms_s2r(ms ? S : 0), ms_phr(ms ? S : 0);

    std::vector<double> params(D), w_loc, w_reg;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) params[j] = draws(s, j);
        tulpa::GenericLogPostState<double> state;
        tulpa::initialize_generic_state(params, data, layout, state);
        for (int u = 0; u < n_units; u++) {
            spatial_unit(s, u) =
                tulpa::generic_spatial_unit_effect(u, data, layout, state);
        }
        if (temporal && (int)state.phi_temporal.size() == n_temp) {
            for (int t = 0; t < n_temp; t++) {
                temporal_out(s, t) = tulpa::generic_temporal_effect(t, state);
            }
        }
        if (mst) {
            if ((int)state.ms_temporal_by_time.size() != n_mst) {
                Rcpp::stop("cpp_sampler_field_draws: the multi-scale temporal "
                           "field was not assembled at draw %d.", s + 1);
            }
            for (int t = 0; t < n_mst; t++) {
                ms_temporal(s, t) = state.ms_temporal_by_time[t];
            }
        }
        if (gp) {
            if ((int)state.gp_w.size() < n_gp) {
                Rcpp::stop("cpp_sampler_field_draws: the GP field was not "
                           "reconstructed at draw %d.", s + 1);
            }
            for (int k = 0; k < n_gp; k++) gp_w(s, k) = state.gp_w[k];
            gp_s2[s] = tulpa::math::safe_exp(params[layout.log_sigma2_gp_idx]);
            gp_ph[s] = tulpa::math::safe_exp(params[layout.log_phi_gp_idx]);
        }
        if (hsgp) {
            for (int k = 0; k < n_hsgp; k++) {
                hsgp_z(s, k) = params[layout.hsgp_beta_start + k];
            }
            hs_s2[s] = tulpa::math::safe_exp(params[layout.log_sigma2_hsgp_idx]);
            hs_ls[s] = tulpa::math::safe_exp(params[layout.log_lengthscale_hsgp_idx]);
        }
        if (ms) {
            if (ms_nc) {
                const auto& msd = data.multiscale_gp_data;
                tulpa::apply_nngp_nc_term_double(
                    params, layout.gp_local_start, layout.log_sigma2_gp_local_idx,
                    layout.log_phi_gp_local_idx, n_ms,
                    tulpa_gp::make_msgp_nc_view_local(msd), w_loc);
                tulpa::apply_nngp_nc_term_double(
                    params, layout.gp_regional_start,
                    layout.log_sigma2_gp_regional_idx,
                    layout.log_phi_gp_regional_idx, n_ms,
                    tulpa_gp::make_msgp_nc_view_regional(msd), w_reg);
            }
            for (int k = 0; k < n_ms; k++) {
                ms_local(s, k) = ms_nc ? w_loc[k]
                                       : params[layout.gp_local_start + k];
                ms_regional(s, k) = ms_nc ? w_reg[k]
                                          : params[layout.gp_regional_start + k];
            }
            ms_s2l[s] = tulpa::math::safe_exp(params[layout.log_sigma2_gp_local_idx]);
            ms_phl[s] = tulpa::math::safe_exp(params[layout.log_phi_gp_local_idx]);
            ms_s2r[s] = tulpa::math::safe_exp(params[layout.log_sigma2_gp_regional_idx]);
            ms_phr[s] = tulpa::math::safe_exp(params[layout.log_phi_gp_regional_idx]);
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("spatial_unit") = spatial_unit,
        Rcpp::Named("temporal") = temporal_out,
        Rcpp::Named("n_times") = data.n_times,
        Rcpp::Named("gp_w") = gp_w,
        Rcpp::Named("gp_sigma2") = gp_s2,
        Rcpp::Named("gp_phi") = gp_ph,
        Rcpp::Named("hsgp_z") = hsgp_z,
        Rcpp::Named("hsgp_sigma2") = hs_s2,
        Rcpp::Named("hsgp_lengthscale") = hs_ls,
        Rcpp::Named("ms_is_hsgp") = ms && data.msgp_is_hsgp,
        Rcpp::Named("ms_local") = ms_local,
        Rcpp::Named("ms_regional") = ms_regional,
        Rcpp::Named("ms_sigma2_local") = ms_s2l,
        Rcpp::Named("ms_phi_local") = ms_phl,
        Rcpp::Named("ms_sigma2_regional") = ms_s2r,
        Rcpp::Named("ms_phi_regional") = ms_phr,
        Rcpp::Named("ms_temporal") = ms_temporal);
}
