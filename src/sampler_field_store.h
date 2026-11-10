// sampler_field_store.h
// The stored form of a ModelData draw: what every sampler writes out for a
// position `q` of its own coordinates.
//
// A kernel samples on the parameterization the density is written in, which
// for a non-centered block is the standardized coordinate z, not the field.
// The stored draw carries the field the likelihood read at that position
// instead: each non-centered GP / SVC / multiscale-GP / temporal-GP block, the
// non-centered multiscale trend, seasonal and short-term arm is replaced by its
// forward transform, and every block whose level is removed on its way into eta
// (SVC, TVC, the non-centred multiscale trend and seasonal arm) is stored
// centred. Every reader of `$draws` therefore sees
// one convention whatever backend produced them, and reads them back at the
// centered parameterization, on which the forward transform is the identity.
//
// The scales come off `q` through safe_exp, the same bounded map the
// log-posterior's own transform uses (nngp_nc_term_apply.cpp), so the stored
// field is the same function of q as the field the likelihood saw and a large
// draw gives a bounded reconstruction rather than Inf * 0. Each block is
// independent (a model can carry more than one structured term at once), so
// every one is applied to `row` on its own condition.

#ifndef TULPA_SAMPLER_FIELD_STORE_H
#define TULPA_SAMPLER_FIELD_STORE_H

#include "hmc_gp.h"
#include "hmc_sampler.h"
#include "sampler_log_prob.h"

#include <cstddef>
#include <vector>

#include <tulpa/sum_to_zero.h>
#include "tulpa_priors_temporal.h"
#include "tulpa_priors_mstemporal.h"

namespace tulpa {

// Overwrite the field blocks of `row` (a copy of `q`) with their stored form.
// `q` stays the sampled coordinates: every transform reads from it.
inline void store_field_draw(double* row, const std::vector<double>& q,
                             const tulpa_hmc::ModelData& data,
                             const tulpa_hmc::ParamLayout& layout) {
    // POD-pointer TLS (constant init, no thread-atexit destructor): a
    // lazily-initialized thread_local object here corrupts the heap under the
    // mingw toolchain when chains run in parallel. The workspace intentionally
    // leaks per thread.
    static thread_local tulpa_gp::NNGPNCWorkspace* ws_p = nullptr;
    if (!ws_p) ws_p = new tulpa_gp::NNGPNCWorkspace();
    tulpa_gp::NNGPNCWorkspace& ws = *ws_p;

    if (data.gp_parameterization == 1 && data.has_gp && layout.is_gp) {
        const double sigma2 = tulpa::math::safe_exp(q[layout.log_sigma2_gp_idx]);
        const double phi = tulpa::math::safe_exp(q[layout.log_phi_gp_idx]);
        tulpa_gp::nngp_nc_forward(&q[layout.gp_w_start], sigma2, phi,
                                  tulpa_gp::make_gp_nc_view(data.gp_data), ws);
        const int n_gp = data.gp_data.n_obs;
        for (int i = 0; i < n_gp; i++) row[layout.gp_w_start + i] = ws.w[i];
    }

    // SVC on the NNGP path. An HSGP SVC carries basis coefficients, not a
    // field, and every reader reconstructs its centred field from them. Under
    // the non-centered parameterization the slice becomes the reconstructed
    // w; under the centered one it already is w. Either way it is stored
    // centred, as svc_center_eta feeds it to the likelihood.
    if (data.has_svc && layout.has_svc && !data.svc_is_hsgp) {
        const auto& sv = data.svc_data;
        const bool nc = (data.svc_parameterization == 1);
        const tulpa_gp::NNGPNCView view = tulpa_gp::make_svc_nc_view(sv);
        const int n_svc = sv.n_obs;
        for (int j = 0; j < sv.n_svc; j++) {
            const int w0 = layout.svc_w_start + j * n_svc;
            if (nc) {
                const double sigma2 =
                    tulpa::math::safe_exp(q[layout.log_sigma2_svc_start + j]);
                const double phi =
                    tulpa::math::safe_exp(q[layout.log_phi_svc_start + j]);
                tulpa_gp::nngp_nc_forward(&q[w0], sigma2, phi, view, ws);
                for (int i = 0; i < n_svc; i++) row[w0 + i] = ws.w[i];
            }
            (void)tulpa::s2z_centre_component(row, w0, n_svc);
        }
    }

    // TVC: each (group, term) block's level is removed by tvc_center_eta() on
    // its way into the linear predictor.
    if (data.has_tvc && layout.has_tvc && data.tvc_data.n_tvc > 0) {
        const auto& tv = data.tvc_data;
        tulpa::s2z_centre_blocks(row + layout.tvc_w_start,
                                 tv.n_groups * tv.n_tvc, tv.n_times);
    }

    if (layout.has_temporal) {
        std::vector<double> f;
        double tau = 0, rho = 0, sigma2 = 0, phi = 0;
        (void)tulpa::priors::compute_temporal_prior(q, data, layout, f,
                                                   tau, rho, sigma2, phi);
        if (data.temporal_type == TemporalType::RW1 || data.temporal_type == TemporalType::RW2) {
            if (data.temporal_group_centering)
                tulpa::s2z_centre_blocks(f.data(), data.n_temporal_groups, data.n_times);
            else
                tulpa::s2z_centre_component(f.data(), 0, static_cast<int>(f.size()));
        }
        std::copy(f.begin(), f.end(), row + layout.temporal_start);
    }

    // Non-centred multiscale trend: the sampled slice is the walk's level,
    // slope and standardized innovations; the stored draw is the walk
    // multiscale_trend_values() hands the likelihood, centred as
    // temporal_effects_by_time centres it on its way into eta.
    if (data.has_multiscale_temporal && layout.has_multiscale_temporal &&
        tulpa::priors::multiscale_trend_noncentered(data, layout)) {
        std::vector<double> trend;
        (void)tulpa::priors::multiscale_trend_values<double>(q, data, layout,
                                                             trend);
        const int n_trend = layout.trend_end - layout.trend_start;
        for (int t = 0; t < n_trend; t++) row[layout.trend_start + t] = trend[t];
        (void)tulpa::s2z_centre_component(row, layout.trend_start, n_trend);
    }

    // Non-centred multiscale seasonal arm: the sampled slice is the cyclic
    // walk's level and standardized coordinates; the stored draw is the walk
    // multiscale_seasonal_values() hands the likelihood, centred as
    // temporal_effects_by_time centres it on its way into eta.
    if (data.has_multiscale_temporal && layout.has_multiscale_temporal &&
        tulpa::priors::multiscale_seasonal_noncentered(data, layout)) {
        std::vector<double> seasonal;
        (void)tulpa::priors::multiscale_seasonal_values<double>(q, data, layout,
                                                                seasonal);
        const int n_seasonal = layout.seasonal_end - layout.seasonal_start;
        for (int t = 0; t < n_seasonal; t++)
            row[layout.seasonal_start + t] = seasonal[t];
        (void)tulpa::s2z_centre_component(row, layout.seasonal_start, n_seasonal);
    }

    // Non-centred multiscale short-term arm: the sampled slice is its
    // standardized innovations; the stored draw is the AR1 / IID field
    // multiscale_short_values() hands the likelihood. The arm is proper and
    // enters eta uncentred.
    if (data.has_multiscale_temporal && layout.has_multiscale_temporal &&
        tulpa::priors::multiscale_short_noncentered(data, layout)) {
        std::vector<double> short_term;
        (void)tulpa::priors::multiscale_short_values<double>(q, data, layout,
                                                             short_term);
        const int n_short = layout.short_term_end - layout.short_term_start;
        for (int t = 0; t < n_short; t++)
            row[layout.short_term_start + t] = short_term[t];
    }

    // Non-centered multiscale GP: the same transform, once per scale.
    if (data.msgp_parameterization == 1 && data.has_multiscale_gp &&
        layout.is_multiscale_gp && !data.msgp_is_hsgp) {
        const auto& ms = data.multiscale_gp_data;
        const int n_ms = ms.n_obs;
        tulpa_gp::nngp_nc_forward(
            &q[layout.gp_local_start],
            tulpa::math::safe_exp(q[layout.log_sigma2_gp_local_idx]),
            tulpa::math::safe_exp(q[layout.log_phi_gp_local_idx]),
            tulpa_gp::make_msgp_nc_view_local(ms), ws);
        for (int i = 0; i < n_ms; i++) row[layout.gp_local_start + i] = ws.w[i];
        tulpa_gp::nngp_nc_forward(
            &q[layout.gp_regional_start],
            tulpa::math::safe_exp(q[layout.log_sigma2_gp_regional_idx]),
            tulpa::math::safe_exp(q[layout.log_phi_gp_regional_idx]),
            tulpa_gp::make_msgp_nc_view_regional(ms), ws);
        for (int i = 0; i < n_ms; i++) {
            row[layout.gp_regional_start + i] = ws.w[i];
        }
    }
}

// The stored form of every row of a draw matrix (n_draws x total_params,
// column-major), in place. A kernel's per-draw log posterior is a function of
// its own coordinates, so it is read off the rows before they are stored.
inline void store_field_rows(Rcpp::NumericMatrix& draws,
                             const tulpa_hmc::ModelData& data,
                             const tulpa_hmc::ParamLayout& layout) {
    const int n = draws.nrow();
    const int D = draws.ncol();
    std::vector<double> q(static_cast<std::size_t>(D)), row(q.size());
    for (int s = 0; s < n; s++) {
        for (int j = 0; j < D; j++) q[j] = draws(s, j);
        row = q;
        store_field_draw(row.data(), q, data, layout);
        for (int j = 0; j < D; j++) draws(s, j) = row[j];
    }
}

// A backend's draw matrix made the stored one: the per-draw log posterior is
// evaluated on the sampled coordinates and returned, then the rows are put in
// their stored form.
inline Rcpp::NumericVector stored_draws(Rcpp::NumericMatrix& draws,
                                        const tulpa_hmc::ModelData& data,
                                        const tulpa_hmc::ParamLayout& layout) {
    Rcpp::NumericVector log_prob = sampler_log_prob_rows(draws, data, layout);
    store_field_rows(draws, data, layout);
    return log_prob;
}

}  // namespace tulpa

#endif  // TULPA_SAMPLER_FIELD_STORE_H
