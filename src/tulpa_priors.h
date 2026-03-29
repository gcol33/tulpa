// tulpa_priors.h
// Templated prior computation helpers extracted from compute_log_post_impl.
// Each helper computes prior contributions and fills effect vectors
// that the observation loop uses.
//
// NOTE: This header must be included AFTER hmc_sampler.h which defines
// ModelData, ParamLayout, ModelType, TemporalType

#ifndef TULPA_PRIORS_H
#define TULPA_PRIORS_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "hmc_svc_autodiff.h"
#include "hmc_tvc_autodiff.h"
#include "hmc_latent_autodiff.h"
#include "hmc_temporal_multiscale_autodiff.h"

// Forward declarations - these must be included BEFORE this header
// via hmc_sampler.h: ModelData, ParamLayout, types

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 1. Random effects prior (multi-term, slopes, correlated slopes)
// ============================================================================

template<typename T>
T compute_re_prior(const std::vector<T>& params, const ModelData& data,
                   const ParamLayout& layout,
                   std::vector<T>& re_vals, std::vector<int>& re_term_offsets)
{
    T log_post = T(0.0);

    if (layout.has_re) {
        int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

        // Compute total RE size for pre-allocation
        int total_re_vals = 0;
        re_term_offsets.resize(n_terms);
        for (int t = 0; t < n_terms; t++) {
            re_term_offsets[t] = total_re_vals;
            int n_groups_t = (n_terms > 1 || data.n_re_terms > 0)
                             ? data.re_n_groups_multi[t] : data.n_re_groups;
            int n_coefs_t = layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
            total_re_vals += n_groups_t * n_coefs_t;
        }
        re_vals.resize(total_re_vals, T(0.0));

        for (int t = 0; t < n_terms; t++) {
            int n_groups_t = (n_terms > 1 || data.n_re_terms > 0)
                             ? data.re_n_groups_multi[t] : data.n_re_groups;
            int n_coefs_t = layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
            bool is_correlated = layout.has_re_slopes &&
                                 !layout.re_correlated_multi.empty() &&
                                 layout.re_correlated_multi[t];

            // Extract sigma parameters for this term
            std::vector<T> sigmas_t(n_coefs_t);
            for (int c = 0; c < n_coefs_t; c++) {
                int log_sigma_idx;
                if (layout.has_re_slopes) {
                    log_sigma_idx = layout.log_sigma_re_slopes[t][c];
                } else if (n_terms > 1) {
                    log_sigma_idx = layout.log_sigma_re_multi[t];
                } else {
                    log_sigma_idx = layout.log_sigma_re_idx;
                }
                T log_sigma = params[log_sigma_idx];
                sigmas_t[c] = safe_exp(log_sigma);

                // Half-Cauchy prior on sigma
                log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);
            }

            // Correlated slopes: tanh-Cholesky parameterization + LKJ prior
            std::vector<T> L_flat_t;
            if (is_correlated && n_coefs_t > 1) {
                int chol_start = layout.chol_re_start_multi[t];
                L_flat_t.resize(n_coefs_t * n_coefs_t, T(0.0));

                T log_jac_tanh = T(0.0);
                int chol_idx = 0;
                for (int row = 0; row < n_coefs_t; row++) {
                    T row_sum_sq = T(0.0);
                    for (int col = 0; col < row; col++) {
                        T raw_ij = params[chol_start + chol_idx];
                        T l_ij = safe_tanh(raw_ij);
                        L_flat_t[row * n_coefs_t + col] = l_ij;
                        row_sum_sq = row_sum_sq + l_ij * l_ij;
                        // Jacobian: log|d(tanh)/d(raw)| = log(1 - tanh^2)
                        T sech2 = T(1.0) - l_ij * l_ij;
                        log_jac_tanh = log_jac_tanh + safe_log(safe_max(sech2, T(1e-300)));
                        chol_idx++;
                    }
                    // Diagonal: guaranteed positive since tanh^2 < 1
                    T diag_sq = T(1.0) - row_sum_sq;
                    if (get_value(diag_sq) < 1e-10) {
                        return T(-INFINITY);
                    }
                    L_flat_t[row * n_coefs_t + row] = safe_sqrt(diag_sq);
                }

                // Tanh Jacobian
                log_post = log_post + log_jac_tanh;

                // LKJ(eta=2) prior: p(L) propto det(L*L')^(eta-1)
                T eta_lkj = T(2.0);
                for (int k = 0; k < n_coefs_t; k++) {
                    T L_kk = L_flat_t[k * n_coefs_t + k];
                    log_post = log_post + (eta_lkj - T(1.0)
                               + T((n_coefs_t - k - 1) / 2.0)) * T(2.0) * safe_log(L_kk);
                }

                // Jacobian for Cholesky -> correlation transformation
                for (int k = 1; k < n_coefs_t; k++) {
                    T L_kk = L_flat_t[k * n_coefs_t + k];
                    log_post = log_post + T(n_coefs_t - k) * safe_log(L_kk);
                }
            }

            // Get RE start index
            int re_start_t = (n_terms > 1 || layout.has_re_slopes)
                             ? layout.re_start_multi[t] : layout.re_start;
            int off = re_term_offsets[t];

            if (is_correlated && n_coefs_t > 1) {
                // NC correlated: z ~ N(0,I), re = diag(sigma) * L * z
                for (int g = 0; g < n_groups_t; g++) {
                    for (int c = 0; c < n_coefs_t; c++) {
                        T z_gc = params[re_start_t + g * n_coefs_t + c];
                        log_post = log_post - T(0.5) * z_gc * z_gc;

                        // re[c] = sigma[c] * (L[c,:] . z[g])
                        T Lz_c = T(0.0);
                        for (int k = 0; k <= c; k++) {
                            Lz_c = Lz_c + L_flat_t[c * n_coefs_t + k]
                                   * params[re_start_t + g * n_coefs_t + k];
                        }
                        re_vals[off + g * n_coefs_t + c] = sigmas_t[c] * Lz_c;
                    }
                }
            } else if (data.re_parameterization == 1) {
                // NC uncorrelated: z ~ N(0,I), re = sigma * z
                for (int g = 0; g < n_groups_t; g++) {
                    for (int c = 0; c < n_coefs_t; c++) {
                        T z_gc = params[re_start_t + g * n_coefs_t + c];
                        log_post = log_post - T(0.5) * z_gc * z_gc;
                        re_vals[off + g * n_coefs_t + c] = sigmas_t[c] * z_gc;
                    }
                }
            } else {
                // Centered: re ~ N(0, sigma^2)
                for (int g = 0; g < n_groups_t; g++) {
                    for (int c = 0; c < n_coefs_t; c++) {
                        T re_val = params[re_start_t + g * n_coefs_t + c];
                        T tau_c = T(1.0) / (sigmas_t[c] * sigmas_t[c] + T(1e-10));
                        log_post = log_post - T(0.5) * tau_c * re_val * re_val;
                        log_post = log_post + T(0.5) * safe_log(tau_c);
                        re_vals[off + g * n_coefs_t + c] = re_val;
                    }
                }
            }
        }
    }

    return log_post;
}

// ============================================================================
// 2. Spatial ICAR/BYM2 prior
// ============================================================================

template<typename T>
T compute_spatial_icar_bym2_prior(const std::vector<T>& params, const ModelData& data,
                                   const ParamLayout& layout,
                                   const T*& phi_spatial_out, T& tau_spatial_out,
                                   T& sigma_s_bym2_out, T& sigma_u_bym2_out,
                                   const T*& theta_bym2_out)
{
    T log_post = T(0.0);

    if (layout.has_spatial) {
        phi_spatial_out = &params[layout.spatial_start];

        if (layout.is_bym2) {
            // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
            T sigma_total_bym2 = safe_exp(params[layout.log_sigma_bym2_idx]);
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2 = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            sigma_s_bym2_out = sigma_total_bym2 * sqrt(rho_bym2);
            sigma_u_bym2_out = sigma_total_bym2 * sqrt(T(1.0) - rho_bym2);

            theta_bym2_out = &params[layout.theta_bym2_start];

            // BYM2 Riebler: Half-Cauchy on sigma_total
            T log_sigma = params[layout.log_sigma_bym2_idx];
            log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);

            // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
            // log p(logit_rho) = log(rho) + log(1-rho)
            T rho_bym2_prior = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            log_post = log_post + log(rho_bym2_prior)
                                + log(T(1.0) - rho_bym2_prior);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial_out[i] * phi_spatial_out[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial_out[i] * phi_spatial_out[j];
                    }
                }
            }
            log_post = log_post - T(0.5) * quad_form;

            // Soft sum-to-zero constraint on ICAR phi
            {
                T phi_sum = T(0.0);
                for (int s = 0; s < data.n_spatial_units; s++) phi_sum = phi_sum + phi_spatial_out[s];
                log_post = log_post - T(0.5) * T(0.01) * phi_sum * phi_sum;
            }

            // N(0, I) prior on theta
            for (int s = 0; s < data.n_spatial_units; s++) {
                log_post = log_post - T(0.5) * theta_bym2_out[s] * theta_bym2_out[s];
            }
        } else {
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial_out = safe_exp(log_tau);

            // ICAR: Gamma prior on tau
            log_post = log_post + log_prior_gamma(log_tau, data.tau_spatial_shape, data.tau_spatial_rate);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial_out[i] * phi_spatial_out[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial_out[i] * phi_spatial_out[j];
                    }
                }
            }
            int J = data.n_spatial_units;
            T log_tau_sp = params[layout.log_tau_spatial_idx];
            log_post = log_post + T(0.5 * (J - 1)) * log_tau_sp - T(0.5) * tau_spatial_out * quad_form;
        }
    }

    return log_post;
}

// ============================================================================
// 3. GP spatial prior
// ============================================================================

template<typename T>
T compute_gp_spatial_prior(const std::vector<T>& params, const ModelData& data,
                            const ParamLayout& layout, std::vector<T>& gp_w)
{
    T log_post = T(0.0);

    if (layout.is_gp && data.has_gp) {
        // Extract hyperparameters from log-scale
        T log_sigma2_gp = params[layout.log_sigma2_gp_idx];
        T log_phi_gp = params[layout.log_phi_gp_idx];
        T sigma2_gp = safe_exp(log_sigma2_gp);
        T phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 + Jacobian for log transform
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds + Jacobian
        double phi_val = get_value(phi_gp);
        if (phi_val < data.gp_phi_prior_lower || phi_val > data.gp_phi_prior_upper) {
            return T(-INFINITY);
        }
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects w[0..n_gp-1]
        int n_gp = layout.gp_w_end - layout.gp_w_start;
        gp_w.resize(n_gp);
        for (int k = 0; k < n_gp; k++) {
            gp_w[k] = params[layout.gp_w_start + k];
        }

        // Apply RSR projection if enabled
        if (data.has_rsr && !data.rsr_projection.empty()) {
            std::vector<T> w_projected(data.rsr_n, T(0.0));
            for (int ii = 0; ii < data.rsr_n; ii++) {
                for (int jj = 0; jj < data.rsr_n; jj++) {
                    w_projected[ii] = w_projected[ii]
                        + T(data.rsr_projection[ii * data.rsr_n + jj]) * gp_w[jj];
                }
            }
            gp_w = w_projected;
        }

        // NNGP log-likelihood on spatial effects
        log_post = log_post + tulpa_gp::gp_nngp_log_lik_t(
            gp_w, sigma2_gp, phi_gp, data.gp_data);
    }

    return log_post;
}

// ============================================================================
// 4. Multiscale GP spatial prior
// ============================================================================

template<typename T>
T compute_multiscale_gp_prior(const std::vector<T>& params, const ModelData& data,
                               const ParamLayout& layout, std::vector<T>& ms_gp_effect)
{
    T log_post = T(0.0);

    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        if (data.msgp_is_hsgp) {
            // --- HSGP-MSGP: two independent HSGP evaluations with shared basis ---
            T log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
            T log_ls_local = params[layout.log_phi_gp_local_idx];  // log_lengthscale
            T sigma2_local_h = safe_exp(log_sigma2_local);
            T ls_local = safe_exp(log_ls_local);

            T log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
            T log_ls_regional = params[layout.log_phi_gp_regional_idx];
            T sigma2_regional_h = safe_exp(log_sigma2_regional);
            T ls_regional = safe_exp(log_ls_regional);

            int m_total = data.msgp_hsgp_data.m_total;

            // PC priors on sigma for both scales
            T sigma_local = safe_sqrt(sigma2_local_h);
            double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
            log_post = log_post + T(std::log(rate_local)) - T(rate_local) * sigma_local
                     - safe_log(T(2.0) * sigma_local);
            log_post = log_post + log_sigma2_local * T(0.5);  // Jacobian

            T sigma_regional = safe_sqrt(sigma2_regional_h);
            double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
            log_post = log_post + T(std::log(rate_regional)) - T(rate_regional) * sigma_regional
                     - safe_log(T(2.0) * sigma_regional);
            log_post = log_post + log_sigma2_regional * T(0.5);  // Jacobian

            // LogNormal priors on lengthscales (centered at scale-appropriate ranges)
            T z_local = (log_ls_local - T(data.ms_log_ls_local_mean)) / T(data.ms_log_ls_local_sd);
            log_post = log_post - T(0.5) * z_local * z_local - T(std::log(data.ms_log_ls_local_sd));

            T z_regional = (log_ls_regional - T(data.ms_log_ls_regional_mean)) / T(data.ms_log_ls_regional_sd);
            log_post = log_post - T(0.5) * z_regional * z_regional - T(std::log(data.ms_log_ls_regional_sd));

            // N(0, I) priors on beta coefficients
            for (int j = 0; j < m_total; j++) {
                T beta_local_j = params[layout.gp_local_start + j];
                T beta_regional_j = params[layout.gp_regional_start + j];
                log_post = log_post - T(0.5) * beta_local_j * beta_local_j;
                log_post = log_post - T(0.5) * beta_regional_j * beta_regional_j;
            }

            // Evaluate HSGP spatial effects for both scales separately
            // (matching compute_log_post's separate evaluation order for float precision)
            std::vector<T> f_local(data.N, T(0.0));
            std::vector<T> f_regional(data.N, T(0.0));
            for (int j = 0; j < m_total; j++) {
                double omega_sq = data.msgp_hsgp_data.eigenvalues[j];
                T beta_local_j = params[layout.gp_local_start + j];
                T beta_regional_j = params[layout.gp_regional_start + j];

                // Spectral density for local scale
                T S_local_j = sigma2_local_h * T(std::sqrt(2.0 * M_PI)) * ls_local
                    * safe_exp(T(-0.5) * ls_local * ls_local * T(omega_sq));
                T scaled_local_j = safe_sqrt(S_local_j) * beta_local_j;

                // Spectral density for regional scale
                T S_regional_j = sigma2_regional_h * T(std::sqrt(2.0 * M_PI)) * ls_regional
                    * safe_exp(T(-0.5) * ls_regional * ls_regional * T(omega_sq));
                T scaled_regional_j = safe_sqrt(S_regional_j) * beta_regional_j;

                for (int ii = 0; ii < data.N; ii++) {
                    double phi_ij = data.msgp_hsgp_data.phi_flat[ii * m_total + j];
                    f_local[ii] = f_local[ii] + T(phi_ij) * scaled_local_j;
                    f_regional[ii] = f_regional[ii] + T(phi_ij) * scaled_regional_j;
                }
            }
            ms_gp_effect.resize(data.N, T(0.0));
            for (int ii = 0; ii < data.N; ii++) {
                ms_gp_effect[ii] = f_local[ii] + f_regional[ii];
            }
        } else {
            // --- NNGP-MSGP: standard implementation ---
            T log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
            T log_phi_local = params[layout.log_phi_gp_local_idx];
            T log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
            T log_phi_regional = params[layout.log_phi_gp_regional_idx];

            T sigma2_local_n = safe_exp(log_sigma2_local);
            T phi_local = safe_exp(log_phi_local);
            T sigma2_regional_n = safe_exp(log_sigma2_regional);
            T phi_regional = safe_exp(log_phi_regional);

            // PC priors on sigma2 + Jacobians
            log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
                sigma2_local_n, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
            log_post = log_post + log_sigma2_local;  // Jacobian

            log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
                sigma2_regional_n, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
            log_post = log_post + log_sigma2_regional;  // Jacobian

            // Uniform priors on phi (range) within bounds + Jacobians
            double phi_local_val = get_value(phi_local);
            if (phi_local_val < data.multiscale_gp_data.range_local_lower ||
                phi_local_val > data.multiscale_gp_data.range_local_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_local;  // Jacobian

            double phi_regional_val = get_value(phi_regional);
            if (phi_regional_val < data.multiscale_gp_data.range_regional_lower ||
                phi_regional_val > data.multiscale_gp_data.range_regional_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_regional;  // Jacobian

            // Extract local GP effects
            int n_gp_local = layout.gp_local_end - layout.gp_local_start;
            std::vector<T> ms_gp_w_local(n_gp_local);
            for (int k = 0; k < n_gp_local; k++) {
                ms_gp_w_local[k] = params[layout.gp_local_start + k];
            }

            // Extract regional GP effects
            int n_gp_regional = layout.gp_regional_end - layout.gp_regional_start;
            std::vector<T> ms_gp_w_regional(n_gp_regional);
            for (int k = 0; k < n_gp_regional; k++) {
                ms_gp_w_regional[k] = params[layout.gp_regional_start + k];
            }

            // Apply RSR projection if enabled
            if (data.has_rsr && !data.rsr_projection.empty()) {
                std::vector<T> local_proj(data.rsr_n, T(0.0));
                std::vector<T> regional_proj(data.rsr_n, T(0.0));
                for (int ii = 0; ii < data.rsr_n; ii++) {
                    for (int jj = 0; jj < data.rsr_n; jj++) {
                        local_proj[ii] = local_proj[ii]
                            + T(data.rsr_projection[ii * data.rsr_n + jj]) * ms_gp_w_local[jj];
                        regional_proj[ii] = regional_proj[ii]
                            + T(data.rsr_projection[ii * data.rsr_n + jj]) * ms_gp_w_regional[jj];
                    }
                }
                ms_gp_w_local = local_proj;
                ms_gp_w_regional = regional_proj;
            }

            // Multiscale NNGP log-likelihood for both scales
            log_post = log_post + tulpa_gp::multiscale_gp_log_lik_t(
                ms_gp_w_local, ms_gp_w_regional,
                sigma2_local_n, phi_local, sigma2_regional_n, phi_regional,
                data.multiscale_gp_data);

            // Precompute combined effect at observation level
            ms_gp_effect.resize(data.N, T(0.0));
            for (int ii = 0; ii < data.N; ii++) {
                int loc = data.multiscale_gp_data.obs_to_loc[ii];
                ms_gp_effect[ii] = ms_gp_w_local[loc] + ms_gp_w_regional[loc];
            }
        }
    }

    return log_post;
}

// ============================================================================
// 5. HSGP spatial prior
// ============================================================================

template<typename T>
T compute_hsgp_spatial_prior(const std::vector<T>& params, const ModelData& data,
                              const ParamLayout& layout, std::vector<T>& hsgp_f)
{
    T log_post = T(0.0);

    if (layout.is_hsgp && data.has_hsgp) {
        T log_sigma2_hsgp = params[layout.log_sigma2_hsgp_idx];
        T log_lengthscale_hsgp = params[layout.log_lengthscale_hsgp_idx];
        T sigma2_hsgp = safe_exp(log_sigma2_hsgp);
        T lengthscale_hsgp = safe_exp(log_lengthscale_hsgp);

        int m_total = data.hsgp_data.m_total;

        // Extract beta coefficients
        std::vector<T> hsgp_beta(m_total);
        for (int j = 0; j < m_total; j++) {
            hsgp_beta[j] = params[layout.hsgp_beta_start + j];
        }

        // PC prior on sigma: P(sigma > 1) = 0.01 -> rate = 4.6
        // log p(sigma) = log(rate) - rate*sigma - log(2*sigma)
        T sigma_hsgp = safe_sqrt(sigma2_hsgp);
        T rate_sigma_hsgp = T(4.6);
        log_post = log_post + safe_log(rate_sigma_hsgp) - rate_sigma_hsgp * sigma_hsgp
                   - safe_log(T(2.0) * sigma_hsgp);
        log_post = log_post + log_sigma2_hsgp * T(0.5);  // Jacobian: d(sigma)/d(log_sigma2)

        // LogNormal(0, 1) prior on lengthscale
        // log p(ell) = -0.5 * log(ell)^2  (Jacobian cancels)
        log_post = log_post - T(0.5) * log_lengthscale_hsgp * log_lengthscale_hsgp;

        // N(0, I) prior on beta
        for (int j = 0; j < m_total; j++) {
            log_post = log_post - T(0.5) * hsgp_beta[j] * hsgp_beta[j];
        }

        // Evaluate HSGP spatial effect: f = Phi * (sqrt(S) .* beta)
        // Phi and eigenvalues are double (precomputed data), but sigma2/lengthscale/beta are T
        hsgp_f.resize(data.N, T(0.0));
        for (int j = 0; j < m_total; j++) {
            T S_j = sigma2_hsgp * T(std::sqrt(2.0 * M_PI)) * lengthscale_hsgp
                    * safe_exp(T(-0.5) * lengthscale_hsgp * lengthscale_hsgp
                               * T(data.hsgp_data.eigenvalues[j]));
            T scaled_beta_j = safe_sqrt(S_j) * hsgp_beta[j];
            for (int i = 0; i < data.N; i++) {
                hsgp_f[i] = hsgp_f[i]
                    + T(data.hsgp_data.phi_flat[i * m_total + j]) * scaled_beta_j;
            }
        }
    }

    return log_post;
}

// ============================================================================
// 6. Temporal prior (RW1/RW2/AR1/GP)
// ============================================================================

template<typename T>
T compute_temporal_prior(const std::vector<T>& params, const ModelData& data,
                          const ParamLayout& layout, std::vector<T>& phi_temporal,
                          T& tau_temporal_out, T& rho_ar1_out,
                          T& sigma2_temporal_gp_out, T& phi_temporal_gp_out)
{
    T log_post = T(0.0);

    if (layout.has_temporal) {
        // Extract temporal effects (common to all temporal types)
        int n_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            phi_temporal[t] = params[layout.temporal_start + t];
        }

        if (layout.is_temporal_gp) {
            // Temporal GP: sigma2 and phi (lengthscale) parameters
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            T logit_phi = params[layout.logit_phi_temporal_gp_idx];
            sigma2_temporal_gp_out = safe_exp(log_sigma2);

            // Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
            T sigmoid_phi = inv_logit(logit_phi);
            double phi_lower_lp = data.temporal_gp_phi_prior_lower;
            double phi_range_lp = data.temporal_gp_phi_prior_upper - phi_lower_lp;
            phi_temporal_gp_out = T(phi_lower_lp) + T(phi_range_lp) * sigmoid_phi;
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            tau_temporal_out = safe_exp(log_tau);

            if (layout.is_ar1) {
                T logit_rho = params[layout.logit_rho_ar1_idx];
                rho_ar1_out = inv_logit(logit_rho);
            }
        }

        int T_times = data.n_times;

        if (layout.is_temporal_gp) {
            // Temporal GP: PC prior on sigma2, logit-bounded phi (lengthscale)

            // PC prior on sigma2 (favor smaller variance)
            double rate = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
            T sigma_gp = safe_sqrt(sigma2_temporal_gp_out);
            log_post = log_post + T(std::log(rate)) - T(rate) * sigma_gp - safe_log(T(2.0) * sigma_gp);
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            log_post = log_post + log_sigma2;  // Jacobian for log transform

            // Uniform prior on phi: logit-bounded parameterization guarantees bounds
            // Jacobian: log(phi - lower) + log(upper - phi) - log(range)
            double phi_lower_pr = data.temporal_gp_phi_prior_lower;
            double phi_upper_pr = data.temporal_gp_phi_prior_upper;
            double phi_range_pr = phi_upper_pr - phi_lower_pr;
            log_post = log_post + safe_log(phi_temporal_gp_out - T(phi_lower_pr))
                     + safe_log(T(phi_upper_pr) - phi_temporal_gp_out)
                     - T(std::log(phi_range_pr));

            const bool use_nc = (data.temporal_gp_parameterization == 1);

            // Precompute shared rho[t] and derived quantities once (same dt for all groups)
                std::vector<T> rho_shared(T_times > 1 ? T_times - 1 : 0);
                std::vector<T> log_one_minus_rho2_shared(T_times > 1 ? T_times - 1 : 0);
                std::vector<T> a_shared(T_times > 1 ? T_times - 1 : 0);
                T sigma_t = safe_sqrt(sigma2_temporal_gp_out);
                for (int t = 1; t < T_times; t++) {
                    double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                    rho_shared[t - 1] = safe_exp(T(-dt) / phi_temporal_gp_out);
                    T one_minus_rho2 = T(1.0) - rho_shared[t - 1] * rho_shared[t - 1];
                    T one_minus_rho2_safe = safe_max(one_minus_rho2, T(1e-10));
                    log_one_minus_rho2_shared[t - 1] = safe_log(one_minus_rho2_safe);
                    a_shared[t - 1] = sigma_t * safe_sqrt(one_minus_rho2_safe);
                }

            if (use_nc) {
                // Non-centered: params store z ~ N(0,1)
                // Prior: z ~ N(0, I) for each temporal effect
                for (int t = 0; t < n_temporal; t++) {
                    log_post = log_post - T(0.5) * phi_temporal[t] * phi_temporal[t];
                }

                // Jacobian of transform f = g(z, sigma2, phi):
                // log|det(df/dz)| = T*log(sigma) + 0.5*sum_{t>=1} log(1 - rho_t^2) per group
                T log_jac_per_group = T(T_times) * safe_log(sigma_t);
                for (int t = 1; t < T_times; t++) {
                    log_jac_per_group = log_jac_per_group + T(0.5) * log_one_minus_rho2_shared[t - 1];
                }
                log_post = log_post + T(data.n_temporal_groups) * log_jac_per_group;

                // Forward transform z -> f: overwrite phi_temporal for use in obs loop
                // f[0] = sigma * z[0]
                // f[t] = rho_t * f[t-1] + a_t * z[t]
                std::vector<T> f_reconstructed(n_temporal);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    int off = g * T_times;
                    f_reconstructed[off] = sigma_t * phi_temporal[off];
                    for (int t = 1; t < T_times; t++) {
                        f_reconstructed[off + t] = rho_shared[t - 1] * f_reconstructed[off + t - 1] + a_shared[t - 1] * phi_temporal[off + t];
                    }
                }
                // Replace phi_temporal with reconstructed f for observation loop
                phi_temporal = std::move(f_reconstructed);
            } else {
                // Centered: GP log-likelihood using state-space representation
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    T f0 = phi_temporal[g * T_times];
                    log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2_temporal_gp_out);
                    log_post = log_post - T(0.5) * f0 * f0 / sigma2_temporal_gp_out;

                    for (int t = 1; t < T_times; t++) {
                        T f_prev = phi_temporal[g * T_times + t - 1];
                        T f_curr = phi_temporal[g * T_times + t];

                        T cond_var = sigma2_temporal_gp_out * (T(1.0) - rho_shared[t - 1] * rho_shared[t - 1]);
                        T cond_var_safe = safe_max(cond_var, T(1e-10));
                        T cond_mean = rho_shared[t - 1] * f_prev;
                        T resid = f_curr - cond_mean;

                        log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var_safe);
                        log_post = log_post - T(0.5) * resid * resid / cond_var_safe;
                    }
                }
            }
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_temporal_shape, data.tau_temporal_rate);

            if (data.temporal_type == TemporalType::RW1) {
                // RW1: sum of (phi[t] - phi[t-1])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 1; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t] - phi_temporal[g * T_times + t - 1];
                        quad_form = quad_form + diff * diff;
                    }
                    // Cyclic: add wrap-around edge (phi[0] - phi[T-1])
                    if (data.temporal_cyclic) {
                        T diff_cyclic = phi_temporal[g * T_times] - phi_temporal[g * T_times + T_times - 1];
                        quad_form = quad_form + diff_cyclic * diff_cyclic;
                    }
                }
                // Rank: T for cyclic, T-1 for non-cyclic
                int rank_rw1 = data.temporal_cyclic ? T_times : T_times - 1;
                log_post = log_post + T(0.5 * rank_rw1 * data.n_temporal_groups) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal_out * quad_form;

            } else if (data.temporal_type == TemporalType::RW2) {
                // RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 2; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t]
                               - T(2.0) * phi_temporal[g * T_times + t - 1]
                               + phi_temporal[g * T_times + t - 2];
                        quad_form = quad_form + diff * diff;
                    }
                    // Cyclic: add wrap-around second-order differences
                    if (data.temporal_cyclic && T_times >= 3) {
                        T d2_a = phi_temporal[g * T_times + T_times - 2]
                               - T(2.0) * phi_temporal[g * T_times + T_times - 1]
                               + phi_temporal[g * T_times];
                        T d2_b = phi_temporal[g * T_times + T_times - 1]
                               - T(2.0) * phi_temporal[g * T_times]
                               + phi_temporal[g * T_times + 1];
                        quad_form = quad_form + d2_a * d2_a + d2_b * d2_b;
                    }
                }
                // Rank: T for cyclic, T-2 for non-cyclic
                int rank_rw2 = data.temporal_cyclic ? T_times : T_times - 2;
                log_post = log_post + T(0.5 * rank_rw2 * data.n_temporal_groups) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal_out * quad_form;

            } else if (data.temporal_type == TemporalType::AR1) {
                // AR1: phi[t] | phi[t-1] ~ N(rho * phi[t-1], 1/tau)
                // Uniform(0,1) prior on rho with logit Jacobian: log(rho) + log(1-rho)
                log_post = log_post + safe_log(rho_ar1_out) + safe_log(T(1.0) - rho_ar1_out);

                T sigma2_ar1 = T(1.0) / tau_temporal_out;
                T one_minus_rho2 = T(1.0) - rho_ar1_out * rho_ar1_out;

                for (int g = 0; g < data.n_temporal_groups; g++) {
                    // First time point: phi[0] ~ N(0, sigma^2/(1-rho^2))
                    T var_stationary = sigma2_ar1 / one_minus_rho2;
                    log_post = log_post - T(0.5) * phi_temporal[g * T_times] * phi_temporal[g * T_times] / var_stationary;
                    // Normalization: -0.5 * log(2*pi*var_stationary)
                    log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * var_stationary);

                    // Subsequent: phi[t] | phi[t-1] ~ N(rho*phi[t-1], sigma^2)
                    T log_norm_cond = T(-0.5) * safe_log(T(2.0 * M_PI) * sigma2_ar1);
                    for (int t = 1; t < T_times; t++) {
                        T resid = phi_temporal[g * T_times + t] - rho_ar1_out * phi_temporal[g * T_times + t - 1];
                        log_post = log_post - T(0.5) * tau_temporal_out * resid * resid;
                        log_post = log_post + log_norm_cond;
                    }
                }
            }
        }
    }

    return log_post;
}

// ============================================================================
// 7. Multiscale temporal prior
// ============================================================================

template<typename T>
T compute_multiscale_temporal_prior(const std::vector<T>& params, const ModelData& data,
                                     const ParamLayout& layout, std::vector<T>& ms_temporal_eta)
{
    T log_post = T(0.0);

    if (layout.has_multiscale_temporal) {
        const auto& ms_data = data.multiscale_temporal_data;

        std::vector<T> ms_trend;
        std::vector<T> ms_seasonal;
        std::vector<T> ms_short_term;
        T ms_sigma2_trend = T(1.0);
        T ms_sigma2_seasonal = T(1.0);
        T ms_sigma2_short = T(1.0);
        T ms_rho_short = T(0.5);

        // Trend component
        if (layout.log_sigma2_trend_idx >= 0) {
            T log_sigma2_trend = params[layout.log_sigma2_trend_idx];
            ms_sigma2_trend = safe_exp(log_sigma2_trend);

            int n_trend = layout.trend_end - layout.trend_start;
            ms_trend.resize(n_trend);
            for (int t = 0; t < n_trend; t++) {
                ms_trend[t] = params[layout.trend_start + t];
            }

            // PC prior on sigma2_trend + Jacobian for log transform
            log_post = log_post + tulpa_multiscale_ad::log_prior_sigma2_temporal_pc(
                ms_sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
            log_post = log_post + log_sigma2_trend;  // Jacobian
        }

        // Seasonal component
        if (layout.log_sigma2_seasonal_idx >= 0) {
            T log_sigma2_seasonal = params[layout.log_sigma2_seasonal_idx];
            ms_sigma2_seasonal = safe_exp(log_sigma2_seasonal);

            int n_seasonal = layout.seasonal_end - layout.seasonal_start;
            ms_seasonal.resize(n_seasonal);
            for (int t = 0; t < n_seasonal; t++) {
                ms_seasonal[t] = params[layout.seasonal_start + t];
            }

            // PC prior on sigma2_seasonal + Jacobian
            log_post = log_post + tulpa_multiscale_ad::log_prior_sigma2_temporal_pc(
                ms_sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
            log_post = log_post + log_sigma2_seasonal;  // Jacobian
        }

        // Short-term component
        if (layout.log_sigma2_short_idx >= 0) {
            T log_sigma2_short = params[layout.log_sigma2_short_idx];
            ms_sigma2_short = safe_exp(log_sigma2_short);

            int n_short = layout.short_term_end - layout.short_term_start;
            ms_short_term.resize(n_short);
            for (int t = 0; t < n_short; t++) {
                ms_short_term[t] = params[layout.short_term_start + t];
            }

            // PC prior on sigma2_short + Jacobian
            log_post = log_post + tulpa_multiscale_ad::log_prior_sigma2_temporal_pc(
                ms_sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
            log_post = log_post + log_sigma2_short;  // Jacobian

            // AR1 rho parameter
            if (layout.logit_rho_short_idx >= 0) {
                T logit_rho_short = params[layout.logit_rho_short_idx];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho_short);
                ms_rho_short = T(2.0) * u - T(1.0);

                // Beta(2,2) prior on u + Jacobian for logit transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Beta(2,2)
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Jacobian
            }
        }

        // GMRF log-likelihood for all components
        log_post = log_post + tulpa_multiscale_ad::multiscale_temporal_log_lik(
            ms_trend, ms_seasonal, ms_short_term,
            ms_sigma2_trend, ms_sigma2_seasonal, ms_sigma2_short, ms_rho_short,
            ms_data);

        // Precompute multiscale temporal contribution to linear predictor
        tulpa_multiscale_ad::compute_temporal_eta(
            ms_trend, ms_seasonal, ms_short_term, ms_data, ms_temporal_eta);
    }

    return log_post;
}

// ============================================================================
// 8. SVC (Spatially-Varying Coefficients) prior
// ============================================================================

template<typename T>
T compute_svc_prior(const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<T>& svc_eta)
{
    T log_post = T(0.0);

    if (layout.has_svc && data.svc_data.n_svc > 0) {
        int n_svc = data.svc_data.n_svc;
        int n_obs = data.svc_data.n_obs;

        if (data.svc_is_hsgp) {
            // HSGP-based SVC: basis function approximation
            int m_total = data.svc_hsgp_data.m_total;

            // Per-term: sigma2, lengthscale, beta[m_total]
            svc_eta.resize(n_obs, T(0.0));
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                T sigma2_j = safe_exp(log_sigma2);

                // PC prior on sigma
                T sigma_j = safe_sqrt(sigma2_j);
                double rate_sigma = 4.6;
                log_post = log_post - rate_sigma * sigma_j + T(0.5) * log_sigma2;

                T log_ls = params[layout.log_phi_svc_start + j];
                T ls_j = safe_exp(log_ls);

                // LogNormal(0,1) on lengthscale
                log_post = log_post - T(0.5) * log_ls * log_ls;

                // Extract beta_j and compute f_j = Phi * (sqrt(S_j) * beta_j)
                // N(0, I) prior on beta
                for (int k = 0; k < m_total; k++) {
                    T beta_jk = params[layout.svc_w_start + j * m_total + k];
                    log_post = log_post - T(0.5) * beta_jk * beta_jk;

                    // Compute scaled beta: sqrt(S(eigenvalue_k, sigma2_j, ls_j)) * beta_jk
                    double omega_sq = data.svc_hsgp_data.eigenvalues[k];
                    T S_k = sigma2_j * T(std::sqrt(2.0 * M_PI)) * ls_j *
                            safe_exp(T(-0.5) * ls_j * ls_j * T(omega_sq));
                    T sqrt_S_k = safe_sqrt(S_k);

                    // Accumulate f_j[i] = sum_k phi[i,k] * sqrt_S_k * beta_jk
                    for (int i = 0; i < n_obs; i++) {
                        double phi_ik = data.svc_hsgp_data.phi_flat[i * m_total + k];
                        svc_eta[i] = svc_eta[i] + T(phi_ik) * sqrt_S_k * beta_jk *
                                     T(data.svc_data.X_svc[i * n_svc + j]);
                    }
                }
            }
        } else {
            // NNGP-based SVC (original path)

            // Extract sigma2 (spatial variance) parameters
            std::vector<T> svc_sigma2(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                svc_sigma2[j] = safe_exp(log_sigma2);

                T sigma = safe_sqrt(svc_sigma2[j]);
                double scale = data.svc_sigma2_prior_scale;
                log_post = log_post - safe_log(T(1.0) + sigma * sigma / T(scale * scale));
                log_post = log_post + log_sigma2;
            }

            // Extract phi (spatial range) parameters
            std::vector<T> svc_phi(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_phi = params[layout.log_phi_svc_start + j];
                svc_phi[j] = safe_exp(log_phi);

                double phi_val = get_value(svc_phi[j]);
                if (phi_val < data.svc_phi_prior_lower || phi_val > data.svc_phi_prior_upper) {
                    return T(-INFINITY);
                }
                log_post = log_post + log_phi;
            }

            // Extract SVC values
            int n_svc_params = n_svc * n_obs;
            std::vector<T> svc_w_flat(n_svc_params);
            for (int k = 0; k < n_svc_params; k++) {
                svc_w_flat[k] = params[layout.svc_w_start + k];
            }

            // NNGP prior on each SVC term
            for (int j = 0; j < n_svc; j++) {
                std::vector<T> w_j(n_obs);
                for (int k = 0; k < n_obs; k++) {
                    w_j[k] = svc_w_flat[j * n_obs + k];
                }
                log_post = log_post + tulpa_svc_ad::nngp_log_lik(w_j, svc_sigma2[j], svc_phi[j], data.svc_data);
            }

            // Soft sum-to-zero constraint
            log_post = log_post + tulpa_svc_ad::svc_sum_to_zero_penalty(svc_w_flat, data.svc_data, 1.0);

            // Precompute SVC contribution to linear predictor
            svc_eta.resize(n_obs, T(0.0));
            tulpa_svc_ad::compute_svc_eta(svc_w_flat, data.svc_data, svc_eta);
        }
    }

    return log_post;
}

// ============================================================================
// 9. TVC (Temporally-Varying Coefficients) prior
// ============================================================================

template<typename T>
T compute_tvc_prior(const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<T>& tvc_eta)
{
    T log_post = T(0.0);

    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        int n_tvc = data.tvc_data.n_tvc;
        int n_groups = data.tvc_data.n_groups;
        int n_times = data.tvc_data.n_times;
        int n_obs = data.tvc_data.n_obs;

        // Extract tau (precision) parameters
        std::vector<T> tvc_tau(n_tvc);
        for (int j = 0; j < n_tvc; j++) {
            T log_tau = params[layout.log_tau_tvc_start + j];
            tvc_tau[j] = safe_exp(log_tau);

            // PC prior on tau (exponential prior on sigma = 1/sqrt(tau))
            // P(sigma > U) = alpha  =>  rate = -log(alpha) / U
            T sigma_tvc = T(1.0) / safe_sqrt(tvc_tau[j]);
            T rate_tvc = T(-std::log(0.01) / 1.0);  // P(sigma > 1) = 0.01
            log_post = log_post + safe_log(rate_tvc) - rate_tvc * sigma_tvc
                     - safe_log(T(2.0) * sigma_tvc);
            log_post = log_post + log_tau;  // Jacobian for log transform
        }

        // Extract rho (AR1 correlation) parameters if AR1 structure
        std::vector<T> tvc_rho(n_tvc, T(0.0));
        if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
            for (int j = 0; j < n_tvc; j++) {
                T logit_rho = params[layout.logit_rho_tvc_start + j];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho);
                tvc_rho[j] = T(2.0) * u - T(1.0);

                // Uniform(-1, 1) prior on rho
                // Jacobian for logit((rho+1)/2) transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        std::vector<T> tvc_w_flat(n_tvc_params);
        for (int k = 0; k < n_tvc_params; k++) {
            tvc_w_flat[k] = params[layout.tvc_w_start + k];
        }

        // TVC temporal prior (RW1, RW2, or AR1)
        log_post = log_post + tulpa_tvc_ad::tvc_log_prior(
            tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho
        );

        // Soft sum-to-zero constraint for identifiability
        log_post = log_post + tulpa_tvc_ad::tvc_sum_to_zero_penalty(
            tvc_w_flat, data.tvc_data, 0.001
        );

        // Precompute TVC contribution to linear predictor
        tvc_eta.resize(n_obs, T(0.0));
        tulpa_tvc_ad::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
    }

    return log_post;
}

// ============================================================================
// 10. Latent factors prior
// ============================================================================

template<typename T>
T compute_latent_prior(const std::vector<T>& params, const ModelData& data,
                        const ParamLayout& layout, std::vector<T>& latent_eta)
{
    T log_post = T(0.0);

    if (layout.has_latent && data.latent_n_factors > 0) {
        int K = data.latent_n_factors;
        int N = data.N;

        // Extract log_sigma parameters
        std::vector<T> log_sigma_latent(K);
        for (int k = 0; k < K; k++) {
            log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        }

        // Compute sigma from log_sigma
        std::vector<T> latent_sigma(K);
        tulpa_latent_ad::extract_sigma(latent_sigma, log_sigma_latent);

        // Extract factors and apply constraint
        int n_factor_params = N * K;
        std::vector<T> latent_factors(n_factor_params);
        for (int j = 0; j < n_factor_params; j++) {
            latent_factors[j] = params[layout.latent_factor_start + j];
        }

        // Apply identifiability constraint
        if (data.latent_constraint == 0) {  // SUM_TO_ZERO
            tulpa_latent_ad::apply_sum_to_zero(latent_factors, N, K);
        } else {  // FIRST_ZERO
            tulpa_latent_ad::apply_first_zero(latent_factors, N, K);
        }

        // Sigma prior: Exponential on sigma (PC prior)
        log_post = log_post + tulpa_latent_ad::latent_sigma_log_prior(
            log_sigma_latent, data.latent_sigma_prior_rate
        );

        // Factor prior: N(0, 1) on each factor score
        tulpa_latent::LatentConstraint constraint =
            (data.latent_constraint == 0) ? tulpa_latent::LatentConstraint::SUM_TO_ZERO
                                          : tulpa_latent::LatentConstraint::FIRST_ZERO;
        log_post = log_post + tulpa_latent_ad::latent_factor_log_prior(
            latent_factors, N, K, constraint
        );

        // Precompute latent factor contribution to linear predictor
        latent_eta.resize(N, T(0.0));
        tulpa_latent_ad::latent_contributions_all(latent_eta, latent_factors, latent_sigma, N, K);
    }

    return log_post;
}

// ============================================================================
// 11. Spatiotemporal interaction prior
// ============================================================================

template<typename T>
T compute_st_prior(const std::vector<T>& params, const ModelData& data,
                    const ParamLayout& layout, std::vector<T>& st_delta)
{
    T log_post = T(0.0);

    if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
        // Extract precision parameter
        T log_tau_st = params[layout.log_tau_st_idx];
        T tau_st = safe_exp(log_tau_st);

        // PC prior on tau (exponential on sigma = 1/sqrt(tau))
        T sigma_st = T(1.0) / safe_sqrt(tau_st);
        T lambda_st = T(-std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U);
        log_post = log_post + safe_log(lambda_st) - lambda_st * sigma_st
                 - safe_log(T(2.0) * sigma_st);
        log_post = log_post + log_tau_st;  // Jacobian for log transform

        // AR1 rho parameter
        T rho_st = T(0.0);
        if (layout.logit_rho_st_idx >= 0) {
            T logit_rho_st = params[layout.logit_rho_st_idx];
            T u_st = inv_logit(logit_rho_st);
            rho_st = T(2.0) * u_st - T(1.0);  // Map to (-1, 1)

            // Uniform(-1, 1) prior on rho
            // Jacobian for logit((rho+1)/2) transform
            log_post = log_post + safe_log(u_st) + safe_log(T(1.0) - u_st);
        }

        // GP range parameters
        T phi_st_space = T(1.0);
        T phi_st_time = T(1.0);
        if (layout.is_st_gp) {
            T log_phi_space = params[layout.log_phi_st_space_idx];
            T log_phi_time = params[layout.log_phi_st_time_idx];
            phi_st_space = safe_exp(log_phi_space);
            phi_st_time = safe_exp(log_phi_time);

            // Uniform prior within bounds
            double phi_space_val = get_value(phi_st_space);
            if (phi_space_val < data.st_phi_space_prior_lower ||
                phi_space_val > data.st_phi_space_prior_upper) {
                return T(-INFINITY);
            }
            double phi_time_val = get_value(phi_st_time);
            if (phi_time_val < data.st_phi_time_prior_lower ||
                phi_time_val > data.st_phi_time_prior_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_space + log_phi_time;  // Jacobians
        }

        // Extract delta parameters
        int n_st_params = layout.st_delta_end - layout.st_delta_start;
        st_delta.resize(n_st_params);
        for (int k = 0; k < n_st_params; k++) {
            st_delta[k] = params[layout.st_delta_start + k];
        }

        int S = data.spatiotemporal_data.n_spatial;
        int T_st = data.spatiotemporal_data.n_times;

        // NC reparameterization for Type IV
        const bool st_use_nc = (data.st_parameterization == 1 &&
                                data.spatiotemporal_data.type == STType::TYPE_IV);

        if (st_use_nc) {
            // Forward transform: delta = z / sqrt(tau_st)
            T inv_scale = T(1.0) / safe_sqrt(tau_st);
            std::vector<T> st_delta_nc(n_st_params);
            for (int k = 0; k < n_st_params; k++) {
                st_delta_nc[k] = st_delta[k] * inv_scale;
            }

            // NC prior: -0.5 * z^T (Q_s ⊗ Q_t) z  (tau-free GMRF)
            // Type IV Kronecker quadratic form on z (with tau=1)
            // Compute per-spatial-unit temporal GMRF quadratic form
            T nc_quad = T(0.0);
            for (int s = 0; s < S; s++) {
                // Extract temporal series for this spatial unit
                // Apply spatial neighbor structure
                T spatial_quad_s = T(0.0);
                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                    for (int t = 1; t < T_st; t++) {
                        T diff = st_delta[s * T_st + t] - st_delta[s * T_st + t - 1];
                        spatial_quad_s = spatial_quad_s + diff * diff;
                    }
                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                    for (int t = 2; t < T_st; t++) {
                        T d2 = st_delta[s * T_st + t]
                             - T(2.0) * st_delta[s * T_st + t - 1]
                             + st_delta[s * T_st + t - 2];
                        spatial_quad_s = spatial_quad_s + d2 * d2;
                    }
                }
                // Now multiply by spatial structure (neighbor weights)
                int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                    : data.spatiotemporal_data.n_neighbors[s];
                nc_quad = nc_quad + T(n_neigh) * spatial_quad_s;
                // Off-diagonal: -2 * (neighbor pairs)
                if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                    int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                    int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                    for (int jj = row_start_s; jj < row_end_s; jj++) {
                        int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                        if (s2 > s) {
                            // Temporal quadratic form between units s and s2
                            T cross = T(0.0);
                            if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                                for (int t = 1; t < T_st; t++) {
                                    T diff_s = st_delta[s * T_st + t] - st_delta[s * T_st + t - 1];
                                    T diff_s2 = st_delta[s2 * T_st + t] - st_delta[s2 * T_st + t - 1];
                                    cross = cross + diff_s * diff_s2;
                                }
                            } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                                for (int t = 2; t < T_st; t++) {
                                    T d2_s = st_delta[s * T_st + t]
                                           - T(2.0) * st_delta[s * T_st + t - 1]
                                           + st_delta[s * T_st + t - 2];
                                    T d2_s2 = st_delta[s2 * T_st + t]
                                            - T(2.0) * st_delta[s2 * T_st + t - 1]
                                            + st_delta[s2 * T_st + t - 2];
                                    cross = cross + d2_s * d2_s2;
                                }
                            }
                            nc_quad = nc_quad - T(2.0) * cross;
                        }
                    }
                }
            }
            log_post = log_post - T(0.5) * nc_quad;

            // Rank term with actual tau and Jacobian correction
            int rank_space = S - 1;
            int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) : (T_st - 2);
            if (data.spatiotemporal_data.temporal_cyclic) rank_time = T_st;
            int total_rank = rank_space * rank_time;
            int ST_total = S * T_st;
            log_post = log_post + T(0.5 * (total_rank - ST_total)) * safe_log(tau_st);

            // Sum-to-zero on reconstructed delta
            T sum_s = T(0.0), sum_t = T(0.0);
            for (int s = 0; s < S; s++) {
                T row_sum = T(0.0);
                for (int t = 0; t < T_st; t++) {
                    row_sum = row_sum + st_delta_nc[s * T_st + t];
                }
                sum_s = sum_s + row_sum * row_sum;
            }
            for (int t = 0; t < T_st; t++) {
                T col_sum = T(0.0);
                for (int s = 0; s < S; s++) {
                    col_sum = col_sum + st_delta_nc[s * T_st + t];
                }
                sum_t = sum_t + col_sum * col_sum;
            }
            log_post = log_post - T(0.5) * T(0.001) * (sum_s + sum_t);

            // Replace st_delta with reconstructed delta for observation loop
            st_delta = std::move(st_delta_nc);

        } else if (data.st_is_hsgp) {
            // HSGP-ST: spectral basis interaction (centered)
            int M = data.st_hsgp_data.m_total;

            // HSGP-ST hyperparameters
            T sigma2_st_hsgp = safe_exp(params[layout.log_sigma2_st_hsgp_idx]);
            T lengthscale_st_hsgp = safe_exp(params[layout.log_lengthscale_st_hsgp_idx]);

            // PC prior on sigma_st_hsgp: rate=4.6
            T sigma_st_h = safe_sqrt(sigma2_st_hsgp);
            log_post = log_post - T(4.6) * sigma_st_h + T(0.5) * params[layout.log_sigma2_st_hsgp_idx];

            // LogNormal(0,1) on lengthscale
            T log_ls_st = params[layout.log_lengthscale_st_hsgp_idx];
            log_post = log_post - T(0.5) * log_ls_st * log_ls_st;

            // Per-basis-function temporal GMRF prior
            int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                         (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            if (data.spatiotemporal_data.temporal_cyclic) rank_t = T_st;

            for (int j = 0; j < M; j++) {
                double omega_sq = data.st_hsgp_data.eigenvalues[j];
                T S_j = sigma2_st_hsgp * T(std::sqrt(2.0 * M_PI)) * lengthscale_st_hsgp
                    * safe_exp(T(-0.5) * lengthscale_st_hsgp * lengthscale_st_hsgp * T(omega_sq));
                T prec_j = tau_st / safe_max(S_j, T(1e-10));

                // GMRF quadratic form: -0.5 * prec_j * delta_j' Q_t delta_j
                T qf = T(0.0);
                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                    for (int t = 1; t < T_st; t++) {
                        T diff = st_delta[j * T_st + t] - st_delta[j * T_st + t - 1];
                        qf = qf + diff * diff;
                    }
                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                    for (int t = 2; t < T_st; t++) {
                        T d2 = st_delta[j * T_st + t]
                             - T(2.0) * st_delta[j * T_st + t - 1]
                             + st_delta[j * T_st + t - 2];
                        qf = qf + d2 * d2;
                    }
                }
                log_post = log_post + T(0.5 * rank_t) * safe_log(prec_j)
                         - T(0.5) * prec_j * qf;

                // Soft sum-to-zero per basis function
                T sum_j = T(0.0);
                for (int t = 0; t < T_st; t++) sum_j = sum_j + st_delta[j * T_st + t];
                log_post = log_post - T(0.5) * T(0.001) * sum_j * sum_j;
            }

        } else {
            // Centered parameterization (ICAR/BYM2 spatial: Types I-IV)
            if (data.spatiotemporal_data.type == STType::TYPE_I) {
                // IID: delta[s,t] ~ N(0, 1/tau)
                T quad = T(0.0);
                for (int k = 0; k < n_st_params; k++) {
                    quad = quad + st_delta[k] * st_delta[k];
                }
                log_post = log_post + T(0.5 * n_st_params) * safe_log(tau_st)
                         - T(0.5) * tau_st * quad;

            } else if (data.spatiotemporal_data.type == STType::TYPE_II) {
                // Structured time at each location
                for (int s = 0; s < S; s++) {
                    T quad = T(0.0);
                    if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                        for (int t = 1; t < T_st; t++) {
                            T diff = st_delta[s * T_st + t] - st_delta[s * T_st + t - 1];
                            quad = quad + diff * diff;
                        }
                        if (data.spatiotemporal_data.temporal_cyclic) {
                            T diff_c = st_delta[s * T_st] - st_delta[s * T_st + T_st - 1];
                            quad = quad + diff_c * diff_c;
                        }
                        int rank = data.spatiotemporal_data.temporal_cyclic ? T_st : T_st - 1;
                        log_post = log_post + T(0.5 * rank) * safe_log(tau_st)
                                 - T(0.5) * tau_st * quad;
                    } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                        for (int t = 2; t < T_st; t++) {
                            T d2 = st_delta[s * T_st + t]
                                 - T(2.0) * st_delta[s * T_st + t - 1]
                                 + st_delta[s * T_st + t - 2];
                            quad = quad + d2 * d2;
                        }
                        if (data.spatiotemporal_data.temporal_cyclic && T_st >= 3) {
                            T d2_a = st_delta[s * T_st + T_st - 2]
                                   - T(2.0) * st_delta[s * T_st + T_st - 1]
                                   + st_delta[s * T_st];
                            T d2_b = st_delta[s * T_st + T_st - 1]
                                   - T(2.0) * st_delta[s * T_st]
                                   + st_delta[s * T_st + 1];
                            quad = quad + d2_a * d2_a + d2_b * d2_b;
                        }
                        int rank = data.spatiotemporal_data.temporal_cyclic ? T_st : T_st - 2;
                        log_post = log_post + T(0.5 * rank) * safe_log(tau_st)
                                 - T(0.5) * tau_st * quad;
                    }
                }

            } else if (data.spatiotemporal_data.type == STType::TYPE_III) {
                // Structured space at each time point (ICAR)
                int rank_s = S - 1;
                for (int t = 0; t < T_st; t++) {
                    // Compute ICAR quadratic form for spatial field at time t
                    T quad = T(0.0);
                    for (int s = 0; s < S; s++) {
                        T delta_st = st_delta[s * T_st + t];
                        int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                            : data.spatiotemporal_data.n_neighbors[s];
                        quad = quad + T(n_neigh) * delta_st * delta_st;
                        if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                            int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                            int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                            for (int jj = row_start_s; jj < row_end_s; jj++) {
                                int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                                if (s2 > s) {
                                    T delta_s2t = st_delta[s2 * T_st + t];
                                    quad = quad - T(2.0) * delta_st * delta_s2t;
                                }
                            }
                        }
                    }
                    log_post = log_post + T(0.5 * rank_s) * safe_log(tau_st)
                             - T(0.5) * tau_st * quad;
                }

            } else if (data.spatiotemporal_data.type == STType::TYPE_IV) {
                // Kronecker: Q_delta = Q_s ⊗ Q_t
                // Compute using per-spatial-unit temporal GMRF
                for (int s = 0; s < S; s++) {
                    T spatial_quad_s = T(0.0);
                    if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                        for (int t = 1; t < T_st; t++) {
                            T diff = st_delta[s * T_st + t] - st_delta[s * T_st + t - 1];
                            spatial_quad_s = spatial_quad_s + diff * diff;
                        }
                    } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                        for (int t = 2; t < T_st; t++) {
                            T d2 = st_delta[s * T_st + t]
                                 - T(2.0) * st_delta[s * T_st + t - 1]
                                 + st_delta[s * T_st + t - 2];
                            spatial_quad_s = spatial_quad_s + d2 * d2;
                        }
                    }
                    int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                        : data.spatiotemporal_data.n_neighbors[s];
                    T weighted_quad = T(n_neigh) * spatial_quad_s;

                    if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                        int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                        int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                        for (int jj = row_start_s; jj < row_end_s; jj++) {
                            int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                            if (s2 > s) {
                                T cross = T(0.0);
                                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                                    for (int t = 1; t < T_st; t++) {
                                        T diff_s_t = st_delta[s * T_st + t] - st_delta[s * T_st + t - 1];
                                        T diff_s2_t = st_delta[s2 * T_st + t] - st_delta[s2 * T_st + t - 1];
                                        cross = cross + diff_s_t * diff_s2_t;
                                    }
                                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                                    for (int t = 2; t < T_st; t++) {
                                        T d2_s = st_delta[s * T_st + t]
                                               - T(2.0) * st_delta[s * T_st + t - 1]
                                               + st_delta[s * T_st + t - 2];
                                        T d2_s2 = st_delta[s2 * T_st + t]
                                                - T(2.0) * st_delta[s2 * T_st + t - 1]
                                                + st_delta[s2 * T_st + t - 2];
                                        cross = cross + d2_s * d2_s2;
                                    }
                                }
                                weighted_quad = weighted_quad - T(2.0) * cross;
                            }
                        }
                    }
                    log_post = log_post - T(0.5) * tau_st * weighted_quad;
                }
                // Rank terms
                int rank_space = S - 1;
                int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) : (T_st - 2);
                if (data.spatiotemporal_data.temporal_cyclic) rank_time = T_st;
                int total_rank = rank_space * rank_time;
                log_post = log_post + T(0.5 * total_rank) * safe_log(tau_st);
            }

            // Soft sum-to-zero constraint
            T sum_s = T(0.0), sum_t = T(0.0);
            for (int s = 0; s < S; s++) {
                T row_sum = T(0.0);
                for (int t = 0; t < T_st; t++) {
                    row_sum = row_sum + st_delta[s * T_st + t];
                }
                sum_s = sum_s + row_sum * row_sum;
            }
            for (int t = 0; t < T_st; t++) {
                T col_sum = T(0.0);
                for (int s = 0; s < S; s++) {
                    col_sum = col_sum + st_delta[s * T_st + t];
                }
                sum_t = sum_t + col_sum * col_sum;
            }
            log_post = log_post - T(0.5) * T(0.001) * (sum_s + sum_t);
        }
    }

    return log_post;
}

// ============================================================================
// 12. Zero-inflation / One-inflation prior
// ============================================================================

template<typename T>
T compute_zi_oi_prior(const std::vector<T>& params, const ModelData& data,
                       const ParamLayout& layout,
                       std::vector<T>& beta_zi, std::vector<T>& beta_oi)
{
    T log_post = T(0.0);

    if (layout.has_zi && data.p_zi > 0) {
        beta_zi.resize(data.p_zi);
        for (int j = 0; j < data.p_zi; j++) {
            beta_zi[j] = params[layout.beta_zi_start + j];
        }
        // Prior on beta_zi: N(0, zi_prior_sd^2)
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd);
        for (int j = 0; j < data.p_zi; j++) {
            log_post = log_post + log_prior_normal(beta_zi[j], tau_zi);
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        beta_oi.resize(data.p_oi);
        for (int j = 0; j < data.p_oi; j++) {
            beta_oi[j] = params[layout.beta_oi_start + j];
        }
        // Prior on beta_oi: N(0, oi_prior_sd^2)
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd);
        for (int j = 0; j < data.p_oi; j++) {
            log_post = log_post + log_prior_normal(beta_oi[j], tau_oi);
        }
    }

    return log_post;
}

} // namespace priors
} // namespace tulpa

#endif
