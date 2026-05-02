// gibbs_spatial_hsgp.h
// Sliced from gibbs_spatial.h. HSGP helpers and run_gibbs_hsgp.

#pragma once

#include "gibbs_spatial_data.h"

namespace gibbs {

// =========================================================================
// HSGP Gibbs sampler
// Replaces ICAR phi[S] with m² basis coefficients hsgp_beta[j]
// Spatial effect at obs i: f[i] = sum_j Phi[i,j] * sqrt(S_j) * beta_j
// =========================================================================

// Full log-likelihood with HSGP spatial (uses cached f vector)
inline double full_log_lik_hsgp(const double* hsgp_f, const double* beta_num,
                                 const double* beta_denom,
                                 double phi_num, double phi_denom,
                                 const GibbsData& d,
                                 const double* tvc_w = nullptr,
                                 const double* phi_temporal = nullptr) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);

    for (int i = 0; i < d.N; i++) {
        double eta_num = hsgp_f[i];
        double eta_denom = is_binomial ? 0.0 : (d.hsgp_shared ? hsgp_f[i] : 0.0);

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        if (tvc_w != nullptr && d.has_tvc) {
            double tvc_eff = compute_tvc_eta_obs(i, tvc_w, d);
            eta_num += tvc_eff;
            if (!is_binomial && d.tvc_shared) eta_denom += tvc_eff;
        }

        if (d.has_temporal && phi_temporal != nullptr) {
            double temp_eff = compute_temporal_eta_obs(i, phi_temporal, d);
            eta_num += temp_eff;
            if (!is_binomial && d.temporal_shared) eta_denom += temp_eff;
        }

        ll += obs_log_lik(i, eta_num, eta_denom, phi_num, phi_denom, d);
    }
    return ll;
}

// Recompute cached f[N] from beta, sigma2, lengthscale
inline void hsgp_recompute_f(double* hsgp_f, const double* hsgp_beta,
                              double sigma2, double lengthscale,
                              const GibbsData& d) {
    int M = d.hsgp_m_total;
    // Precompute sqrt(S_j)
    std::vector<double> sqrt_S(M);
    for (int j = 0; j < M; j++) {
        double S_j = tulpa_hsgp::spectral_density_se(d.hsgp_eigenvalues[j], sigma2, lengthscale);
        sqrt_S[j] = std::sqrt(std::max(S_j, 1e-20));
    }
    // f[i] = sum_j Phi[i,j] * sqrt(S_j) * beta_j
    for (int i = 0; i < d.N; i++) {
        double f = 0.0;
        for (int j = 0; j < M; j++) {
            f += d.hsgp_Phi_flat[i * M + j] * sqrt_S[j] * hsgp_beta[j];
        }
        hsgp_f[i] = f;
    }
}

inline GibbsResult run_gibbs_hsgp(
    const GibbsData& d,
    int n_iter,
    int n_warmup,
    int thin,
    unsigned int seed,
    bool verbose
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    int N = d.N;
    int M = d.hsgp_m_total;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);
    bool has_phi_num = (d.family == GibbsFamily::NEGBIN_NEGBIN ||
                        d.family == GibbsFamily::NEGBIN_GAMMA ||
                        d.family == GibbsFamily::GAMMA_GAMMA ||
                        d.family == GibbsFamily::LOGNORMAL ||
                        d.family == GibbsFamily::BETA_BINOMIAL);
    bool has_phi_denom = !is_binomial;

    int p_num = d.p_num;
    int p_denom = is_binomial ? 0 : d.p_denom;
    // Params: beta_num, beta_denom, [log_phi_num], [log_phi_denom], log_sigma2_hsgp, log_lengthscale_hsgp
    int n_hyper = 2;  // log_sigma2, log_lengthscale
    if (has_phi_num) n_hyper++;
    if (has_phi_denom) n_hyper++;
    int n_params = p_num + p_denom + n_hyper;

    // Initialize
    std::vector<double> beta_num(p_num, 0.0);
    std::vector<double> beta_denom(p_denom, 0.0);
    std::vector<double> hsgp_beta(M, 0.0);
    double log_sigma2 = 0.0;        // sigma2 = 1
    double log_lengthscale = 0.0;   // lengthscale = 1
    double log_phi_num = std::log(1.0);
    double log_phi_denom = std::log(5.0);

    if (p_num > 0) beta_num[0] = 1.0;
    if (p_denom > 0) beta_denom[0] = 1.0;

    // Cached spatial field f[N]
    std::vector<double> hsgp_f(N, 0.0);
    hsgp_recompute_f(hsgp_f.data(), hsgp_beta.data(),
                      std::exp(log_sigma2), std::exp(log_lengthscale), d);

    // Cached sqrt(S_j) for incremental updates
    std::vector<double> sqrt_S(M);
    for (int j = 0; j < M; j++) {
        double S_j = tulpa_hsgp::spectral_density_se(
            d.hsgp_eigenvalues[j], std::exp(log_sigma2), std::exp(log_lengthscale));
        sqrt_S[j] = std::sqrt(std::max(S_j, 1e-20));
    }

    // Proposal scales
    double hsgp_beta_scale = 0.5;
    double hsgp_hyper_scale = 0.1;
    double beta_scale = 0.1;
    double disp_scale = 0.1;

    // Acceptance tracking
    int hsgp_beta_accept = 0, hsgp_beta_total = 0;
    int hsgp_hyper_accept = 0, hsgp_hyper_total = 0;
    int beta_accept = 0, beta_total = 0;
    int disp_accept = 0, disp_total = 0;

    // TVC
    int tvc_n_w = d.has_tvc ? (d.tvc_n_groups * d.tvc_n_terms * d.tvc_n_times) : 0;
    std::vector<double> tvc_w(tvc_n_w, 0.0);
    std::vector<double> tvc_tau(d.has_tvc ? d.tvc_n_terms : 0, 1.0);
    int tvc_accept = 0, tvc_total = 0;
    const double* tvc_w_ptr = d.has_tvc ? tvc_w.data() : nullptr;

    // Temporal GMRF
    int T_temporal = d.has_temporal ? d.temporal_n_times : 0;
    int n_temporal_groups = d.has_temporal ? d.temporal_n_groups : 0;
    int n_temporal_params = T_temporal * n_temporal_groups;
    std::vector<double> phi_temporal(n_temporal_params, 0.0);
    double log_tau_temporal = std::log(1.0);
    int temporal_accept = 0, temporal_total = 0;
    const double* phi_temporal_ptr = d.has_temporal ? phi_temporal.data() : nullptr;

    // Output
    int n_save = (n_iter - n_warmup) / thin;
    GibbsResult result;
    result.n_params = n_params;
    result.n_save = n_save;
    result.S = 0;  // No site-level phi for HSGP
    result.hsgp_m_total = M;
    result.draws_flat.resize(n_save * n_params);
    result.hsgp_beta_draws_flat.resize(n_save * M);
    result.hsgp_f_draws_flat.resize(n_save * N);
    result.tvc_n_w = tvc_n_w;
    if (d.has_tvc) {
        result.tvc_w_draws_flat.resize(n_save * tvc_n_w);
        result.tvc_tau_draws_flat.resize(n_save * d.tvc_n_terms);
    }
    result.temporal_n_params = n_temporal_params;
    if (d.has_temporal) {
        result.temporal_draws_flat.resize(n_save * n_temporal_params);
    }

    int save_idx = 0;

    // ---- Main Gibbs loop ----
    for (int iter = 0; iter < n_iter; iter++) {
        double phi_num_val = has_phi_num ? std::exp(log_phi_num) : 1.0;
        double phi_denom_val = has_phi_denom ? std::exp(log_phi_denom) : 1.0;

        // ---- 1. Update HSGP basis coefficients via univariate MH ----
        for (int j = 0; j < M; j++) {
            double beta_j_prop = hsgp_beta[j] + hsgp_beta_scale * rnorm(rng);

            // Incremental f update: f_prop[i] = f[i] + Phi[i,j] * sqrt_S[j] * (beta_prop - beta_curr)
            double delta = beta_j_prop - hsgp_beta[j];
            double delta_scaled = sqrt_S[j] * delta;

            // Compute log-likelihood ratio over all obs
            double ll_diff = 0.0;
            for (int i = 0; i < N; i++) {
                double f_prop_i = hsgp_f[i] + d.hsgp_Phi_flat[i * M + j] * delta_scaled;

                double eta_num_c = hsgp_f[i], eta_num_p = f_prop_i;
                double eta_denom_c = is_binomial ? 0.0 : (d.hsgp_shared ? hsgp_f[i] : 0.0);
                double eta_denom_p = is_binomial ? 0.0 : (d.hsgp_shared ? f_prop_i : 0.0);

                for (int p = 0; p < d.p_num; p++) {
                    double xb = d.X_num[i * d.p_num + p] * beta_num[p];
                    eta_num_c += xb; eta_num_p += xb;
                }
                if (!is_binomial) {
                    for (int p = 0; p < d.p_denom; p++) {
                        double xb = d.X_denom[i * d.p_denom + p] * beta_denom[p];
                        eta_denom_c += xb; eta_denom_p += xb;
                    }
                }
                if (d.has_tvc && tvc_w_ptr) {
                    double tvc_eff = compute_tvc_eta_obs(i, tvc_w_ptr, d);
                    eta_num_c += tvc_eff; eta_num_p += tvc_eff;
                    if (!is_binomial && d.tvc_shared) { eta_denom_c += tvc_eff; eta_denom_p += tvc_eff; }
                }
                if (d.has_temporal && phi_temporal_ptr) {
                    double temp_eff = compute_temporal_eta_obs(i, phi_temporal_ptr, d);
                    eta_num_c += temp_eff; eta_num_p += temp_eff;
                    if (!is_binomial && d.temporal_shared) { eta_denom_c += temp_eff; eta_denom_p += temp_eff; }
                }

                ll_diff += obs_log_lik(i, eta_num_p, eta_denom_p, phi_num_val, phi_denom_val, d)
                         - obs_log_lik(i, eta_num_c, eta_denom_c, phi_num_val, phi_denom_val, d);
            }

            // Prior ratio: N(0, 1) on beta (NC parameterization)
            double lp_diff = -0.5 * (beta_j_prop * beta_j_prop - hsgp_beta[j] * hsgp_beta[j]);

            if (std::log(runif(rng)) < ll_diff + lp_diff) {
                hsgp_beta[j] = beta_j_prop;
                // Update cached f incrementally
                for (int i = 0; i < N; i++)
                    hsgp_f[i] += d.hsgp_Phi_flat[i * M + j] * delta_scaled;
                hsgp_beta_accept++;
            }
            hsgp_beta_total++;
        }

        // ---- 2. Update HSGP hyperparams (sigma2, lengthscale) via MH ----
        {
            double log_sigma2_prop = log_sigma2 + hsgp_hyper_scale * rnorm(rng);
            double log_ls_prop = log_lengthscale + hsgp_hyper_scale * rnorm(rng);
            double sigma2_prop = std::exp(log_sigma2_prop);
            double ls_prop = std::exp(log_ls_prop);

            // Recompute f with proposed hyperparams
            std::vector<double> hsgp_f_prop(N);
            std::vector<double> sqrt_S_prop(M);
            for (int j_h = 0; j_h < M; j_h++) {
                double S_j = tulpa_hsgp::spectral_density_se(d.hsgp_eigenvalues[j_h], sigma2_prop, ls_prop);
                sqrt_S_prop[j_h] = std::sqrt(std::max(S_j, 1e-20));
            }
            for (int i = 0; i < N; i++) {
                double f = 0.0;
                for (int j_h = 0; j_h < M; j_h++)
                    f += d.hsgp_Phi_flat[i * M + j_h] * sqrt_S_prop[j_h] * hsgp_beta[j_h];
                hsgp_f_prop[i] = f;
            }

            double ll_curr = full_log_lik_hsgp(hsgp_f.data(), beta_num.data(), beta_denom.data(),
                                                phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);
            double ll_prop = full_log_lik_hsgp(hsgp_f_prop.data(), beta_num.data(), beta_denom.data(),
                                                phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);

            // PC prior on sigma: -lambda*sigma + 0.5*log_sigma2
            double sigma_curr = std::sqrt(std::exp(log_sigma2));
            double sigma_prop = std::sqrt(sigma2_prop);
            double lp_diff = -4.605 * (sigma_prop - sigma_curr)
                           + 0.5 * (log_sigma2_prop - log_sigma2);
            // LogNormal(0,1) on lengthscale
            lp_diff += -0.5 * (log_ls_prop * log_ls_prop - log_lengthscale * log_lengthscale);

            if (std::log(runif(rng)) < ll_prop - ll_curr + lp_diff) {
                log_sigma2 = log_sigma2_prop;
                log_lengthscale = log_ls_prop;
                hsgp_f = hsgp_f_prop;
                sqrt_S = sqrt_S_prop;
                hsgp_hyper_accept++;
            }
            hsgp_hyper_total++;
        }

        // ---- 3. Update beta via block MH ----
        {
            std::vector<double> beta_num_prop(beta_num);
            std::vector<double> beta_denom_prop(beta_denom);
            for (int j = 0; j < p_num; j++) beta_num_prop[j] += beta_scale * rnorm(rng);
            for (int j = 0; j < p_denom; j++) beta_denom_prop[j] += beta_scale * rnorm(rng);

            double ll_curr = full_log_lik_hsgp(hsgp_f.data(), beta_num.data(), beta_denom.data(),
                                                phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);
            double ll_prop = full_log_lik_hsgp(hsgp_f.data(), beta_num_prop.data(), beta_denom_prop.data(),
                                                phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);

            double lp_curr = 0.0, lp_prop = 0.0;
            for (int j = 0; j < p_num; j++) {
                lp_curr -= 0.5 * beta_num[j] * beta_num[j] / 100.0;
                lp_prop -= 0.5 * beta_num_prop[j] * beta_num_prop[j] / 100.0;
            }
            for (int j = 0; j < p_denom; j++) {
                lp_curr -= 0.5 * beta_denom[j] * beta_denom[j] / 100.0;
                lp_prop -= 0.5 * beta_denom_prop[j] * beta_denom_prop[j] / 100.0;
            }

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                beta_num = beta_num_prop;
                beta_denom = beta_denom_prop;
                beta_accept++;
            }
            beta_total++;
        }

        // ---- 4. Update dispersion via MH ----
        if (has_phi_num || has_phi_denom) {
            double log_pn_prop = log_phi_num, log_pd_prop = log_phi_denom;
            if (has_phi_num) log_pn_prop += disp_scale * rnorm(rng);
            if (has_phi_denom) log_pd_prop += disp_scale * rnorm(rng);

            double pn_c = has_phi_num ? std::exp(log_phi_num) : 1.0;
            double pd_c = has_phi_denom ? std::exp(log_phi_denom) : 1.0;
            double pn_p = has_phi_num ? std::exp(log_pn_prop) : 1.0;
            double pd_p = has_phi_denom ? std::exp(log_pd_prop) : 1.0;

            double ll_curr = full_log_lik_hsgp(hsgp_f.data(), beta_num.data(), beta_denom.data(),
                                                pn_c, pd_c, d, tvc_w_ptr, phi_temporal_ptr);
            double ll_prop = full_log_lik_hsgp(hsgp_f.data(), beta_num.data(), beta_denom.data(),
                                                pn_p, pd_p, d, tvc_w_ptr, phi_temporal_ptr);

            double lp_curr = 0.0, lp_prop = 0.0;
            if (has_phi_num) { lp_curr += std::log(pn_c) - 0.5 * pn_c; lp_prop += std::log(pn_p) - 0.5 * pn_p; }
            if (has_phi_denom) { lp_curr += std::log(pd_c) - 0.5 * pd_c; lp_prop += std::log(pd_p) - 0.5 * pd_p; }

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_phi_num = log_pn_prop;
                log_phi_denom = log_pd_prop;
                disp_accept++;
            }
            disp_total++;
        }

        // ---- 5. Update TVC via component-wise MH ----
        if (d.has_tvc) {
            for (int j = 0; j < d.tvc_n_terms; j++) {
                for (int g = 0; g < d.tvc_n_groups; g++) {
                    for (int t = 0; t < d.tvc_n_times; t++) {
                        int w_idx = (g * d.tvc_n_terms + j) * d.tvc_n_times + t;
                        int n_nb = 0; double nb_sum = 0.0;
                        if (t > 0) { nb_sum += tvc_w[(g * d.tvc_n_terms + j) * d.tvc_n_times + t - 1]; n_nb++; }
                        if (t < d.tvc_n_times - 1) { nb_sum += tvc_w[(g * d.tvc_n_terms + j) * d.tvc_n_times + t + 1]; n_nb++; }
                        double cond_prec = tvc_tau[j] * std::max(n_nb, 1);
                        double cond_mean = (n_nb > 0) ? nb_sum / n_nb : 0.0;
                        double w_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

                        double ll_diff = 0.0;
                        for (int idx = d.time_obs_ptr[t]; idx < d.time_obs_ptr[t + 1]; idx++) {
                            int i = d.time_obs_idx[idx];
                            double eta_num_base = hsgp_f[i];
                            double eta_denom_base = is_binomial ? 0.0 : (d.hsgp_shared ? hsgp_f[i] : 0.0);
                            for (int p = 0; p < d.p_num; p++) eta_num_base += d.X_num[i * d.p_num + p] * beta_num[p];
                            if (!is_binomial)
                                for (int p = 0; p < d.p_denom; p++) eta_denom_base += d.X_denom[i * d.p_denom + p] * beta_denom[p];
                            if (d.has_temporal && phi_temporal_ptr) {
                                double temp_eff = compute_temporal_eta_obs(i, phi_temporal_ptr, d);
                                eta_num_base += temp_eff;
                                if (!is_binomial && d.temporal_shared) eta_denom_base += temp_eff;
                            }
                            double x_jt = d.tvc_X[i * d.tvc_n_terms + j];
                            double tvc_other = compute_tvc_eta_obs(i, tvc_w.data(), d) - x_jt * tvc_w[w_idx];
                            double en_c = eta_num_base + tvc_other + x_jt * tvc_w[w_idx];
                            double ed_c = eta_denom_base + (d.tvc_shared ? tvc_other + x_jt * tvc_w[w_idx] : 0.0);
                            double en_p = eta_num_base + tvc_other + x_jt * w_prop;
                            double ed_p = eta_denom_base + (d.tvc_shared ? tvc_other + x_jt * w_prop : 0.0);
                            ll_diff += obs_log_lik(i, en_p, ed_p, phi_num_val, phi_denom_val, d)
                                     - obs_log_lik(i, en_c, ed_c, phi_num_val, phi_denom_val, d);
                        }
                        if (std::log(runif(rng)) < ll_diff) { tvc_w[w_idx] = w_prop; tvc_accept++; }
                        tvc_total++;
                    }
                }
                // Conjugate tau update
                double qf = 0.0;
                for (int gg = 0; gg < d.tvc_n_groups; gg++)
                    for (int tt = 1; tt < d.tvc_n_times; tt++) {
                        double diff = tvc_w[(gg * d.tvc_n_terms + j) * d.tvc_n_times + tt]
                                    - tvc_w[(gg * d.tvc_n_terms + j) * d.tvc_n_times + tt - 1];
                        qf += diff * diff;
                    }
                int rank = (d.tvc_n_times - 1) * d.tvc_n_groups;
                std::gamma_distribution<double> gd_t(0.5 * rank, 1.0 / (0.5 * qf + 0.01));
                tvc_tau[j] = gd_t(rng);
            }
        }

        // ---- 6. Update temporal GMRF ----
        if (d.has_temporal) {
            double tau_t = std::exp(log_tau_temporal);
            for (int g = 0; g < n_temporal_groups; g++) {
                for (int t = 0; t < T_temporal; t++) {
                    int flat_idx = g * T_temporal + t;
                    int n_nb = 0; double nb_sum = 0.0;
                    if (t > 0) { nb_sum += phi_temporal[g * T_temporal + t - 1]; n_nb++; }
                    if (t < T_temporal - 1) { nb_sum += phi_temporal[g * T_temporal + t + 1]; n_nb++; }
                    double cond_prec = tau_t * std::max(n_nb, 1);
                    double cond_mean = (n_nb > 0) ? nb_sum / n_nb : 0.0;
                    double phi_t_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

                    double ll_diff = 0.0;
                    for (int i = 0; i < N; i++) {
                        if (d.temporal_time_idx[i] - 1 != t || d.temporal_group_idx[i] - 1 != g) continue;
                        double eta_num_base = hsgp_f[i];
                        double eta_denom_base = is_binomial ? 0.0 : (d.hsgp_shared ? hsgp_f[i] : 0.0);
                        for (int p = 0; p < d.p_num; p++) eta_num_base += d.X_num[i * d.p_num + p] * beta_num[p];
                        if (!is_binomial)
                            for (int p = 0; p < d.p_denom; p++) eta_denom_base += d.X_denom[i * d.p_denom + p] * beta_denom[p];
                        if (d.has_tvc && tvc_w_ptr) {
                            double tvc_eff = compute_tvc_eta_obs(i, tvc_w_ptr, d);
                            eta_num_base += tvc_eff;
                            if (!is_binomial && d.tvc_shared) eta_denom_base += tvc_eff;
                        }
                        double en_c = eta_num_base + phi_temporal[flat_idx];
                        double ed_c = eta_denom_base + (d.temporal_shared ? phi_temporal[flat_idx] : 0.0);
                        double en_p = eta_num_base + phi_t_prop;
                        double ed_p = eta_denom_base + (d.temporal_shared ? phi_t_prop : 0.0);
                        ll_diff += obs_log_lik(i, en_p, ed_p, phi_num_val, phi_denom_val, d)
                                 - obs_log_lik(i, en_c, ed_c, phi_num_val, phi_denom_val, d);
                    }
                    if (std::log(runif(rng)) < ll_diff) { phi_temporal[flat_idx] = phi_t_prop; temporal_accept++; }
                    temporal_total++;
                }
            }
            // Sum-to-zero + tau update
            for (int g = 0; g < n_temporal_groups; g++) {
                double t_mean = 0.0;
                for (int t = 0; t < T_temporal; t++) t_mean += phi_temporal[g * T_temporal + t];
                t_mean /= T_temporal;
                for (int t = 0; t < T_temporal; t++) phi_temporal[g * T_temporal + t] -= t_mean;
            }
            double qf_t = 0.0;
            for (int g = 0; g < n_temporal_groups; g++)
                for (int t = 1; t < T_temporal; t++) {
                    double diff = phi_temporal[g * T_temporal + t] - phi_temporal[g * T_temporal + t - 1];
                    qf_t += diff * diff;
                }
            int rank_t = (T_temporal - 1) * n_temporal_groups;
            if (0.5 * qf_t + 0.01 > 1e-10) {
                std::gamma_distribution<double> gd_t(0.5 * rank_t, 1.0 / (0.5 * qf_t + 0.01));
                double tau_new = std::clamp(gd_t(rng), 0.01, 1000.0);
                log_tau_temporal = std::log(tau_new);
            }
        }

        // ---- Adaptation ----
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            double target = 0.44;
            auto adapt = [&](int& acc, int& tot, double& scale) {
                if (tot > 0) {
                    double rate = (double)acc / tot;
                    if (rate > target + 0.05) scale *= 1.2;
                    else if (rate < target - 0.05) scale *= 0.8;
                    scale = std::clamp(scale, 0.001, 5.0);
                    acc = 0; tot = 0;
                }
            };
            adapt(hsgp_beta_accept, hsgp_beta_total, hsgp_beta_scale);
            adapt(hsgp_hyper_accept, hsgp_hyper_total, hsgp_hyper_scale);
            adapt(beta_accept, beta_total, beta_scale);
            adapt(disp_accept, disp_total, disp_scale);
        }

        // ---- Store draws ----
        if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
            int row = save_idx * n_params;
            int col = 0;
            for (int j = 0; j < p_num; j++) result.draws_flat[row + col++] = beta_num[j];
            for (int j = 0; j < p_denom; j++) result.draws_flat[row + col++] = beta_denom[j];
            if (has_phi_num) result.draws_flat[row + col++] = log_phi_num;
            if (has_phi_denom) result.draws_flat[row + col++] = log_phi_denom;
            result.draws_flat[row + col++] = log_sigma2;
            result.draws_flat[row + col++] = log_lengthscale;

            for (int j = 0; j < M; j++)
                result.hsgp_beta_draws_flat[save_idx * M + j] = hsgp_beta[j];
            for (int i = 0; i < N; i++)
                result.hsgp_f_draws_flat[save_idx * N + i] = hsgp_f[i];

            if (d.has_tvc) {
                for (int k = 0; k < tvc_n_w; k++) result.tvc_w_draws_flat[save_idx * tvc_n_w + k] = tvc_w[k];
                for (int j = 0; j < d.tvc_n_terms; j++) result.tvc_tau_draws_flat[save_idx * d.tvc_n_terms + j] = tvc_tau[j];
            }
            if (d.has_temporal) {
                for (int k = 0; k < n_temporal_params; k++) result.temporal_draws_flat[save_idx * n_temporal_params + k] = phi_temporal[k];
            }
            save_idx++;
        }

        if (verbose && (iter + 1) % 200 == 0) {
            Rcpp::Rcout << "  Gibbs iter " << (iter + 1) << "/" << n_iter;
            if (iter < n_warmup) Rcpp::Rcout << " (warmup)";
            Rcpp::Rcout << std::endl;
        }
    }

    // Acceptance rates
    result.accept_hsgp_beta = hsgp_beta_total > 0 ? (double)hsgp_beta_accept / hsgp_beta_total : 0.0;
    result.accept_hsgp_hyper = hsgp_hyper_total > 0 ? (double)hsgp_hyper_accept / hsgp_hyper_total : 0.0;
    result.accept_beta = beta_total > 0 ? (double)beta_accept / beta_total : 0.0;
    result.accept_disp = disp_total > 0 ? (double)disp_accept / disp_total : 0.0;
    result.accept_tvc = tvc_total > 0 ? (double)tvc_accept / tvc_total : 0.0;
    result.accept_temporal = temporal_total > 0 ? (double)temporal_accept / temporal_total : 0.0;

    // Param names
    for (int j = 0; j < p_num; j++) result.param_names.push_back("beta_num[" + std::to_string(j + 1) + "]");
    for (int j = 0; j < p_denom; j++) result.param_names.push_back("beta_denom[" + std::to_string(j + 1) + "]");
    if (has_phi_num) result.param_names.push_back("log_phi_num");
    if (has_phi_denom) result.param_names.push_back("log_phi_denom");
    result.param_names.push_back("log_sigma2_hsgp");
    result.param_names.push_back("log_lengthscale_hsgp");

    return result;
}


} // namespace gibbs
