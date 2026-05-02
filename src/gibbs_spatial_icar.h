// gibbs_spatial_icar.h
// Sliced from gibbs_spatial.h. ICAR conditional helpers and run_gibbs_icar.

#pragma once

#include "gibbs_spatial_data.h"

namespace gibbs {

// =========================================================================
// ICAR prior log-density (up to normalization constant)
// =========================================================================

inline double log_prior_icar(const double* phi, double tau, int S,
                             const int* adj_row_ptr, const int* adj_col_idx,
                             const int* n_neighbors) {
    // -tau/2 * phi' Q phi + (S-1)/2 * log(tau)
    double quad = 0.0;
    for (int s = 0; s < S; s++) {
        quad += n_neighbors[s] * phi[s] * phi[s];
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            quad -= phi[s] * phi[adj_col_idx[k]];
        }
    }
    return -0.5 * tau * quad + 0.5 * (S - 1) * std::log(tau);
}

// ICAR conditional prior for phi_s | phi_{-s}
// Returns (mean, precision) of the full conditional Gaussian
inline std::pair<double, double> icar_conditional(int s, const double* phi,
                                                   double tau,
                                                   const int* adj_row_ptr,
                                                   const int* adj_col_idx,
                                                   const int* n_neighbors) {
    int n_s = n_neighbors[s];
    if (n_s == 0) return {0.0, 0.001};  // Isolated node

    double neighbor_sum = 0.0;
    for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        neighbor_sum += phi[adj_col_idx[k]];
    }
    double prec = tau * n_s;
    double mean = neighbor_sum / n_s;
    return {mean, prec};
}

inline GibbsResult run_gibbs_icar(
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
    std::gamma_distribution<double> rgamma_unit(1.0, 1.0);

    int S = d.S;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);
    bool has_phi_num = (d.family == GibbsFamily::NEGBIN_NEGBIN ||
                        d.family == GibbsFamily::NEGBIN_GAMMA ||
                        d.family == GibbsFamily::GAMMA_GAMMA ||
                        d.family == GibbsFamily::LOGNORMAL ||
                        d.family == GibbsFamily::BETA_BINOMIAL);
    bool has_phi_denom = !is_binomial;

    // Parameter dimensions
    int p_num = d.p_num;
    int p_denom = is_binomial ? 0 : d.p_denom;
    int n_hyper = 1;  // log_tau
    if (has_phi_num) n_hyper++;   // log_phi_num
    if (has_phi_denom) n_hyper++; // log_phi_denom
    int n_params = p_num + p_denom + n_hyper;

    // Initialize parameters
    std::vector<double> beta_num(p_num, 0.0);
    std::vector<double> beta_denom(p_denom, 0.0);
    std::vector<double> phi(S, 0.0);
    double log_tau = std::log(1.0);
    double log_phi_num = std::log(1.0);   // NB size
    double log_phi_denom = std::log(5.0); // Gamma shape or NB size

    // Initialize beta from data (rough OLS-like)
    if (p_num > 0) beta_num[0] = 1.0;
    if (p_denom > 0) beta_denom[0] = 1.0;

    // Adaptation: proposal scales (tuned during warmup)
    double beta_scale = 0.1;
    double disp_scale = 0.1;
    std::vector<double> phi_scale(S, 0.5);

    // Acceptance tracking
    std::vector<int> phi_accept(S, 0);
    std::vector<int> phi_total(S, 0);
    int beta_accept = 0, beta_total = 0;
    int disp_accept = 0, disp_total = 0;

    // PC prior for tau: P(sigma > 1) = 0.01 => lambda = -log(0.01)/1 = 4.605
    double pc_lambda = 4.605;

    // TVC initialization
    int tvc_n_w = d.has_tvc ? (d.tvc_n_groups * d.tvc_n_terms * d.tvc_n_times) : 0;
    std::vector<double> tvc_w(tvc_n_w, 0.0);
    std::vector<double> tvc_tau(d.has_tvc ? d.tvc_n_terms : 0, 1.0);
    std::vector<double> tvc_rho(d.has_tvc ? d.tvc_n_terms : 0, 0.5);
    double tvc_w_scale = 0.2;  // MH proposal scale for TVC coefficients
    int tvc_accept = 0, tvc_total = 0;
    const double* tvc_w_ptr = d.has_tvc ? tvc_w.data() : nullptr;

    // Temporal GMRF initialization
    int T_temporal = d.has_temporal ? d.temporal_n_times : 0;
    int n_temporal_groups = d.has_temporal ? d.temporal_n_groups : 0;
    int n_temporal_params = T_temporal * n_temporal_groups;
    std::vector<double> phi_temporal(n_temporal_params, 0.0);
    double log_tau_temporal = std::log(1.0);
    double rho_ar1 = 0.5;
    int temporal_accept = 0, temporal_total = 0;
    const double* phi_temporal_ptr = d.has_temporal ? phi_temporal.data() : nullptr;

    // Output storage
    int n_save = (n_iter - n_warmup) / thin;
    GibbsResult result;
    result.n_params = n_params;
    result.n_save = n_save;
    result.S = S;
    result.draws_flat.resize(n_save * n_params);
    result.phi_draws_flat.resize(n_save * S);
    result.accept_phi.resize(S, 0.0);
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
        double tau = std::exp(log_tau);
        double phi_num_val = has_phi_num ? std::exp(log_phi_num) : 1.0;
        double phi_denom_val = has_phi_denom ? std::exp(log_phi_denom) : 1.0;

        // ---- 1. Update phi (spatial effects) via MH ----
        for (int s = 0; s < S; s++) {
            // ICAR conditional prior: phi_s | phi_{-s} ~ N(mean, 1/(tau*n_s))
            auto [cond_mean, cond_prec] = icar_conditional(
                s, phi.data(), tau, d.adj_row_ptr, d.adj_col_idx, d.n_neighbors);

            // Propose from ICAR conditional (independence MH)
            double phi_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

            // Log-likelihood ratio (prior cancels with proposal)
            double ll_curr = site_log_lik(s, phi[s], beta_num.data(), beta_denom.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);
            double ll_prop = site_log_lik(s, phi_prop, beta_num.data(), beta_denom.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);

            // Accept/reject (proposal = prior conditional, so MH ratio = lik ratio)
            if (std::log(runif(rng)) < ll_prop - ll_curr) {
                phi[s] = phi_prop;
                phi_accept[s]++;
            }
            phi_total[s]++;
        }

        // Soft sum-to-zero centering
        double phi_mean = 0.0;
        for (int s = 0; s < S; s++) phi_mean += phi[s];
        phi_mean /= S;
        for (int s = 0; s < S; s++) phi[s] -= phi_mean;

        // ---- 2. Update tau (ICAR precision) via conjugate Gamma ----
        {
            // phi' Q phi
            double quad = 0.0;
            for (int s = 0; s < S; s++) {
                quad += d.n_neighbors[s] * phi[s] * phi[s];
                for (int k = d.adj_row_ptr[s]; k < d.adj_row_ptr[s + 1]; k++) {
                    quad -= phi[s] * phi[d.adj_col_idx[k]];
                }
            }
            // Gamma posterior: shape = (S-1)/2, rate = quad/2
            // With PC prior, use MH instead
            double shape = 0.5 * (S - 1);
            double rate = 0.5 * quad;

            // Simple Gamma draw (conjugate with flat prior on tau)
            if (rate > 1e-10) {
                std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
                tau = gamma_dist(rng);
                // Clamp to reasonable range
                tau = std::max(tau, 0.01);
                tau = std::min(tau, 1000.0);
                log_tau = std::log(tau);
            }
        }

        // ---- 3. Update beta via block random-walk MH ----
        {
            // Propose: beta* = beta + scale * N(0, I)
            std::vector<double> beta_num_prop(beta_num);
            std::vector<double> beta_denom_prop(beta_denom);

            for (int j = 0; j < p_num; j++)
                beta_num_prop[j] += beta_scale * rnorm(rng);
            for (int j = 0; j < p_denom; j++)
                beta_denom_prop[j] += beta_scale * rnorm(rng);

            double ll_curr = full_log_lik(phi.data(), beta_num.data(), beta_denom.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);
            double ll_prop = full_log_lik(phi.data(), beta_num_prop.data(), beta_denom_prop.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr, phi_temporal_ptr);

            // Normal prior on beta: N(0, 10^2)
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

        // ---- 4. Update dispersion params via MH on log scale ----
        if (has_phi_num || has_phi_denom) {
            double log_phi_num_prop = log_phi_num;
            double log_phi_denom_prop = log_phi_denom;

            if (has_phi_num)
                log_phi_num_prop += disp_scale * rnorm(rng);
            if (has_phi_denom)
                log_phi_denom_prop += disp_scale * rnorm(rng);

            double pn_curr = has_phi_num ? std::exp(log_phi_num) : 1.0;
            double pd_curr = has_phi_denom ? std::exp(log_phi_denom) : 1.0;
            double pn_prop = has_phi_num ? std::exp(log_phi_num_prop) : 1.0;
            double pd_prop = has_phi_denom ? std::exp(log_phi_denom_prop) : 1.0;

            double ll_curr = full_log_lik(phi.data(), beta_num.data(), beta_denom.data(),
                                          pn_curr, pd_curr, d, tvc_w_ptr, phi_temporal_ptr);
            double ll_prop = full_log_lik(phi.data(), beta_num.data(), beta_denom.data(),
                                          pn_prop, pd_prop, d, tvc_w_ptr, phi_temporal_ptr);

            // PC prior on dispersion: Gamma(2, 0.5) on the parameter
            double lp_curr = 0.0, lp_prop = 0.0;
            if (has_phi_num) {
                lp_curr += std::log(pn_curr) - 0.5 * pn_curr;  // Gamma(2,0.5) + Jacobian
                lp_prop += std::log(pn_prop) - 0.5 * pn_prop;
            }
            if (has_phi_denom) {
                lp_curr += std::log(pd_curr) - 0.5 * pd_curr;
                lp_prop += std::log(pd_prop) - 0.5 * pd_prop;
            }

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_phi_num = log_phi_num_prop;
                log_phi_denom = log_phi_denom_prop;
                disp_accept++;
            }
            disp_total++;
        }

        // ---- 5. Update TVC coefficients via univariate MH with RW1 conditional proposal ----
        if (d.has_tvc) {
            for (int j = 0; j < d.tvc_n_terms; j++) {
                for (int g = 0; g < d.tvc_n_groups; g++) {
                    for (int t = 0; t < d.tvc_n_times; t++) {
                        int w_idx = (g * d.tvc_n_terms + j) * d.tvc_n_times + t;

                        // RW1 conditional prior: N(neighbor_mean, 1/(tau_j * n_neighbors))
                        int n_nb = 0;
                        double nb_sum = 0.0;
                        if (t > 0) { nb_sum += tvc_w[(g * d.tvc_n_terms + j) * d.tvc_n_times + t - 1]; n_nb++; }
                        if (t < d.tvc_n_times - 1) { nb_sum += tvc_w[(g * d.tvc_n_terms + j) * d.tvc_n_times + t + 1]; n_nb++; }
                        double cond_prec = tvc_tau[j] * std::max(n_nb, 1);
                        double cond_mean = (n_nb > 0) ? nb_sum / n_nb : 0.0;

                        // Propose from RW1 conditional
                        double w_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

                        // Compute likelihood ratio at obs with this time point
                        double ll_diff = 0.0;
                        for (int idx = d.time_obs_ptr[t]; idx < d.time_obs_ptr[t + 1]; idx++) {
                            int i = d.time_obs_idx[idx];
                            int s = d.spatial_group[i];
                            double eta_num_base = phi[s];
                            double eta_denom_base = is_binomial ? 0.0 : phi[s];
                            for (int p = 0; p < d.p_num; p++)
                                eta_num_base += d.X_num[i * d.p_num + p] * beta_num[p];
                            if (!is_binomial)
                                for (int p = 0; p < d.p_denom; p++)
                                    eta_denom_base += d.X_denom[i * d.p_denom + p] * beta_denom[p];
                            // Temporal contribution
                            if (d.has_temporal && phi_temporal_ptr != nullptr) {
                                double temp_eff = compute_temporal_eta_obs(i, phi_temporal_ptr, d);
                                eta_num_base += temp_eff;
                                if (!is_binomial && d.temporal_shared) eta_denom_base += temp_eff;
                            }

                            // Compute TVC eta with current vs proposed
                            double x_jt = d.tvc_X[i * d.tvc_n_terms + j];
                            double tvc_other = compute_tvc_eta_obs(i, tvc_w.data(), d) - x_jt * tvc_w[w_idx];
                            double tvc_curr_eta = tvc_other + x_jt * tvc_w[w_idx];
                            double tvc_prop_eta = tvc_other + x_jt * w_prop;

                            double en_c = eta_num_base + tvc_curr_eta;
                            double ed_c = eta_denom_base + (d.tvc_shared ? tvc_curr_eta : 0.0);
                            double en_p = eta_num_base + tvc_prop_eta;
                            double ed_p = eta_denom_base + (d.tvc_shared ? tvc_prop_eta : 0.0);

                            ll_diff += obs_log_lik(i, en_p, ed_p, phi_num_val, phi_denom_val, d)
                                     - obs_log_lik(i, en_c, ed_c, phi_num_val, phi_denom_val, d);
                        }

                        if (std::log(runif(rng)) < ll_diff) {
                            tvc_w[w_idx] = w_prop;
                            tvc_accept++;
                        }
                        tvc_total++;
                    }
                }

                // Update tau[j] via conjugate Gamma (RW1)
                double qf = 0.0;
                for (int gg = 0; gg < d.tvc_n_groups; gg++) {
                    for (int tt = 1; tt < d.tvc_n_times; tt++) {
                        double diff = tvc_w[(gg * d.tvc_n_terms + j) * d.tvc_n_times + tt]
                                    - tvc_w[(gg * d.tvc_n_terms + j) * d.tvc_n_times + tt - 1];
                        qf += diff * diff;
                    }
                }
                int rank = (d.tvc_n_times - 1) * d.tvc_n_groups;
                double shape_t = 0.5 * rank;
                double rate_t = 0.5 * qf + 0.01;
                std::gamma_distribution<double> gd_t(shape_t, 1.0 / rate_t);
                tvc_tau[j] = gd_t(rng);
            }
        }

        // ---- 6. Update temporal GMRF effects (RW1/AR1) via component-wise MH ----
        if (d.has_temporal) {
            double tau_t = std::exp(log_tau_temporal);
            for (int g = 0; g < n_temporal_groups; g++) {
                for (int t = 0; t < T_temporal; t++) {
                    int flat_idx = g * T_temporal + t;

                    // RW1 conditional prior: N(neighbor_mean, 1/(tau*n_neighbors))
                    int n_nb = 0;
                    double nb_sum = 0.0;
                    if (t > 0) { nb_sum += phi_temporal[g * T_temporal + t - 1]; n_nb++; }
                    if (t < T_temporal - 1) { nb_sum += phi_temporal[g * T_temporal + t + 1]; n_nb++; }
                    double cond_prec = tau_t * std::max(n_nb, 1);
                    double cond_mean = (n_nb > 0) ? nb_sum / n_nb : 0.0;

                    // Propose from conditional
                    double phi_t_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

                    // Compute likelihood ratio over ALL observations at this time point
                    double ll_diff = 0.0;
                    for (int i = 0; i < d.N; i++) {
                        int obs_t = d.temporal_time_idx[i] - 1;
                        int obs_g = d.temporal_group_idx[i] - 1;
                        if (obs_t != t || obs_g != g) continue;

                        int s = d.spatial_group[i];
                        double eta_num_base = phi[s];
                        double eta_denom_base = is_binomial ? 0.0 : phi[s];
                        for (int p = 0; p < d.p_num; p++)
                            eta_num_base += d.X_num[i * d.p_num + p] * beta_num[p];
                        if (!is_binomial)
                            for (int p = 0; p < d.p_denom; p++)
                                eta_denom_base += d.X_denom[i * d.p_denom + p] * beta_denom[p];
                        if (tvc_w_ptr != nullptr && d.has_tvc) {
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

                    if (std::log(runif(rng)) < ll_diff) {
                        phi_temporal[flat_idx] = phi_t_prop;
                        temporal_accept++;
                    }
                    temporal_total++;
                }
            }

            // Soft sum-to-zero for temporal effects per group
            for (int g = 0; g < n_temporal_groups; g++) {
                double t_mean = 0.0;
                for (int t = 0; t < T_temporal; t++) t_mean += phi_temporal[g * T_temporal + t];
                t_mean /= T_temporal;
                for (int t = 0; t < T_temporal; t++) phi_temporal[g * T_temporal + t] -= t_mean;
            }

            // Update tau_temporal via conjugate Gamma (RW1 quadratic form)
            double qf_t = 0.0;
            for (int g = 0; g < n_temporal_groups; g++) {
                for (int t = 1; t < T_temporal; t++) {
                    double diff = phi_temporal[g * T_temporal + t] - phi_temporal[g * T_temporal + t - 1];
                    qf_t += diff * diff;
                }
            }
            int rank_t = (T_temporal - 1) * n_temporal_groups;
            double shape_t = 0.5 * rank_t;
            double rate_t = 0.5 * qf_t + 0.01;
            if (rate_t > 1e-10) {
                std::gamma_distribution<double> gd_t(shape_t, 1.0 / rate_t);
                double tau_t_new = gd_t(rng);
                tau_t_new = std::max(tau_t_new, 0.01);
                tau_t_new = std::min(tau_t_new, 1000.0);
                log_tau_temporal = std::log(tau_t_new);
            }
        }

        // ---- Adaptation during warmup ----
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            double target_rate = 0.44;  // Optimal for univariate

            // Adapt beta scale
            double beta_rate = (double)beta_accept / beta_total;
            if (beta_rate > target_rate + 0.05) beta_scale *= 1.2;
            else if (beta_rate < target_rate - 0.05) beta_scale *= 0.8;
            beta_scale = std::max(0.001, std::min(beta_scale, 5.0));
            beta_accept = 0; beta_total = 0;

            // Adapt dispersion scale
            if (has_phi_num || has_phi_denom) {
                double disp_rate = (double)disp_accept / disp_total;
                if (disp_rate > target_rate + 0.05) disp_scale *= 1.2;
                else if (disp_rate < target_rate - 0.05) disp_scale *= 0.8;
                disp_scale = std::max(0.001, std::min(disp_scale, 5.0));
                disp_accept = 0; disp_total = 0;
            }
        }

        // ---- Store draws after warmup ----
        if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
            int row = save_idx * n_params;
            int col = 0;
            for (int j = 0; j < p_num; j++) result.draws_flat[row + col++] = beta_num[j];
            for (int j = 0; j < p_denom; j++) result.draws_flat[row + col++] = beta_denom[j];
            if (has_phi_num) result.draws_flat[row + col++] = log_phi_num;
            if (has_phi_denom) result.draws_flat[row + col++] = log_phi_denom;
            result.draws_flat[row + col++] = log_tau;

            // Store phi
            for (int s = 0; s < S; s++)
                result.phi_draws_flat[save_idx * S + s] = phi[s];

            // Store TVC draws
            if (d.has_tvc) {
                for (int k = 0; k < tvc_n_w; k++)
                    result.tvc_w_draws_flat[save_idx * tvc_n_w + k] = tvc_w[k];
                for (int j = 0; j < d.tvc_n_terms; j++)
                    result.tvc_tau_draws_flat[save_idx * d.tvc_n_terms + j] = tvc_tau[j];
            }

            // Store temporal draws
            if (d.has_temporal) {
                for (int k = 0; k < n_temporal_params; k++)
                    result.temporal_draws_flat[save_idx * n_temporal_params + k] = phi_temporal[k];
            }

            save_idx++;
        }

        // Progress
        if (verbose && (iter + 1) % 200 == 0) {
            Rcpp::Rcout << "  Gibbs iter " << (iter + 1) << "/" << n_iter;
            if (iter < n_warmup) Rcpp::Rcout << " (warmup)";
            Rcpp::Rcout << std::endl;
        }
    }

    // Compute acceptance rates
    for (int s = 0; s < S; s++) {
        result.accept_phi[s] = phi_total[s] > 0 ?
            (double)phi_accept[s] / phi_total[s] : 0.0;
    }
    result.accept_beta = beta_total > 0 ? (double)beta_accept / beta_total : 0.0;
    result.accept_disp = disp_total > 0 ? (double)disp_accept / disp_total : 0.0;
    result.accept_tvc = tvc_total > 0 ? (double)tvc_accept / tvc_total : 0.0;
    result.accept_temporal = temporal_total > 0 ? (double)temporal_accept / temporal_total : 0.0;

    // Build parameter names
    for (int j = 0; j < p_num; j++)
        result.param_names.push_back("beta_num[" + std::to_string(j + 1) + "]");
    for (int j = 0; j < p_denom; j++)
        result.param_names.push_back("beta_denom[" + std::to_string(j + 1) + "]");
    if (has_phi_num) result.param_names.push_back("log_phi_num");
    if (has_phi_denom) result.param_names.push_back("log_phi_denom");
    result.param_names.push_back("log_tau");

    return result;
}


} // namespace gibbs
