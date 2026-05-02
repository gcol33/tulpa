// gibbs_spatial_bym2.h
// Sliced from gibbs_spatial.h. BYM2 helpers and run_gibbs_bym2.

#pragma once

#include "gibbs_spatial_data.h"

namespace gibbs {

// =========================================================================
// BYM2 Gibbs sampler
// Riebler parameterization: spatial_effect = sigma*(sqrt(rho)*scale*phi + sqrt(1-rho)*theta)
// phi = ICAR structured, theta = iid unstructured
// =========================================================================

// Compute total spatial effect for site s under BYM2
inline double bym2_spatial_effect(int s, const double* phi, const double* theta,
                                   double sigma_total, double rho,
                                   double scale_factor) {
    double sigma_s = sigma_total * std::sqrt(rho);
    double sigma_u = sigma_total * std::sqrt(1.0 - rho);
    return sigma_s * scale_factor * phi[s] + sigma_u * theta[s];
}

// Site log-lik for BYM2 (spatial effect computed from phi + theta)
inline double site_log_lik_bym2(int s, const double* phi, const double* theta,
                                 double sigma_total, double rho, double scale_factor,
                                 const double* beta_num, const double* beta_denom,
                                 double disp_num, double disp_denom,
                                 const GibbsData& d) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);
    double spatial = bym2_spatial_effect(s, phi, theta, sigma_total, rho, scale_factor);

    for (int idx = d.site_obs_ptr[s]; idx < d.site_obs_ptr[s + 1]; idx++) {
        int i = d.site_obs_idx[idx];
        double eta_num = spatial;
        double eta_denom = is_binomial ? 0.0 : spatial;

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        ll += obs_log_lik(i, eta_num, eta_denom, disp_num, disp_denom, d);
    }
    return ll;
}

// Full log-lik for BYM2
inline double full_log_lik_bym2(const double* phi, const double* theta,
                                 double sigma_total, double rho, double scale_factor,
                                 const double* beta_num, const double* beta_denom,
                                 double disp_num, double disp_denom,
                                 const GibbsData& d) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);

    for (int i = 0; i < d.N; i++) {
        int s = d.spatial_group[i];
        double spatial = bym2_spatial_effect(s, phi, theta, sigma_total, rho, scale_factor);
        double eta_num = spatial;
        double eta_denom = is_binomial ? 0.0 : spatial;

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        ll += obs_log_lik(i, eta_num, eta_denom, disp_num, disp_denom, d);
    }
    return ll;
}

inline GibbsResult run_gibbs_bym2(
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

    int S = d.S;
    double scale_factor = d.bym2_scale;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);
    bool has_disp_num = (d.family == GibbsFamily::NEGBIN_NEGBIN ||
                         d.family == GibbsFamily::NEGBIN_GAMMA ||
                         d.family == GibbsFamily::GAMMA_GAMMA ||
                         d.family == GibbsFamily::LOGNORMAL ||
                         d.family == GibbsFamily::BETA_BINOMIAL);
    bool has_disp_denom = !is_binomial;

    // Parameter dimensions
    int p_num = d.p_num;
    int p_denom = is_binomial ? 0 : d.p_denom;
    int n_hyper = 2;  // log_sigma_total, logit_rho
    if (has_disp_num) n_hyper++;
    if (has_disp_denom) n_hyper++;
    int n_params = p_num + p_denom + n_hyper;

    // Initialize
    std::vector<double> beta_num(p_num, 0.0);
    std::vector<double> beta_denom(p_denom, 0.0);
    std::vector<double> phi(S, 0.0);    // ICAR structured
    std::vector<double> theta(S, 0.0);  // iid unstructured
    double log_sigma_total = std::log(1.0);
    double logit_rho = 0.0;  // rho = 0.5
    double log_disp_num = std::log(1.0);
    double log_disp_denom = std::log(5.0);

    if (p_num > 0) beta_num[0] = 1.0;
    if (p_denom > 0) beta_denom[0] = 1.0;

    // Proposal scales
    double beta_scale = 0.1;
    double disp_scale = 0.1;
    double sigma_scale = 0.1;
    double rho_scale = 0.3;

    // Acceptance tracking
    std::vector<int> phi_accept(S, 0), phi_total(S, 0);
    std::vector<int> theta_accept(S, 0), theta_total(S, 0);
    int beta_accept_cnt = 0, beta_total_cnt = 0;
    int disp_accept_cnt = 0, disp_total_cnt = 0;
    int sigma_accept_cnt = 0, sigma_total_cnt = 0;
    int rho_accept_cnt = 0, rho_total_cnt = 0;

    // Output
    int n_save = (n_iter - n_warmup) / thin;
    GibbsResult result;
    result.n_params = n_params;
    result.n_save = n_save;
    result.S = S;
    result.draws_flat.resize(n_save * n_params);
    result.phi_draws_flat.resize(n_save * S);  // Store total spatial effect
    result.accept_phi.resize(S, 0.0);

    int save_idx = 0;

    // ---- Main Gibbs loop ----
    for (int iter = 0; iter < n_iter; iter++) {
        double sigma_total = std::exp(log_sigma_total);
        double rho = 1.0 / (1.0 + std::exp(-logit_rho));
        double disp_num_val = has_disp_num ? std::exp(log_disp_num) : 1.0;
        double disp_denom_val = has_disp_denom ? std::exp(log_disp_denom) : 1.0;

        // Implied ICAR precision for the structured component
        double sigma_s = sigma_total * std::sqrt(rho);
        double tau_phi = (sigma_s * scale_factor > 1e-10) ?
                         1.0 / (sigma_s * sigma_s * scale_factor * scale_factor) : 100.0;

        // ---- 1. Update phi (ICAR structured) via MH ----
        for (int s = 0; s < S; s++) {
            auto [cond_mean, cond_prec] = icar_conditional(
                s, phi.data(), tau_phi, d.adj_row_ptr, d.adj_col_idx, d.n_neighbors);

            double phi_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

            // Need full site log-lik with proposed phi change
            double old_phi_s = phi[s];
            double ll_curr = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            phi[s] = phi_prop;
            double ll_prop = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            if (std::log(runif(rng)) < ll_prop - ll_curr) {
                phi_accept[s]++;
            } else {
                phi[s] = old_phi_s;  // Reject
            }
            phi_total[s]++;
        }

        // Soft sum-to-zero for phi
        double phi_mean = 0.0;
        for (int s = 0; s < S; s++) phi_mean += phi[s];
        phi_mean /= S;
        for (int s = 0; s < S; s++) phi[s] -= phi_mean;

        // ---- 2. Update theta (iid unstructured) via MH ----
        // Prior: theta_s ~ N(0, 1)
        for (int s = 0; s < S; s++) {
            double theta_prop = rnorm(rng);  // Propose from prior N(0,1)

            double old_theta_s = theta[s];
            double ll_curr = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            theta[s] = theta_prop;
            double ll_prop = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            if (std::log(runif(rng)) < ll_prop - ll_curr) {
                theta_accept[s]++;
            } else {
                theta[s] = old_theta_s;
            }
            theta_total[s]++;
        }

        // ---- 3. Update sigma_total via MH on log scale ----
        {
            double log_sigma_prop = log_sigma_total + sigma_scale * rnorm(rng);
            double sigma_prop = std::exp(log_sigma_prop);

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_prop, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            // PC prior on sigma_total: P(sigma > 1) = 0.01
            double pc_lambda = 4.605;
            double lp_curr = -pc_lambda * sigma_total + log_sigma_total;  // PC + Jacobian
            double lp_prop = -pc_lambda * sigma_prop + log_sigma_prop;

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_sigma_total = log_sigma_prop;
                sigma_accept_cnt++;
            }
            sigma_total_cnt++;
        }

        // ---- 4. Update rho via MH on logit scale ----
        {
            double logit_rho_prop = logit_rho + rho_scale * rnorm(rng);
            double rho_prop = 1.0 / (1.0 + std::exp(-logit_rho_prop));

            sigma_total = std::exp(log_sigma_total);  // Refresh

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho_prop, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            // Beta(0.5, 0.5) prior on rho + logit Jacobian
            // log p(rho) = -0.5*log(rho) - 0.5*log(1-rho) + const
            // Jacobian of logit: log(rho*(1-rho))
            double lp_curr = -0.5 * std::log(rho) - 0.5 * std::log(1.0 - rho)
                            + std::log(rho) + std::log(1.0 - rho);  // = 0.5*log(rho*(1-rho))
            double lp_prop = -0.5 * std::log(rho_prop) - 0.5 * std::log(1.0 - rho_prop)
                            + std::log(rho_prop) + std::log(1.0 - rho_prop);

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                logit_rho = logit_rho_prop;
                rho_accept_cnt++;
            }
            rho_total_cnt++;
        }

        // ---- 5. Update beta via block MH ----
        {
            sigma_total = std::exp(log_sigma_total);
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            std::vector<double> beta_num_prop(beta_num);
            std::vector<double> beta_denom_prop(beta_denom);
            for (int j = 0; j < p_num; j++)
                beta_num_prop[j] += beta_scale * rnorm(rng);
            for (int j = 0; j < p_denom; j++)
                beta_denom_prop[j] += beta_scale * rnorm(rng);

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num_prop.data(), beta_denom_prop.data(),
                                                disp_num_val, disp_denom_val, d);

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
                beta_accept_cnt++;
            }
            beta_total_cnt++;
        }

        // ---- 6. Update dispersion via MH ----
        if (has_disp_num || has_disp_denom) {
            sigma_total = std::exp(log_sigma_total);
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            double log_dn_prop = log_disp_num;
            double log_dd_prop = log_disp_denom;
            if (has_disp_num) log_dn_prop += disp_scale * rnorm(rng);
            if (has_disp_denom) log_dd_prop += disp_scale * rnorm(rng);

            double dn_curr = has_disp_num ? std::exp(log_disp_num) : 1.0;
            double dd_curr = has_disp_denom ? std::exp(log_disp_denom) : 1.0;
            double dn_prop = has_disp_num ? std::exp(log_dn_prop) : 1.0;
            double dd_prop = has_disp_denom ? std::exp(log_dd_prop) : 1.0;

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                dn_curr, dd_curr, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                dn_prop, dd_prop, d);

            double lp_curr = 0.0, lp_prop = 0.0;
            if (has_disp_num) {
                lp_curr += std::log(dn_curr) - 0.5 * dn_curr;
                lp_prop += std::log(dn_prop) - 0.5 * dn_prop;
            }
            if (has_disp_denom) {
                lp_curr += std::log(dd_curr) - 0.5 * dd_curr;
                lp_prop += std::log(dd_prop) - 0.5 * dd_prop;
            }

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_disp_num = log_dn_prop;
                log_disp_denom = log_dd_prop;
                disp_accept_cnt++;
            }
            disp_total_cnt++;
        }

        // ---- Adaptation during warmup ----
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            double target = 0.44;

            auto adapt = [&](double& scale, int& acc, int& tot) {
                if (tot == 0) return;
                double rate = (double)acc / tot;
                if (rate > target + 0.05) scale *= 1.2;
                else if (rate < target - 0.05) scale *= 0.8;
                scale = std::max(0.001, std::min(scale, 5.0));
                acc = 0; tot = 0;
            };

            adapt(beta_scale, beta_accept_cnt, beta_total_cnt);
            adapt(sigma_scale, sigma_accept_cnt, sigma_total_cnt);
            adapt(rho_scale, rho_accept_cnt, rho_total_cnt);
            if (has_disp_num || has_disp_denom)
                adapt(disp_scale, disp_accept_cnt, disp_total_cnt);
        }

        // ---- Store draws ----
        if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
            sigma_total = std::exp(log_sigma_total);
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            int row = save_idx * n_params;
            int col = 0;
            for (int j = 0; j < p_num; j++) result.draws_flat[row + col++] = beta_num[j];
            for (int j = 0; j < p_denom; j++) result.draws_flat[row + col++] = beta_denom[j];
            if (has_disp_num) result.draws_flat[row + col++] = log_disp_num;
            if (has_disp_denom) result.draws_flat[row + col++] = log_disp_denom;
            result.draws_flat[row + col++] = log_sigma_total;
            result.draws_flat[row + col++] = logit_rho;

            // Store total spatial effect per site
            for (int s = 0; s < S; s++) {
                result.phi_draws_flat[save_idx * S + s] =
                    bym2_spatial_effect(s, phi.data(), theta.data(),
                                        sigma_total, rho, scale_factor);
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
    for (int s = 0; s < S; s++) {
        result.accept_phi[s] = phi_total[s] > 0 ?
            (double)phi_accept[s] / phi_total[s] : 0.0;
    }
    result.accept_beta = beta_total_cnt > 0 ? (double)beta_accept_cnt / beta_total_cnt : 0.0;
    result.accept_disp = disp_total_cnt > 0 ? (double)disp_accept_cnt / disp_total_cnt : 0.0;

    // Param names
    for (int j = 0; j < p_num; j++)
        result.param_names.push_back("beta_num[" + std::to_string(j + 1) + "]");
    for (int j = 0; j < p_denom; j++)
        result.param_names.push_back("beta_denom[" + std::to_string(j + 1) + "]");
    if (has_disp_num) result.param_names.push_back("log_disp_num");
    if (has_disp_denom) result.param_names.push_back("log_disp_denom");
    result.param_names.push_back("log_sigma_total");
    result.param_names.push_back("logit_rho");

    return result;
}


} // namespace gibbs
