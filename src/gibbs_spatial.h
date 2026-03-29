// gibbs_spatial.h
// Component-wise Gibbs sampler for ICAR/BYM2 spatial models
// Designed for large S where HMC struggles with dimensionality
//
// Update scheme:
//   1. phi_s | phi_{-s}, beta, tau, y  (univariate MH per site)
//   2. tau | phi                        (conjugate Gamma for ICAR)
//   3. beta | phi, y                    (block MH with Gaussian proposal)
//   4. dispersion | rest                (univariate MH on log scale)
//   For BYM2: sigma, rho, phi, theta all updated

#pragma once

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "hmc_hsgp.h"  // For spectral_density_se()

namespace gibbs {

// =========================================================================
// Site-level log-likelihood (sum over observations at a site)
// =========================================================================

enum class GibbsFamily { POISSON_GAMMA, NEGBIN_NEGBIN, BINOMIAL, NEGBIN_GAMMA,
                         GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL };

struct GibbsData {
    int N;                              // Total observations
    int S;                              // Number of spatial units
    int p_num;                          // Numerator predictors
    int p_denom;                        // Denominator predictors
    GibbsFamily family;

    const double* X_num;                // N x p_num (row-major)
    const double* X_denom;              // N x p_denom (row-major)
    const int* y_num;                   // Integer numerator response
    const double* y_num_cont;           // Continuous numerator (GG, LN)
    const int* y_denom;                 // Integer denominator response (NB, Binomial)
    const double* y_denom_cont;         // Continuous denominator (PG, NB-Gamma, GG, LN)

    const int* spatial_group;           // Maps obs -> site (0-based)

    // CSR obs-to-site mapping (built once)
    std::vector<int> site_obs_ptr;      // site_obs_ptr[s] = start index
    std::vector<int> site_obs_idx;      // observation indices for each site

    // Adjacency (CSR format)
    const int* adj_row_ptr;
    const int* adj_col_idx;
    const int* n_neighbors;

    // BYM2
    bool is_bym2;
    double bym2_scale;

    // TVC (Temporally-Varying Coefficients)
    bool has_tvc = false;
    int tvc_n_times = 0;
    int tvc_n_groups = 1;
    int tvc_n_terms = 0;
    std::vector<int> tvc_time_index;    // obs -> time (1-based)
    std::vector<int> tvc_group_index;   // obs -> group (1-based)
    std::vector<double> tvc_X;          // n_obs x n_terms (row-major)
    bool tvc_shared = true;
    int tvc_structure = 0;              // 0=RW1, 1=RW2, 2=AR1

    // Time-indexed obs mapping for TVC updates (built once)
    // time_obs_ptr[t] = start index in time_obs_idx for time t
    std::vector<int> time_obs_ptr;
    std::vector<int> time_obs_idx;

    // Temporal GMRF (RW1/AR1)
    bool has_temporal = false;
    int temporal_type = 0;                  // 0=RW1, 1=AR1
    int temporal_n_times = 0;
    int temporal_n_groups = 1;
    std::vector<int> temporal_time_idx;     // obs -> time (1-based)
    std::vector<int> temporal_group_idx;    // obs -> group (1-based)
    bool temporal_shared = true;

    // HSGP spatial (alternative to ICAR/BYM2)
    bool is_hsgp = false;
    int hsgp_m_total = 0;                   // m^2 basis functions
    std::vector<double> hsgp_Phi_flat;      // N x m_total (row-major)
    std::vector<double> hsgp_eigenvalues;   // m_total eigenvalues
    bool hsgp_shared = true;
};

// Build CSR site->obs mapping
inline void build_site_obs_map(GibbsData& d) {
    d.site_obs_ptr.assign(d.S + 1, 0);
    for (int i = 0; i < d.N; i++) {
        d.site_obs_ptr[d.spatial_group[i] + 1]++;
    }
    for (int s = 0; s < d.S; s++) {
        d.site_obs_ptr[s + 1] += d.site_obs_ptr[s];
    }
    d.site_obs_idx.resize(d.N);
    std::vector<int> pos(d.site_obs_ptr.begin(), d.site_obs_ptr.end());
    for (int i = 0; i < d.N; i++) {
        int s = d.spatial_group[i];
        d.site_obs_idx[pos[s]++] = i;
    }
}

// Build CSR time->obs mapping for TVC updates
inline void build_time_obs_map(GibbsData& d) {
    if (!d.has_tvc || d.tvc_n_times == 0) return;
    d.time_obs_ptr.assign(d.tvc_n_times + 1, 0);
    for (int i = 0; i < d.N; i++) {
        int t = d.tvc_time_index[i] - 1;  // 0-based
        if (t >= 0 && t < d.tvc_n_times)
            d.time_obs_ptr[t + 1]++;
    }
    for (int t = 0; t < d.tvc_n_times; t++)
        d.time_obs_ptr[t + 1] += d.time_obs_ptr[t];
    d.time_obs_idx.resize(d.N);
    std::vector<int> pos(d.time_obs_ptr.begin(), d.time_obs_ptr.end());
    for (int i = 0; i < d.N; i++) {
        int t = d.tvc_time_index[i] - 1;
        if (t >= 0 && t < d.tvc_n_times)
            d.time_obs_idx[pos[t]++] = i;
    }
}

// Compute TVC contribution to eta for observation i
inline double compute_tvc_eta_obs(int i, const double* tvc_w, const GibbsData& d) {
    if (!d.has_tvc) return 0.0;
    int t = d.tvc_time_index[i] - 1;
    int g = d.tvc_group_index[i] - 1;
    double eta = 0.0;
    for (int j = 0; j < d.tvc_n_terms; j++) {
        int w_idx = (g * d.tvc_n_terms + j) * d.tvc_n_times + t;
        eta += d.tvc_X[i * d.tvc_n_terms + j] * tvc_w[w_idx];
    }
    return eta;
}

// Log-likelihood for a single observation given eta_num, eta_denom
inline double obs_log_lik(int i, double eta_num, double eta_denom,
                          double phi_num, double phi_denom,
                          const GibbsData& d) {
    double ll = 0.0;

    switch (d.family) {
    case GibbsFamily::POISSON_GAMMA: {
        double mu = std::exp(std::min(eta_num, 20.0));
        ll += d.y_num[i] * eta_num - mu;                // Poisson (drop y! constant)
        double alpha = phi_denom;
        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double y_d = d.y_denom_cont[i];
        ll += alpha * std::log(alpha) - std::lgamma(alpha)
            + (alpha - 1.0) * std::log(y_d) - alpha * (y_d / mu_d + eta_denom);
        // Gamma: alpha*log(alpha/mu) + (alpha-1)*log(y) - alpha*y/mu - lgamma(alpha)
        // = alpha*log(alpha) - alpha*log(mu) + (alpha-1)*log(y) - alpha*y/mu - lgamma(alpha)
        break;
    }
    case GibbsFamily::NEGBIN_NEGBIN: {
        double mu_n = std::exp(std::min(eta_num, 20.0));
        double r_n = phi_num;
        int y_n = d.y_num[i];
        ll += std::lgamma(y_n + r_n) - std::lgamma(r_n) - std::lgamma(y_n + 1.0)
            + r_n * std::log(r_n / (mu_n + r_n))
            + y_n * std::log(mu_n / (mu_n + r_n));

        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double r_d = phi_denom;
        int y_d = d.y_denom[i];
        ll += std::lgamma(y_d + r_d) - std::lgamma(r_d) - std::lgamma(y_d + 1.0)
            + r_d * std::log(r_d / (mu_d + r_d))
            + y_d * std::log(mu_d / (mu_d + r_d));
        break;
    }
    case GibbsFamily::BINOMIAL: {
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y = d.y_num[i];
        int n_trials = d.y_denom[i];
        ll += y * std::log(p + 1e-300) + (n_trials - y) * std::log(1.0 - p + 1e-300);
        break;
    }
    case GibbsFamily::NEGBIN_GAMMA: {
        double mu_n = std::exp(std::min(eta_num, 20.0));
        double r_n = phi_num;
        int y_n = d.y_num[i];
        ll += std::lgamma(y_n + r_n) - std::lgamma(r_n) - std::lgamma(y_n + 1.0)
            + r_n * std::log(r_n / (mu_n + r_n))
            + y_n * std::log(mu_n / (mu_n + r_n));

        double alpha = phi_denom;
        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double y_d = d.y_denom_cont[i];
        ll += alpha * std::log(alpha) - std::lgamma(alpha)
            + (alpha - 1.0) * std::log(y_d) - alpha * (y_d / mu_d + eta_denom);
        break;
    }
    case GibbsFamily::GAMMA_GAMMA: {
        // Gamma(shape=phi, rate=phi/mu): LL = phi*log(phi/mu) - lgamma(phi) + (phi-1)*log(y) - phi*y/mu
        double mu_n = std::exp(std::min(eta_num, 20.0));
        double y_n = d.y_num_cont[i];
        if (y_n > 0.0) {
            ll += phi_num * std::log(phi_num / mu_n) - std::lgamma(phi_num)
                + (phi_num - 1.0) * std::log(y_n) - phi_num * y_n / mu_n;
        }
        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double y_d = d.y_denom_cont[i];
        if (y_d > 0.0) {
            ll += phi_denom * std::log(phi_denom / mu_d) - std::lgamma(phi_denom)
                + (phi_denom - 1.0) * std::log(y_d) - phi_denom * y_d / mu_d;
        }
        break;
    }
    case GibbsFamily::LOGNORMAL: {
        // log(y) ~ Normal(mu, sigma^2) where mu = eta, sigma = phi
        // LL = -log(y) - log(sigma) - 0.5*((log(y) - mu) / sigma)^2
        double mu_n = eta_num;  // mu IS the linear predictor
        double sigma_n = phi_num;
        double y_n = d.y_num_cont[i];
        double z_n = (std::log(y_n) - mu_n) / sigma_n;
        ll += -std::log(y_n) - std::log(sigma_n) - 0.5 * z_n * z_n;

        double mu_d = eta_denom;
        double sigma_d = phi_denom;
        double y_d = d.y_denom_cont[i];
        double z_d = (std::log(y_d) - mu_d) / sigma_d;
        ll += -std::log(y_d) - std::log(sigma_d) - 0.5 * z_d * z_d;
        break;
    }
    case GibbsFamily::BETA_BINOMIAL: {
        // BetaBinom(n, alpha, beta) where p = logistic(eta), phi = alpha + beta
        // alpha = p * phi, beta_param = (1-p) * phi
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y = d.y_num[i];
        int n_trials = d.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;
        ll += std::lgamma(y + alpha) + std::lgamma(n_trials - y + beta_param)
            - std::lgamma(n_trials + phi_num)
            - std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num);
        break;
    }
    }
    return ll;
}

// Compute temporal effect for observation i
inline double compute_temporal_eta_obs(int i, const double* phi_temporal,
                                        const GibbsData& d) {
    if (!d.has_temporal || phi_temporal == nullptr) return 0.0;
    int t = d.temporal_time_idx[i] - 1;
    int g = d.temporal_group_idx[i] - 1;
    if (t < 0 || g < 0) return 0.0;
    return phi_temporal[g * d.temporal_n_times + t];
}

// Sum log-likelihood over all observations at site s, given spatial effect value
inline double site_log_lik(int s, double spatial_effect,
                           const double* beta_num, const double* beta_denom,
                           double phi_num, double phi_denom,
                           const GibbsData& d,
                           const double* tvc_w = nullptr,
                           const double* phi_temporal = nullptr) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);

    for (int idx = d.site_obs_ptr[s]; idx < d.site_obs_ptr[s + 1]; idx++) {
        int i = d.site_obs_idx[idx];
        double eta_num = spatial_effect;
        double eta_denom = is_binomial ? 0.0 : spatial_effect;

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        // TVC contribution
        if (tvc_w != nullptr && d.has_tvc) {
            double tvc_eff = compute_tvc_eta_obs(i, tvc_w, d);
            eta_num += tvc_eff;
            if (!is_binomial && d.tvc_shared) eta_denom += tvc_eff;
        }

        // Temporal contribution
        if (d.has_temporal && phi_temporal != nullptr) {
            double temp_eff = compute_temporal_eta_obs(i, phi_temporal, d);
            eta_num += temp_eff;
            if (!is_binomial && d.temporal_shared) eta_denom += temp_eff;
        }

        ll += obs_log_lik(i, eta_num, eta_denom, phi_num, phi_denom, d);
    }
    return ll;
}

// Full log-likelihood over all observations
inline double full_log_lik(const double* phi, const double* beta_num,
                           const double* beta_denom,
                           double phi_num, double phi_denom,
                           const GibbsData& d,
                           const double* tvc_w = nullptr,
                           const double* phi_temporal = nullptr) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL || d.family == GibbsFamily::BETA_BINOMIAL);

    for (int i = 0; i < d.N; i++) {
        int s = d.spatial_group[i];
        double eta_num = phi[s];
        double eta_denom = is_binomial ? 0.0 : phi[s];

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

// =========================================================================
// ICAR prior log-density (up to normalization constant)
// =========================================================================

inline double icar_log_prior(const double* phi, double tau, int S,
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

// =========================================================================
// Main Gibbs sampler
// =========================================================================

struct GibbsResult {
    std::vector<double> draws_flat;     // (n_save × n_params) row-major
    std::vector<double> phi_draws_flat; // (n_save × S)
    int n_params;
    int n_save;
    int S;
    std::vector<std::string> param_names;

    // TVC draws
    std::vector<double> tvc_w_draws_flat;   // (n_save × n_tvc_w) row-major
    std::vector<double> tvc_tau_draws_flat;  // (n_save × n_tvc_terms)
    int tvc_n_w = 0;

    // Temporal GMRF draws
    std::vector<double> temporal_draws_flat; // (n_save × n_temporal_params)
    int temporal_n_params = 0;

    // HSGP draws
    std::vector<double> hsgp_beta_draws_flat;   // n_save x m_total
    std::vector<double> hsgp_f_draws_flat;      // n_save x N (reconstructed spatial field)
    int hsgp_m_total = 0;

    // Diagnostics
    std::vector<double> accept_phi;     // Acceptance rate per site
    double accept_beta;
    double accept_disp;
    double accept_tvc = 0.0;
    double accept_temporal = 0.0;
    double accept_hsgp_beta = 0.0;
    double accept_hsgp_hyper = 0.0;
};

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
