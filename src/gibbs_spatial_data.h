// gibbs_spatial_data.h
// Sliced from gibbs_spatial.h. Holds the GibbsFamily enum, the GibbsData /
// GibbsResult structs, and the family-agnostic likelihood / mapping helpers
// shared by all three samplers (ICAR, BYM2, HSGP).

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

} // namespace gibbs
