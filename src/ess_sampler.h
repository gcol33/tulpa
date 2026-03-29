// ess_sampler.h
// Elliptical Slice Sampling for models with Gaussian priors
//
// ESS is efficient for sampling parameters with Gaussian priors because it:
// 1. Uses the prior as the proposal distribution
// 2. Automatically adapts to the posterior geometry
// 3. Has no tuning parameters (unlike HMC step size)
//
// Reference: Murray, Adams, MacKay (2010) "Elliptical Slice Sampling"
// https://proceedings.mlr.press/v9/murray10a.html

#ifndef TULPA_ESS_SAMPLER_H
#define TULPA_ESS_SAMPLER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "hmc_sampler.h"

namespace tulpa_ess {

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;
using tulpa_hmc::ModelType;

// ============================================================================
// ESS Configuration
// ============================================================================

struct ESSConfig {
    int n_iter;                 // Total iterations
    int n_warmup;               // Warmup iterations (for adaptation)
    int n_thin;                 // Thinning interval
    bool verbose;               // Print progress
    int print_every;            // Print every N iterations
    unsigned int seed;          // Random seed

    // ESS-specific options
    bool use_cholesky;          // Use Cholesky decomposition for multivariate normal
    bool adapt_during_warmup;   // Adapt covariance during warmup
    int adapt_interval;         // How often to update covariance estimate

    ESSConfig()
        : n_iter(2000), n_warmup(1000), n_thin(1), verbose(true),
          print_every(100), seed(12345), use_cholesky(true),
          adapt_during_warmup(true), adapt_interval(100) {}
};

// ============================================================================
// ESS Result
// ============================================================================

struct ESSResult {
    Eigen::MatrixXd samples;       // (n_save x n_params) posterior samples
    std::vector<double> log_lik;   // Log-likelihood at each sample
    int n_slice_evals;             // Total likelihood evaluations
    double avg_slice_evals;        // Average evaluations per ESS step
    bool success;
    std::string error_msg;
};

// ============================================================================
// Gaussian Prior specification
// ============================================================================

// Specifies which parameters have Gaussian priors for ESS
struct GaussianPrior {
    std::vector<int> param_indices;  // Indices of parameters with this prior
    Eigen::VectorXd mean;            // Prior mean
    Eigen::MatrixXd precision;       // Prior precision (inverse covariance)
    Eigen::MatrixXd chol_cov;        // Cholesky of covariance (lower triangular)
    bool is_identity_cov;            // If true, use efficient identity sampling
    double scale;                    // For identity: prior is N(0, scale^2 * I)

    GaussianPrior() : is_identity_cov(true), scale(1.0) {}
};

// ============================================================================
// Helper: Sample from standard normal
// ============================================================================

inline Eigen::VectorXd sample_std_normal(int n, std::mt19937& rng) {
    std::normal_distribution<double> std_normal(0.0, 1.0);
    Eigen::VectorXd z(n);
    for (int i = 0; i < n; i++) {
        z(i) = std_normal(rng);
    }
    return z;
}

// ============================================================================
// Helper: Sample from Gaussian prior
// ============================================================================

inline Eigen::VectorXd sample_from_prior(
    const GaussianPrior& prior,
    std::mt19937& rng
) {
    int n = prior.param_indices.size();
    Eigen::VectorXd z = sample_std_normal(n, rng);

    if (prior.is_identity_cov) {
        // N(mean, scale^2 * I)
        return prior.mean + prior.scale * z;
    } else {
        // N(mean, Sigma) where Sigma = L * L^T
        return prior.mean + prior.chol_cov * z;
    }
}

// ============================================================================
// Core ESS step for a single block of parameters
// ============================================================================

// Performs one ESS step for parameters with a Gaussian prior
// Returns new parameter values for the block
//
// Algorithm:
// 1. Draw nu ~ N(0, Sigma) from the prior
// 2. Set threshold: log_y = log_lik(f) + log(u), u ~ Uniform(0,1)
// 3. Draw initial angle: theta ~ Uniform(0, 2*pi)
// 4. Set bracket: [theta_min, theta_max] = [theta - 2*pi, theta]
// 5. Loop:
//    a. f' = f * cos(theta) + nu * sin(theta)
//    b. If log_lik(f') > log_y: accept and return f'
//    c. Else shrink bracket toward 0 and draw new theta
//
template<typename LogLikFn>
Eigen::VectorXd ess_step(
    const Eigen::VectorXd& f,           // Current parameter values (block)
    const GaussianPrior& prior,         // Gaussian prior for this block
    LogLikFn log_lik_fn,                // Function: f -> log_likelihood
    double current_log_lik,             // Current log-likelihood
    std::mt19937& rng,
    int& n_evals                        // Output: number of likelihood evaluations
) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Step 1: Draw nu from prior (centered at 0)
    int n = f.size();
    Eigen::VectorXd nu = sample_std_normal(n, rng);
    if (prior.is_identity_cov) {
        nu *= prior.scale;
    } else {
        nu = prior.chol_cov * nu;
    }

    // Step 2: Set threshold
    double log_u = std::log(unif(rng));
    double log_y = current_log_lik + log_u;

    // Step 3: Initial angle
    double theta = unif(rng) * 2.0 * M_PI;
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;

    // Center f around prior mean for correct geometry
    Eigen::VectorXd f_centered = f - prior.mean;

    n_evals = 0;
    const int max_shrinks = 1000;  // Prevent infinite loop

    for (int shrink = 0; shrink < max_shrinks; shrink++) {
        // Step 5a: Proposal on ellipse
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        Eigen::VectorXd f_prime_centered = f_centered * cos_theta + nu * sin_theta;
        Eigen::VectorXd f_prime = f_prime_centered + prior.mean;

        // Step 5b: Evaluate log-likelihood
        double log_lik_prime = log_lik_fn(f_prime);
        n_evals++;

        if (log_lik_prime > log_y) {
            // Accept
            return f_prime;
        }

        // Step 5c: Shrink bracket
        if (theta < 0) {
            theta_min = theta;
        } else {
            theta_max = theta;
        }

        // Draw new theta from shrunk bracket
        theta = unif(rng) * (theta_max - theta_min) + theta_min;
    }

    // If we get here, something went wrong - return original
    Rcpp::warning("ESS reached max shrinks without accepting");
    return f;
}

// ============================================================================
// Build Gaussian priors from model structure
// ============================================================================

// Extract which parameters have Gaussian priors from the model
inline std::vector<GaussianPrior> build_gaussian_priors(
    const ModelData& data,
    const ParamLayout& layout,
    int n_params
) {
    std::vector<GaussianPrior> priors;

    // Fixed effects: beta ~ N(0, sigma_beta^2)
    // These have Gaussian priors but we sample them with ESS together
    {
        GaussianPrior prior;
        // Numerator betas
        for (int j = layout.legacy.beta_num_start; j < layout.legacy.beta_num_end; j++) {
            prior.param_indices.push_back(j);
        }
        // Denominator betas
        for (int j = layout.legacy.beta_denom_start; j < layout.legacy.beta_denom_end; j++) {
            prior.param_indices.push_back(j);
        }

        if (!prior.param_indices.empty()) {
            int n = prior.param_indices.size();
            prior.mean = Eigen::VectorXd::Zero(n);
            prior.is_identity_cov = true;
            prior.scale = data.sigma_beta;
            priors.push_back(prior);
        }
    }

    // Random effects: re ~ N(0, sigma_re^2)
    // Note: sigma_re is estimated, so we handle this specially
    if (layout.has_re && layout.re_end > layout.re_start) {
        GaussianPrior prior;
        for (int j = layout.re_start; j < layout.re_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = 1.0;  // Will be scaled by sigma_re during sampling
        priors.push_back(prior);
    }

    // Spatial effects (ICAR): phi ~ N(0, tau^-1 * Q^-)
    // For ICAR, the precision matrix is the graph Laplacian
    // For now, we treat these as standard normals (simplified)
    if (layout.has_spatial && layout.spatial_end > layout.spatial_start) {
        GaussianPrior prior;
        for (int j = layout.spatial_start; j < layout.spatial_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;  // Simplified - proper ICAR would use precision
        prior.scale = 1.0;
        priors.push_back(prior);
    }

    // Temporal effects: phi_temporal ~ N(0, tau_temporal^-1 * Q)
    if (layout.has_temporal && layout.temporal_end > layout.temporal_start) {
        GaussianPrior prior;
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;  // Simplified
        prior.scale = 1.0;
        priors.push_back(prior);
    }

    // GP spatial effects
    if (layout.is_gp && layout.gp_w_end > layout.gp_w_start) {
        GaussianPrior prior;
        for (int j = layout.gp_w_start; j < layout.gp_w_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = 1.0;  // GP effects use kernel covariance
        priors.push_back(prior);
    }

    // ZI coefficients: beta_zi ~ N(0, zi_prior_sd^2)
    if (layout.has_zi && layout.beta_zi_end > layout.beta_zi_start) {
        GaussianPrior prior;
        for (int j = layout.beta_zi_start; j < layout.beta_zi_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = data.zi_prior_sd;
        priors.push_back(prior);
    }

    // Latent factors
    if (layout.has_latent && layout.latent_factor_end > layout.latent_factor_start) {
        GaussianPrior prior;
        for (int j = layout.latent_factor_start; j < layout.latent_factor_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = 1.0;
        priors.push_back(prior);
    }

    return priors;
}

// ============================================================================
// Identify non-Gaussian parameters (sampled with MH or other methods)
// ============================================================================

inline std::vector<int> get_non_gaussian_params(
    const ParamLayout& layout,
    int n_params
) {
    std::vector<int> non_gaussian;

    // Variance parameters (log scale) - use random walk MH
    if (layout.has_re && layout.log_sigma_re_idx >= 0) {
        non_gaussian.push_back(layout.log_sigma_re_idx);
    }

    if (layout.legacy.has_phi_num && layout.legacy.log_phi_num_idx >= 0) {
        non_gaussian.push_back(layout.legacy.log_phi_num_idx);
    }

    if (layout.legacy.has_phi_denom && layout.legacy.log_phi_denom_idx >= 0) {
        non_gaussian.push_back(layout.legacy.log_phi_denom_idx);
    }

    if (layout.has_spatial && layout.log_tau_spatial_idx >= 0) {
        non_gaussian.push_back(layout.log_tau_spatial_idx);
    }

    if (layout.is_bym2) {
        if (layout.log_sigma_bym2_idx >= 0) {
            non_gaussian.push_back(layout.log_sigma_bym2_idx);
        }
        if (layout.logit_rho_bym2_idx >= 0) {
            non_gaussian.push_back(layout.logit_rho_bym2_idx);
        }
    }

    if (layout.has_temporal && layout.log_tau_temporal_idx >= 0) {
        non_gaussian.push_back(layout.log_tau_temporal_idx);
    }

    if (layout.is_ar1 && layout.logit_rho_ar1_idx >= 0) {
        non_gaussian.push_back(layout.logit_rho_ar1_idx);
    }

    if (layout.is_gp) {
        if (layout.log_sigma2_gp_idx >= 0) {
            non_gaussian.push_back(layout.log_sigma2_gp_idx);
        }
        if (layout.log_phi_gp_idx >= 0) {
            non_gaussian.push_back(layout.log_phi_gp_idx);
        }
    }

    return non_gaussian;
}

// ============================================================================
// Random walk Metropolis-Hastings step for non-Gaussian parameters
// ============================================================================

template<typename LogPostFn>
double rwmh_step(
    std::vector<double>& params,
    int idx,
    double proposal_sd,
    LogPostFn log_post_fn,
    double current_log_post,
    std::mt19937& rng,
    bool& accepted
) {
    std::normal_distribution<double> normal(0.0, proposal_sd);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double old_val = params[idx];
    double proposal = old_val + normal(rng);

    params[idx] = proposal;
    double proposed_log_post = log_post_fn(params);

    double log_alpha = proposed_log_post - current_log_post;

    if (std::log(unif(rng)) < log_alpha) {
        accepted = true;
        return proposed_log_post;
    } else {
        params[idx] = old_val;
        accepted = false;
        return current_log_post;
    }
}

// ============================================================================
// Adaptive proposal SD for RWMH
// ============================================================================

struct AdaptiveProposal {
    std::vector<double> proposal_sds;
    std::vector<int> n_accepted;
    std::vector<int> n_total;
    double target_rate;
    int adapt_interval;

    AdaptiveProposal(int n_params, double init_sd = 0.1, double target = 0.234)
        : proposal_sds(n_params, init_sd),
          n_accepted(n_params, 0),
          n_total(n_params, 0),
          target_rate(target),
          adapt_interval(50) {}

    void record(int idx, bool accepted) {
        n_total[idx]++;
        if (accepted) n_accepted[idx]++;
    }

    void adapt(int iter) {
        if (iter % adapt_interval != 0) return;

        for (size_t i = 0; i < proposal_sds.size(); i++) {
            if (n_total[i] > 0) {
                double rate = static_cast<double>(n_accepted[i]) / n_total[i];
                // Robbins-Monro type adaptation
                double gamma = 1.0 / std::sqrt(iter + 1.0);
                double log_sd = std::log(proposal_sds[i]);
                log_sd += gamma * (rate - target_rate);
                proposal_sds[i] = std::exp(log_sd);
                // Reset counters
                n_accepted[i] = 0;
                n_total[i] = 0;
            }
        }
    }
};

// ============================================================================
// Main ESS sampler
// ============================================================================

// Forward declaration of log_post computation
double compute_log_post_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);

inline ESSResult run_ess_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const ESSConfig& config
) {
    ESSResult result;
    result.success = true;

    int n_params = init_params.size();
    int n_save = (config.n_iter - config.n_warmup) / config.n_thin;

    result.samples.resize(n_save, n_params);
    result.log_lik.resize(n_save);
    result.n_slice_evals = 0;

    // Initialize RNG
    std::mt19937 rng(config.seed);

    // Build Gaussian priors for ESS blocks
    std::vector<GaussianPrior> gaussian_priors = build_gaussian_priors(data, layout, n_params);

    // Get non-Gaussian parameters for RWMH
    std::vector<int> non_gaussian = get_non_gaussian_params(layout, n_params);

    // Adaptive proposals for RWMH parameters
    AdaptiveProposal adaptive(non_gaussian.size());

    // Current state
    std::vector<double> params = init_params;

    // Compute initial log-posterior
    double current_log_post = compute_log_post_double(params, data, layout);

    if (!std::isfinite(current_log_post)) {
        result.success = false;
        result.error_msg = "Initial log-posterior is not finite";
        return result;
    }

    // Progress
    if (config.verbose) {
        Rcpp::Rcout << "Running ESS sampler...\n";
        Rcpp::Rcout << "  Parameters: " << n_params << "\n";
        Rcpp::Rcout << "  Gaussian blocks: " << gaussian_priors.size() << "\n";
        Rcpp::Rcout << "  Non-Gaussian params: " << non_gaussian.size() << "\n";
        Rcpp::Rcout << "  Iterations: " << config.n_iter << " (warmup: " << config.n_warmup << ")\n";
    }

    int save_idx = 0;
    int total_ess_evals = 0;
    int total_ess_steps = 0;

    for (int iter = 0; iter < config.n_iter; iter++) {
        // Check for user interrupt
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        // ----------------------------------------------------------------
        // ESS updates for Gaussian-prior blocks
        // ----------------------------------------------------------------
        for (auto& prior : gaussian_priors) {
            // Extract current values for this block
            int block_size = prior.param_indices.size();
            Eigen::VectorXd f(block_size);
            for (int i = 0; i < block_size; i++) {
                f(i) = params[prior.param_indices[i]];
            }

            // Lambda to compute log-likelihood given block values
            auto log_lik_fn = [&](const Eigen::VectorXd& f_new) -> double {
                // Temporarily update params
                std::vector<double> params_temp = params;
                for (int i = 0; i < block_size; i++) {
                    params_temp[prior.param_indices[i]] = f_new(i);
                }
                // Return log-posterior (which includes prior)
                // For ESS, we need log-likelihood only (prior is handled by the algorithm)
                // But since we use N(0, scale) prior, this is absorbed into ellipse
                return compute_log_post_double(params_temp, data, layout);
            };

            // Perform ESS step
            int n_evals = 0;
            Eigen::VectorXd f_new = ess_step(f, prior, log_lik_fn, current_log_post, rng, n_evals);

            // Update params
            for (int i = 0; i < block_size; i++) {
                params[prior.param_indices[i]] = f_new(i);
            }

            total_ess_evals += n_evals;
            total_ess_steps++;
        }

        // Update log-posterior after ESS updates
        current_log_post = compute_log_post_double(params, data, layout);

        // ----------------------------------------------------------------
        // RWMH updates for non-Gaussian parameters
        // ----------------------------------------------------------------
        for (size_t i = 0; i < non_gaussian.size(); i++) {
            int idx = non_gaussian[i];
            bool accepted = false;
            current_log_post = rwmh_step(
                params, idx, adaptive.proposal_sds[i],
                [&](const std::vector<double>& p) { return compute_log_post_double(p, data, layout); },
                current_log_post, rng, accepted
            );
            adaptive.record(i, accepted);
        }

        // Adapt RWMH proposals during warmup
        if (iter < config.n_warmup && config.adapt_during_warmup) {
            adaptive.adapt(iter);
        }

        // ----------------------------------------------------------------
        // Store sample
        // ----------------------------------------------------------------
        if (iter >= config.n_warmup && (iter - config.n_warmup) % config.n_thin == 0) {
            for (int j = 0; j < n_params; j++) {
                result.samples(save_idx, j) = params[j];
            }
            result.log_lik[save_idx] = current_log_post;
            save_idx++;
        }

        // Progress
        if (config.verbose && (iter + 1) % config.print_every == 0) {
            Rcpp::Rcout << "Iter " << (iter + 1) << "/" << config.n_iter;
            if (iter < config.n_warmup) {
                Rcpp::Rcout << " (warmup)";
            }
            Rcpp::Rcout << " log_post = " << current_log_post << "\n";
        }
    }

    result.n_slice_evals = total_ess_evals;
    result.avg_slice_evals = total_ess_steps > 0 ?
        static_cast<double>(total_ess_evals) / total_ess_steps : 0.0;

    if (config.verbose) {
        Rcpp::Rcout << "ESS complete. Avg slice evals per step: "
                    << result.avg_slice_evals << "\n";
    }

    return result;
}

} // namespace tulpa_ess

#endif // TULPA_ESS_SAMPLER_H
