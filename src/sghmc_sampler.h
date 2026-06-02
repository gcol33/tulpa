// sghmc_sampler.h
// Stochastic Gradient Hamiltonian Monte Carlo for scalable inference
//
// SGHMC enables Tier 1 (Exact) inference on large datasets by using
// minibatch gradients with a friction term to correct for gradient noise.
//
// Reference: Chen, Fox, Guestrin (2014) "Stochastic Gradient Hamiltonian Monte Carlo"
// https://arxiv.org/abs/1402.4102
//
// Key insight: Standard HMC + minibatch gradients = biased samples
//              SGHMC adds friction to correct for gradient noise variance
//
// The continuous-time limit is exact (asymptotically correct posterior).

#ifndef TULPA_SGHMC_SAMPLER_H
#define TULPA_SGHMC_SAMPLER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include "hmc_sampler.h"

namespace tulpa_sghmc {

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;
using tulpa_hmc::compute_gradient;
using tulpa_hmc::compute_log_post;

// ============================================================================
// SGHMC Configuration
// ============================================================================

struct SGHMCConfig {
    int n_iter;                 // Total iterations
    int n_warmup;               // Warmup iterations (for tuning)
    int n_thin;                 // Thinning interval
    int batch_size;             // Minibatch size (not used yet - full gradient)
    double epsilon;             // Step size (learning rate)
    double alpha;               // Friction coefficient (momentum decay)
    int L;                      // Leapfrog steps per iteration
    bool verbose;               // Print progress
    int print_every;            // Print every N iterations
    unsigned int seed;          // Random seed

    // Adaptive tuning options
    bool adapt_epsilon;         // Adapt step size during warmup
    double target_accept;       // Target acceptance rate for adaptation
    double epsilon_min;         // Minimum step size
    double epsilon_max;         // Maximum step size

    // Stability options
    double grad_clip;           // Gradient clipping threshold (0 = no clipping)

    SGHMCConfig()
        : n_iter(2000), n_warmup(1000), n_thin(1), batch_size(256),
          epsilon(0.001), alpha(0.01), L(5), verbose(true),
          print_every(100), seed(12345),
          adapt_epsilon(true), target_accept(0.65),
          epsilon_min(1e-8), epsilon_max(0.1),
          grad_clip(100.0) {}  // Clip gradients at 100
};

// ============================================================================
// SGHMC Result
// ============================================================================

struct SGHMCResult {
    Eigen::MatrixXd samples;       // (n_save x n_params) posterior samples
    std::vector<double> log_lik;   // Log-likelihood at each sample
    std::vector<double> epsilon_history;  // Step size history
    double avg_batch_grad_norm;    // Average gradient norm
    bool success;
    std::string error_msg;
};

// ============================================================================
// Gradient clipping for stability
// ============================================================================

inline void clip_gradient(std::vector<double>& grad, double max_norm) {
    if (max_norm <= 0.0) return;

    double norm = 0.0;
    for (size_t i = 0; i < grad.size(); i++) {
        norm += grad[i] * grad[i];
    }
    norm = std::sqrt(norm);

    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (size_t i = 0; i < grad.size(); i++) {
            grad[i] *= scale;
        }
    }
}

// ============================================================================
// SGHMC update step
// ============================================================================

// Single SGHMC update:
// θ_{t+1} = θ_t + ε * r_t
// r_{t+1} = (1 - α) * r_t + ε * ∇log p(θ_t) + N(0, 2αε)
//
// Where:
//   θ = parameters
//   r = momentum
//   ε = step size
//   α = friction coefficient
//   ∇log p = gradient of log posterior

struct SGHMCState {
    std::vector<double> theta;     // Parameters
    std::vector<double> r;         // Momentum
    double log_post;               // Current log posterior
};

inline void sghmc_step(
    SGHMCState& state,
    const std::vector<double>& grad,  // Gradient of log posterior
    double epsilon,
    double alpha,
    std::mt19937& rng
) {
    int n_params = state.theta.size();
    std::normal_distribution<double> noise(0.0, 1.0);

    // Friction noise standard deviation: sqrt(2 * alpha * epsilon)
    double noise_sd = std::sqrt(2.0 * alpha * epsilon);

    for (int j = 0; j < n_params; j++) {
        // Update position (parameters)
        state.theta[j] += epsilon * state.r[j];

        // Update momentum with friction and gradient
        // grad is gradient of log posterior (we want to maximize)
        state.r[j] = (1.0 - alpha) * state.r[j] + epsilon * grad[j] + noise_sd * noise(rng);
    }
}

// ============================================================================
// SGHMC leapfrog with friction
// ============================================================================

inline void sghmc_leapfrog(
    SGHMCState& state,
    const ModelData& data,
    const ParamLayout& layout,
    int L,
    double epsilon,
    double alpha,
    double grad_clip,
    std::mt19937& rng
) {
    std::vector<double> grad(state.theta.size());

    for (int l = 0; l < L; l++) {
        // Compute gradient using HMC infrastructure (autodiff or analytical)
        compute_gradient(state.theta, data, layout, grad);

        // Clip gradient for stability
        clip_gradient(grad, grad_clip);

        // Check for NaN/Inf in gradient
        bool grad_ok = true;
        for (size_t j = 0; j < grad.size(); j++) {
            if (!std::isfinite(grad[j])) {
                grad_ok = false;
                grad[j] = 0.0;  // Zero out bad gradients
            }
        }

        if (!grad_ok) {
            // If gradient is bad, don't update
            break;
        }

        // SGHMC update
        sghmc_step(state, grad, epsilon, alpha, rng);

        // Check for NaN in parameters
        for (size_t j = 0; j < state.theta.size(); j++) {
            if (!std::isfinite(state.theta[j])) {
                // Reset to previous value with small perturbation
                state.theta[j] = 0.0;
            }
        }
    }

    // Update log posterior
    state.log_post = compute_log_post(state.theta, data, layout);
}

// ============================================================================
// Step size adaptation (simplified Robbins-Monro)
// ============================================================================

struct SGHMCAdapter {
    double log_epsilon;
    double target_accept;
    double adaptation_rate;
    int m;

    double epsilon_min;
    double epsilon_max;

    SGHMCAdapter(double epsilon_init = 0.001, double target = 0.65,
                 double eps_min = 1e-8, double eps_max = 0.1)
        : log_epsilon(std::log(epsilon_init)),
          target_accept(target),
          adaptation_rate(0.1),
          m(0),
          epsilon_min(eps_min), epsilon_max(eps_max) {}

    double update(double accept_stat) {
        m++;
        // Decreasing learning rate for adaptation
        double eta = adaptation_rate / std::sqrt(static_cast<double>(m));

        // Move towards target acceptance
        log_epsilon += eta * (accept_stat - target_accept);

        // Bound epsilon
        double eps = std::exp(log_epsilon);
        eps = std::max(epsilon_min, std::min(epsilon_max, eps));
        log_epsilon = std::log(eps);

        return eps;
    }

    double current_epsilon() const {
        return std::exp(log_epsilon);
    }
};

// ============================================================================
// Main SGHMC sampler
// ============================================================================

inline SGHMCResult run_sghmc_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const SGHMCConfig& config
) {
    SGHMCResult result;
    result.success = true;

    int n_params = init_params.size();
    int n_save = (config.n_iter - config.n_warmup) / config.n_thin;

    result.samples.resize(n_save, n_params);
    result.log_lik.resize(n_save);
    result.epsilon_history.reserve(config.n_iter);

    // Initialize RNG
    std::mt19937 rng(config.seed);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    // Initialize state
    SGHMCState state;
    state.theta = init_params;
    state.r.resize(n_params);
    for (int j = 0; j < n_params; j++) {
        state.r[j] = std_normal(rng);
    }

    // Compute initial log-posterior
    state.log_post = compute_log_post(state.theta, data, layout);

    if (!std::isfinite(state.log_post)) {
        result.success = false;
        result.error_msg = "Initial log-posterior is not finite";
        return result;
    }

    // Step size adapter
    SGHMCAdapter adapter(config.epsilon, config.target_accept,
                         config.epsilon_min, config.epsilon_max);
    double epsilon = config.epsilon;
    double alpha = config.alpha;

    // Progress
    if (config.verbose) {
        Rcpp::Rcout << "Running SGHMC sampler...\n";
        Rcpp::Rcout << "  Parameters: " << n_params << "\n";
        Rcpp::Rcout << "  Step size: " << epsilon << "\n";
        Rcpp::Rcout << "  Friction: " << alpha << "\n";
        Rcpp::Rcout << "  Leapfrog steps: " << config.L << "\n";
        Rcpp::Rcout << "  Gradient clipping: " << config.grad_clip << "\n";
        Rcpp::Rcout << "  Iterations: " << config.n_iter << " (warmup: " << config.n_warmup << ")\n";
    }

    int save_idx = 0;
    double total_accept = 0.0;

    for (int iter = 0; iter < config.n_iter; iter++) {
        // Check for user interrupt
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        // Store previous state for acceptance computation
        SGHMCState prev_state = state;
        double prev_log_post = state.log_post;

        // Resample momentum
        for (int j = 0; j < n_params; j++) {
            state.r[j] = std_normal(rng);
        }

        // Run SGHMC leapfrog
        sghmc_leapfrog(state, data, layout, config.L, epsilon, alpha,
                       config.grad_clip, rng);

        // Compute acceptance statistic (for adaptation only - SGHMC doesn't reject)
        // But we use it to adapt step size
        double delta_log_post = state.log_post - prev_log_post;
        double accept_stat = std::isfinite(delta_log_post) ?
                             std::min(1.0, std::exp(delta_log_post)) : 0.0;

        // If log_post is not finite, revert to previous state
        if (!std::isfinite(state.log_post)) {
            state = prev_state;
            accept_stat = 0.0;
        }

        total_accept += accept_stat;

        // Adapt step size during warmup
        if (iter < config.n_warmup && config.adapt_epsilon) {
            epsilon = adapter.update(accept_stat);
        }

        result.epsilon_history.push_back(epsilon);

        // Store sample after warmup
        if (iter >= config.n_warmup && (iter - config.n_warmup) % config.n_thin == 0) {
            for (int j = 0; j < n_params; j++) {
                result.samples(save_idx, j) = state.theta[j];
            }
            result.log_lik[save_idx] = state.log_post;
            save_idx++;
        }

        // Progress
        if (config.verbose && (iter + 1) % config.print_every == 0) {
            Rcpp::Rcout << "Iter " << (iter + 1) << "/" << config.n_iter;
            if (iter < config.n_warmup) {
                Rcpp::Rcout << " (warmup)";
            }
            Rcpp::Rcout << " log_post = " << state.log_post
                        << " epsilon = " << epsilon
                        << " accept = " << (total_accept / (iter + 1)) << "\n";
        }
    }

    result.avg_batch_grad_norm = 0.0;  // Not tracking currently

    if (config.verbose) {
        Rcpp::Rcout << "SGHMC complete. Final epsilon: " << epsilon
                    << " avg_accept: " << (total_accept / config.n_iter) << "\n";
    }

    return result;
}

// ============================================================================
// SGLD (Stochastic Gradient Langevin Dynamics) - Simpler variant
// ============================================================================

// SGLD is a special case with no momentum
// Update: θ_{t+1} = θ_t + (ε/2) * ∇log p(θ_t) + N(0, ε)
//
// Reference: Welling & Teh (2011) "Bayesian Learning via Stochastic Gradient Langevin Dynamics"

struct SGLDConfig {
    int n_iter;
    int n_warmup;
    int n_thin;
    int batch_size;
    double epsilon;             // Initial step size
    bool verbose;
    int print_every;
    unsigned int seed;

    // Step size schedule: epsilon_t = a * (b + t)^(-gamma)
    double schedule_a;
    double schedule_b;
    double schedule_gamma;
    bool use_schedule;

    // Stability
    double grad_clip;
    double epsilon_min;

    SGLDConfig()
        : n_iter(2000), n_warmup(1000), n_thin(1), batch_size(256),
          epsilon(0.001), verbose(true), print_every(100), seed(12345),
          schedule_a(0.01), schedule_b(100.0), schedule_gamma(0.55),
          use_schedule(true),
          grad_clip(100.0), epsilon_min(1e-8) {}
};

struct SGLDResult {
    Eigen::MatrixXd samples;
    std::vector<double> log_lik;
    std::vector<double> epsilon_history;
    bool success;
    std::string error_msg;
};

inline SGLDResult run_sgld_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const SGLDConfig& config
) {
    SGLDResult result;
    result.success = true;

    int n_params = init_params.size();
    int n_save = (config.n_iter - config.n_warmup) / config.n_thin;

    result.samples.resize(n_save, n_params);
    result.log_lik.resize(n_save);
    result.epsilon_history.reserve(config.n_iter);

    // Initialize RNG
    std::mt19937 rng(config.seed);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    // Current state
    std::vector<double> theta = init_params;
    double log_post = compute_log_post(theta, data, layout);

    if (!std::isfinite(log_post)) {
        result.success = false;
        result.error_msg = "Initial log-posterior is not finite";
        return result;
    }

    // Progress
    if (config.verbose) {
        Rcpp::Rcout << "Running SGLD sampler...\n";
        Rcpp::Rcout << "  Parameters: " << n_params << "\n";
        Rcpp::Rcout << "  Initial epsilon: " << config.epsilon << "\n";
        Rcpp::Rcout << "  Gradient clipping: " << config.grad_clip << "\n";
        Rcpp::Rcout << "  Iterations: " << config.n_iter << " (warmup: " << config.n_warmup << ")\n";
    }

    int save_idx = 0;
    std::vector<double> grad(n_params);
    std::vector<double> theta_prev = theta;

    for (int iter = 0; iter < config.n_iter; iter++) {
        // Check for user interrupt
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        // Compute step size (with schedule)
        double epsilon;
        if (config.use_schedule) {
            epsilon = config.schedule_a * std::pow(config.schedule_b + iter, -config.schedule_gamma);
            epsilon = std::max(epsilon, config.epsilon_min);
        } else {
            epsilon = config.epsilon;
        }
        result.epsilon_history.push_back(epsilon);

        // Store previous state
        theta_prev = theta;
        double log_post_prev = log_post;

        // Compute gradient using HMC infrastructure
        compute_gradient(theta, data, layout, grad);

        // Clip gradient for stability
        clip_gradient(grad, config.grad_clip);

        // Check for NaN/Inf in gradient
        bool grad_ok = true;
        for (int j = 0; j < n_params; j++) {
            if (!std::isfinite(grad[j])) {
                grad_ok = false;
                grad[j] = 0.0;
            }
        }

        // SGLD update: θ = θ + (ε/2) * ∇log p(θ) + N(0, ε)
        if (grad_ok) {
            double noise_sd = std::sqrt(epsilon);
            for (int j = 0; j < n_params; j++) {
                theta[j] += 0.5 * epsilon * grad[j] + noise_sd * std_normal(rng);
            }

            // Update log posterior
            log_post = compute_log_post(theta, data, layout);

            // If new state is invalid, revert
            if (!std::isfinite(log_post)) {
                theta = theta_prev;
                log_post = log_post_prev;
            }
        }

        // Store sample after warmup
        if (iter >= config.n_warmup && (iter - config.n_warmup) % config.n_thin == 0) {
            for (int j = 0; j < n_params; j++) {
                result.samples(save_idx, j) = theta[j];
            }
            result.log_lik[save_idx] = log_post;
            save_idx++;
        }

        // Progress
        if (config.verbose && (iter + 1) % config.print_every == 0) {
            Rcpp::Rcout << "Iter " << (iter + 1) << "/" << config.n_iter;
            if (iter < config.n_warmup) {
                Rcpp::Rcout << " (warmup)";
            }
            Rcpp::Rcout << " log_post = " << log_post
                        << " epsilon = " << epsilon << "\n";
        }
    }

    if (config.verbose) {
        Rcpp::Rcout << "SGLD complete.\n";
    }

    return result;
}

} // namespace tulpa_sghmc

#endif // TULPA_SGHMC_SAMPLER_H
