// sghmc_sampler.cpp
// Definitions of the SGHMC and SGLD drivers declared in sghmc_sampler.h.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "sghmc_sampler.h"

namespace tulpa_sghmc {

SGHMCResult run_sghmc_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const SGHMCConfig& config
) {
    SGHMCResult result;
    result.success = true;

    int n_params = init_params.size();
    // Post-warmup iterations store when (iter - n_warmup) % n_thin == 0, which
    // fires ceil(n_post / n_thin) times; a warmup at least as long as the run
    // stores nothing.
    const int n_thin = std::max(1, config.n_thin);
    const int n_post = config.n_iter - config.n_warmup;
    const int n_save = (n_post > 0) ? ((n_post + n_thin - 1) / n_thin) : 0;

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
        if (iter >= config.n_warmup && (iter - config.n_warmup) % n_thin == 0) {
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

SGLDResult run_sgld_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const SGLDConfig& config
) {
    SGLDResult result;
    result.success = true;

    int n_params = init_params.size();
    // Post-warmup iterations store when (iter - n_warmup) % n_thin == 0, which
    // fires ceil(n_post / n_thin) times; a warmup at least as long as the run
    // stores nothing.
    const int n_thin = std::max(1, config.n_thin);
    const int n_post = config.n_iter - config.n_warmup;
    const int n_save = (n_post > 0) ? ((n_post + n_thin - 1) / n_thin) : 0;

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
        if (iter >= config.n_warmup && (iter - config.n_warmup) % n_thin == 0) {
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
