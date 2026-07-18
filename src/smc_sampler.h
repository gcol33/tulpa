// smc_sampler.h
// Sequential Monte Carlo sampler with adaptive tempering
//
// Moves a particle population from prior to posterior via likelihood tempering.
// Each step: reweight, resample (systematic), MCMC mutations.
// The marginal likelihood estimate (log Z) comes free as a byproduct.
//
// Reference: Del Moral, Doucet, Jasra (2006).
//            Chopin & Papaspiliopoulos (2020) "An Introduction to SMC", Springer.

#ifndef TULPA_SMC_SAMPLER_H
#define TULPA_SMC_SAMPLER_H

#include <vector>
#include <functional>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

namespace tulpa_smc {

// ============================================================================
// Result container
// ============================================================================

struct SMCResult {
    std::vector<std::vector<double>> particles;  // final N particles x dim
    std::vector<double> weights;                  // final normalized weights
    double log_marginal_likelihood;               // log Z estimate
    std::vector<double> temperatures;             // beta schedule (including 0 and 1)
    int n_temperatures;                           // number of tempering steps
    std::vector<double> ess_history;              // ESS at each step
    int n_resamples;                              // how many times resampling triggered
    int n_mutations;                              // total MCMC mutation steps applied
};

// ============================================================================
// Helpers (implemented inline below)
// ============================================================================

// Log-sum-exp with numerical stability
inline double log_sum_exp(const std::vector<double>& log_vals) {
    if (log_vals.empty()) return -std::numeric_limits<double>::infinity();
    double max_val = *std::max_element(log_vals.begin(), log_vals.end());
    if (!std::isfinite(max_val)) return max_val;
    double sum = 0.0;
    for (double v : log_vals) {
        sum += std::exp(v - max_val);
    }
    return max_val + std::log(sum);
}

// Compute ESS from unnormalized log-weights
// ESS = (sum w)^2 / sum(w^2) computed in log space
inline double compute_ess(const std::vector<double>& log_weights) {
    int N = static_cast<int>(log_weights.size());
    if (N == 0) return 0.0;

    double max_lw = *std::max_element(log_weights.begin(), log_weights.end());
    if (!std::isfinite(max_lw)) return 0.0;

    double sum1 = 0.0, sum2 = 0.0;
    for (int n = 0; n < N; n++) {
        double w = std::exp(log_weights[n] - max_lw);
        sum1 += w;
        sum2 += w * w;
    }
    if (sum2 == 0.0) return 0.0;
    return (sum1 * sum1) / sum2;
}

// Systematic resampling: given normalized weights, return ancestor indices
inline std::vector<int> systematic_resample(
    const std::vector<double>& weights, int N, std::mt19937& rng
) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double u = unif(rng) / N;
    std::vector<int> ancestors(N);
    double cumsum = weights[0];
    int j = 0;
    for (int i = 0; i < N; i++) {
        double target = u + static_cast<double>(i) / N;
        while (cumsum < target && j < N - 1) {
            j++;
            cumsum += weights[j];
        }
        ancestors[i] = j;
    }
    return ancestors;
}

// Find next temperature via bisection so that ESS(weights) ~ ess_target
// weights_n = exp((beta_new - beta_current) * log_lik_n)
inline double find_next_temperature(
    const std::vector<double>& log_likelihoods,
    double beta_current,
    double ess_target,
    int N
) {
    double lo = beta_current;
    double hi = 1.0;

    // Check if jumping straight to 1.0 keeps ESS above target
    {
        double delta = hi - beta_current;
        std::vector<double> log_w(N);
        for (int n = 0; n < N; n++) log_w[n] = delta * log_likelihoods[n];
        double ess = compute_ess(log_w);
        if (ess >= ess_target) return 1.0;
    }

    // Binary search
    for (int iter = 0; iter < 50; iter++) {
        double mid = 0.5 * (lo + hi);
        double delta = mid - beta_current;

        std::vector<double> log_w(N);
        for (int n = 0; n < N; n++) log_w[n] = delta * log_likelihoods[n];
        double ess = compute_ess(log_w);

        if (ess > ess_target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    double result = 0.5 * (lo + hi);
    if (result > 1.0 - 1e-10) result = 1.0;
    return result;
}

// ============================================================================
// Main SMC sampler
// ============================================================================

// Callbacks:
//   log_prior(theta)        -> log p(theta)
//   log_likelihood(theta)   -> log p(y | theta)
//   prior_sample(theta, rng, beta) -> fill theta with a draw from the prior
//   mcmc_mutation(theta, beta, rng) -> one MCMC step targeting p(theta) * p(y|theta)^beta

inline SMCResult smc_sample(
    const std::function<double(const std::vector<double>&)>& log_prior,
    const std::function<double(const std::vector<double>&)>& log_likelihood,
    const std::function<void(std::vector<double>&, std::mt19937&, double)>& prior_sample,
    const std::function<void(std::vector<double>&, double, std::mt19937&)>& mcmc_mutation,
    int dim,
    int n_particles = 1000,
    double ess_threshold = 0.5,
    int n_mcmc_steps = 5,
    unsigned int seed = 42
) {
    int N = n_particles;
    std::mt19937 rng(seed);

    SMCResult result;
    result.log_marginal_likelihood = 0.0;
    result.n_resamples = 0;
    result.n_mutations = 0;
    result.temperatures.push_back(0.0);

    // ------------------------------------------------------------------
    // 1. Initialize: draw N particles from the prior
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> particles(N, std::vector<double>(dim));
    for (int n = 0; n < N; n++) {
        prior_sample(particles[n], rng, 0.0);
    }

    // Pre-compute log-likelihoods
    std::vector<double> log_liks(N);
    for (int n = 0; n < N; n++) {
        log_liks[n] = log_likelihood(particles[n]);
    }

    double beta = 0.0;
    double ess_target = ess_threshold * N;

    // ------------------------------------------------------------------
    // 2. Tempering loop
    // ------------------------------------------------------------------
    while (beta < 1.0) {
        // (a) Adaptive temperature selection
        double beta_new = find_next_temperature(log_liks, beta, ess_target, N);
        double delta = beta_new - beta;

        // (b) Compute incremental log-weights
        std::vector<double> log_w(N);
        double max_lw = -std::numeric_limits<double>::infinity();
        for (int n = 0; n < N; n++) {
            log_w[n] = delta * log_liks[n];
            if (log_w[n] > max_lw) max_lw = log_w[n];
        }

        // Accumulate log marginal likelihood: log Z_t = log mean(w_n)
        double sum_w = 0.0;
        for (int n = 0; n < N; n++) {
            sum_w += std::exp(log_w[n] - max_lw);
        }
        result.log_marginal_likelihood += max_lw + std::log(sum_w / N);

        // Normalize weights for resampling
        std::vector<double> weights(N);
        for (int n = 0; n < N; n++) {
            weights[n] = std::exp(log_w[n] - max_lw) / sum_w;
        }

        // ESS diagnostic
        double ess = compute_ess(log_w);
        result.ess_history.push_back(ess);

        // (c) Resample every step. After resampling the particles are equally
        // weighted, which is exactly the assumption the equal-weight temperature
        // search (find_next_temperature), the log-mean-weight Z increment above,
        // and the uniform final weights below all rely on. Adaptive resampling
        // (only when ess < target) would instead require carrying normalized
        // weights across rounds and feeding them back into all three.
        // Resampling unconditionally keeps these consistent (Del Moral
        // et al. 2006); the modest extra Monte Carlo variance is offset by the
        // per-particle MCMC mutations that follow.
        {
            auto ancestors = systematic_resample(weights, N, rng);
            auto old_particles = particles;
            auto old_log_liks = log_liks;
            for (int n = 0; n < N; n++) {
                particles[n] = old_particles[ancestors[n]];
                log_liks[n] = old_log_liks[ancestors[n]];
            }
            result.n_resamples++;
        }

        // (d) MCMC mutations
        beta = beta_new;
        result.temperatures.push_back(beta);

        for (int n = 0; n < N; n++) {
            for (int k = 0; k < n_mcmc_steps; k++) {
                mcmc_mutation(particles[n], beta, rng);
            }
            // Refresh log-likelihood after mutation
            log_liks[n] = log_likelihood(particles[n]);
        }
        result.n_mutations += N * n_mcmc_steps;
    }

    // ------------------------------------------------------------------
    // 3. Final: uniform weights after last resampling
    // ------------------------------------------------------------------
    result.particles = particles;
    result.weights.assign(N, 1.0 / N);
    result.n_temperatures = static_cast<int>(result.temperatures.size()) - 1;  // exclude beta=0

    return result;
}

} // namespace tulpa_smc

#endif
