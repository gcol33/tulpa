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

#include <Rcpp.h>
#include <vector>
#include <functional>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>

namespace tulpa_smc {

// Ceiling on adaptive tempering steps. The schedule is data-driven, so it has
// no a priori length; a well-posed population reaches beta = 1 in tens to a few
// hundred steps, while a population the ESS target is unreachable for advances
// by the representable floor below and would otherwise loop without end.
const int kMaxTemperatureSteps = 10000;

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
    if (N <= 0 || static_cast<int>(weights.size()) < N) {
        throw std::invalid_argument(
            "tulpa SMC: systematic_resample needs N >= 1 and one weight per "
            "particle (N = " + std::to_string(N) + ", weights = " +
            std::to_string(weights.size()) + ").");
    }
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

    // `lo` is only ever raised off beta_current by a midpoint that clears the
    // ESS target. When none does -- a population whose incremental weights are
    // dominated by one particle at every representable delta, which is what a
    // non-finite log-likelihood among the particles produces -- `lo` stays at
    // beta_current, `hi` converges down onto it, and the bisection returns the
    // temperature it started from. Floor the result at the next representable
    // temperature so the caller's schedule is strictly increasing and its step
    // cap can end the run instead of it spinning.
    double result = 0.5 * (lo + hi);
    const double floor_beta = std::nextafter(beta_current, 1.0);
    if (!(result > floor_beta)) result = floor_beta;
    if (result > 1.0 - 1e-10) result = 1.0;
    return result;
}

// ============================================================================
// Main SMC sampler
// ============================================================================

// The sampler walks the geometric path
//   pi_beta(theta)  proportional to  pi_0(theta) * exp(beta * h(theta)),
// beta from 0 to 1, where pi_0 is the distribution the initial population
// represents. Callbacks:
//   log_increment(theta)             -> h(theta), the tempering direction
//   initial_sample(theta, rng, beta) -> fill theta with a draw from q
//   mcmc_mutation(theta, beta, rng)  -> one MCMC step invariant for pi_beta
//   initial_log_weight(theta)        -> optional log(pi_0 / q); empty means
//                                       the draws already represent pi_0
//   prepare_mutation(particles, beta)-> optional hook fired once per
//                                       temperature, after resampling and
//                                       before the mutations, so a kernel can
//                                       adapt to the current population
//
// The two paths a caller composes from these:
//   * prior path   : q = pi_0 = p(theta), h = log L. Needs exact prior draws.
//   * reference path: q = pi_0 = any normalized reference r(theta),
//                    h = log p + log L - log r, so pi_1 = p L / Z whatever r
//                    is. The one path open to a prior with no generic sampler.
// Starting from draws that do NOT represent pi_0 and tempering in L alone is
// not a path to the posterior at all: the weights assume pi_0 and the kernel
// is invariant for another distribution, and what comes out is neither
// (gcol33/tulpa#876 -- the RE scale collapsed 3-10x that way).
//
// log_marginal_likelihood estimates log(integral pi_0 exp(h)) and is the
// evidence only when pi_0 and exp(h) carry every normalizing constant.

inline SMCResult smc_sample(
    const std::function<double(const std::vector<double>&)>& log_increment,
    const std::function<void(std::vector<double>&, std::mt19937&, double)>& initial_sample,
    const std::function<void(std::vector<double>&, double, std::mt19937&)>& mcmc_mutation,
    int dim,
    int n_particles = 1000,
    double ess_threshold = 0.5,
    int n_mcmc_steps = 5,
    unsigned int seed = 42,
    const std::function<double(const std::vector<double>&)>& initial_log_weight = nullptr,
    const std::function<void(const std::vector<std::vector<double>>&, double)>&
        prepare_mutation = nullptr
) {
    int N = n_particles;
    std::mt19937 rng(seed);

    SMCResult result;
    result.log_marginal_likelihood = 0.0;
    result.n_resamples = 0;
    result.n_mutations = 0;
    result.temperatures.push_back(0.0);

    // ------------------------------------------------------------------
    // 1. Initialize: draw N particles from q
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> particles(N, std::vector<double>(dim));
    for (int n = 0; n < N; n++) {
        initial_sample(particles[n], rng, 0.0);
    }

    // Tempering direction h at each particle, refreshed after every mutation.
    std::vector<double> incr(N);
    for (int n = 0; n < N; n++) {
        incr[n] = log_increment(particles[n]);
    }

    double beta = 0.0;
    double ess_target = ess_threshold * N;

    // Reweight by exp(log_w), accumulate the normalizing-constant increment,
    // and resample. Resampling at every step leaves the particles equally
    // weighted, which is exactly the assumption the equal-weight temperature
    // search (find_next_temperature), the log-mean-weight Z increment, and the
    // uniform final weights all rely on. Adaptive resampling (only when
    // ess < target) would instead require carrying normalized weights across
    // rounds and feeding them back into all three. Resampling unconditionally
    // keeps these consistent (Del Moral et al. 2006); the modest extra Monte
    // Carlo variance is offset by the per-particle MCMC mutations that follow.
    auto reweight_resample = [&](const std::vector<double>& log_w) {
        double max_lw = *std::max_element(log_w.begin(), log_w.end());
        double sum_w = 0.0;
        for (int n = 0; n < N; n++) sum_w += std::exp(log_w[n] - max_lw);
        result.log_marginal_likelihood += max_lw + std::log(sum_w / N);
        std::vector<double> weights(N);
        for (int n = 0; n < N; n++) {
            weights[n] = std::exp(log_w[n] - max_lw) / sum_w;
        }
        result.ess_history.push_back(compute_ess(log_w));

        auto ancestors = systematic_resample(weights, N, rng);
        auto old_particles = particles;
        auto old_incr = incr;
        for (int n = 0; n < N; n++) {
            particles[n] = old_particles[ancestors[n]];
            incr[n] = old_incr[ancestors[n]];
        }
        result.n_resamples++;
    };

    auto mutate = [&]() {
        if (prepare_mutation) prepare_mutation(particles, beta);
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < n_mcmc_steps; k++) {
                mcmc_mutation(particles[n], beta, rng);
            }
            incr[n] = log_increment(particles[n]);
        }
        result.n_mutations += N * n_mcmc_steps;
    };

    // A population drawn from q but meant to represent pi_0 != q is corrected
    // once, up front, by importance weights pi_0 / q, then moved at beta = 0.
    if (initial_log_weight) {
        std::vector<double> log_w0(N);
        for (int n = 0; n < N; n++) log_w0[n] = initial_log_weight(particles[n]);
        if (!std::isfinite(*std::max_element(log_w0.begin(), log_w0.end()))) {
            throw std::runtime_error(
                "SMC: no initial particle carries a finite pi_0 / q weight.");
        }
        reweight_resample(log_w0);
        mutate();
    }

    // ------------------------------------------------------------------
    // 2. Tempering loop
    // ------------------------------------------------------------------
    int n_steps = 0;
    while (beta < 1.0) {
        Rcpp::checkUserInterrupt();
        if (++n_steps > kMaxTemperatureSteps) {
            throw std::runtime_error(
                "SMC tempering did not reach beta = 1 within " +
                std::to_string(kMaxTemperatureSteps) + " steps (stalled at "
                "beta = " + std::to_string(beta) + "). The ESS target is "
                "unreachable at every temperature above it, which a particle "
                "carrying a non-finite log-likelihood produces; check the "
                "initial population and the likelihood at it.");
        }

        // (a) Adaptive temperature selection
        double beta_new = find_next_temperature(incr, beta, ess_target, N);
        double delta = beta_new - beta;

        // (b) Incremental log-weights, reweight and resample
        std::vector<double> log_w(N);
        for (int n = 0; n < N; n++) log_w[n] = delta * incr[n];
        reweight_resample(log_w);

        // (c) MCMC mutations at the new temperature
        beta = beta_new;
        result.temperatures.push_back(beta);
        mutate();
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
