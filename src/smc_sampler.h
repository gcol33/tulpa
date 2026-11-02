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
    // Monte Carlo variance of log_marginal_likelihood: the sum over
    // reweighting steps of 1 / ESS_t - 1 / N, the variance each step's log
    // mean incremental weight carries when the particles it is averaged over
    // are independent draws (Chopin 2004, Theorem 1, at perfectly mixing
    // kernels).
    double log_marginal_likelihood_var;
    std::vector<double> temperatures;             // path coordinate schedule (0 .. n_legs)
    int n_temperatures;                           // number of tempering steps
    // The equally weighted population at the end of every leg but the last,
    // after its mutations: a sample of that leg's end distribution.
    std::vector<std::vector<std::vector<double>>> leg_end_particles;
    std::vector<double> ess_history;              // ESS at each step
    int n_resamples;                              // how many times resampling triggered
    int n_mutations;                              // total MCMC mutation steps applied
    // Move steps each mutation phase ran (one entry per phase: the beta = 0
    // move of an importance-corrected start, then one per tempering step).
    std::vector<int> moves_history;
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
// When to stop moving: the population autocorrelation rule
// ============================================================================
//
// Buchholz, Chopin and Jacob (2021, Bayesian Analysis 16(3), Section 2.1 (b)
// and Algorithm 3): after each move step, measure for every coordinate j the
// correlation, across particles, between f(x_i(j)) where the move phase
// started and f(x_i(j)) now, with f(x) = x + x^2 so that the first two
// moments are both tracked, and apply the kernel again while it is above
// kSmcMoveCorrelationThreshold (their 0.1). The population has just been
// resampled, so each surviving ancestor appears as several identical copies;
// the moves are what make them distinct, and the rule asks for enough of them
// that the copies no longer remember their common start.
//
// Three choices within that rule, each because the alternative stops early on
// a hierarchical posterior:
//   * f reads each coordinate standardized by the population's mean and SD at
//     the start. x + x^2 is not location-invariant: its slope 1 + 2x vanishes
//     at x = -1/2, and a coordinate sitting there stops being seen to move.
//   * The correlation with the start is measured directly. The paper
//     approximates it by the product of the lag-one correlations, which equals
//     it when a coordinate decorrelates geometrically. A random-effect scale
//     moved along the funnel of its non-centered effects does not: its lag-one
//     correlation drops fast while its correlation with the start lingers, and
//     the product reads 0.1 when the start correlation is still near 0.4.
//   * Every coordinate has to pass. The paper continues while at least 10% of
//     them are above the threshold, which by construction lets a minority
//     through, and in a hierarchical model the slowest coordinate is one
//     scale among many latent values.
// On a gaussian random-intercept model (15 groups, 1000 particles, 20 seeds)
// the log evidence sat 0.30 nats below quadrature with the paper's product
// and 10% rule, 0.17 with the product over every coordinate, 0.15 with the
// direct correlation and the 10% rule, and 0.06 with both changes, its SD then
// matching the reported Monte Carlo SE.
const double kSmcMoveCorrelationThreshold = 0.1;

class MoveDecorrelation {
public:
    explicit MoveDecorrelation(const std::vector<std::vector<double>>& particles)
        : N_(static_cast<int>(particles.size())),
          d_(particles.empty() ? 0 : static_cast<int>(particles[0].size())),
          mu_(d_, 0.0), sd_(d_, 1.0) {
        for (int j = 0; j < d_; j++) {
            double m = 0.0, ss = 0.0;
            for (int i = 0; i < N_; i++) m += particles[i][j];
            m /= N_;
            for (int i = 0; i < N_; i++) {
                const double r = particles[i][j] - m;
                ss += r * r;
            }
            mu_[j] = m;
            if (ss > 0.0) sd_[j] = std::sqrt(ss / N_);
        }
        start_ = stat(particles);
    }

    // True once every coordinate's correlation with the start is at or below
    // the threshold. A coordinate whose statistic carries no spread across the
    // population, at the start or now, has no defined correlation and counts
    // as not yet decorrelated.
    bool decorrelated(const std::vector<std::vector<double>>& particles) const {
        const std::vector<double> cur = stat(particles);
        for (int j = 0; j < d_; j++) {
            if (correlation(start_, cur, j) > kSmcMoveCorrelationThreshold) {
                return false;
            }
        }
        return true;
    }

private:
    int N_, d_;
    std::vector<double> mu_, sd_;
    std::vector<double> start_;  // [N * d], particle-major

    std::vector<double> stat(const std::vector<std::vector<double>>& particles) const {
        std::vector<double> f(static_cast<size_t>(N_) * d_);
        for (int i = 0; i < N_; i++)
            for (int j = 0; j < d_; j++) {
                const double z = (particles[i][j] - mu_[j]) / sd_[j];
                f[static_cast<size_t>(i) * d_ + j] = z + z * z;
            }
        return f;
    }

    double correlation(const std::vector<double>& a, const std::vector<double>& b,
                       int j) const {
        double ma = 0.0, mb = 0.0;
        for (int i = 0; i < N_; i++) {
            ma += a[static_cast<size_t>(i) * d_ + j];
            mb += b[static_cast<size_t>(i) * d_ + j];
        }
        ma /= N_;
        mb /= N_;
        double saa = 0.0, sbb = 0.0, sab = 0.0;
        for (int i = 0; i < N_; i++) {
            const double da = a[static_cast<size_t>(i) * d_ + j] - ma;
            const double db = b[static_cast<size_t>(i) * d_ + j] - mb;
            saa += da * da;
            sbb += db * db;
            sab += da * db;
        }
        if (!(saa > 0.0 && sbb > 0.0)) return 1.0;
        const double r = sab / std::sqrt(saa * sbb);
        return std::isfinite(r) ? r : 1.0;
    }
};


// ============================================================================
// The two-leg path through a tempered posterior
// ============================================================================
//
// A path that passes through p L^b on its way to the posterior p L, as a
// function of the path coordinate tau. From a reference q the target at tau is
//   (1 - s) log q + s log p + s lambda log L,
// with leg 1 (tau in [0, 1]) at s = tau, lambda = b, bridging q to p L^b, and
// leg 2 (tau in (1, 2]) at s = 1, lambda = b + (tau - 1)(1 - b), tempering the
// likelihood from b to 1. At b = 1 there is one leg, s = tau and lambda = 1:
// the plain bridge q^(1 - tau) (p L)^tau.

struct PathPoint {
    double s;        // weight moved off the reference onto p L^lambda
    double lambda;   // likelihood power
};

inline PathPoint path_point(double tau, double b) {
    if (tau <= 1.0) return {tau, b};
    return {1.0, b + (tau - 1.0) * (1.0 - b)};
}

// The likelihood power at tau when the population represents p from the start
// and only L is tempered: b tau on leg 1, then from b to 1 on leg 2.
inline double prior_path_power(double tau, double b) {
    if (tau <= 1.0) return tau * b;
    return b + (tau - 1.0) * (1.0 - b);
}

// ============================================================================
// Main SMC sampler
// ============================================================================

// The sampler walks a path of geometric legs. Leg k runs
//   pi_{k,t}(theta)  proportional to  pi_{k,0}(theta) * exp(t * h_k(theta)),
// t from 0 to 1, where pi_{0,0} is the distribution the initial population
// represents and pi_{k+1,0} = pi_{k,1}, so every leg starts where the last one
// ended. The path coordinate tau = k + t runs from 0 to the number of legs and
// is what the kernel and the schedule see; a single leg is the plain tempering
// path with tau = beta. Callbacks:
//   legs[k](theta)                   -> h_k(theta), leg k's tempering direction
//   initial_sample(theta, rng, tau)  -> fill theta with a draw from q
//   mcmc_mutation(theta, tau, rng, i) -> one MCMC step invariant for pi at
//                                       tau on particle i (the index lets a
//                                       kernel keep per-particle state)
//   initial_log_weight(theta)        -> optional log(pi_0 / q); empty means
//                                       the draws already represent pi_0
//   prepare_mutation(particles, tau) -> optional hook fired once per
//                                       temperature, after resampling and
//                                       before the mutations, so a kernel can
//                                       adapt to the current population
//
// Every mutation phase moves the whole population one step at a time, at
// least min_moves and at most max_moves steps, and stops in between once
// MoveDecorrelation says the resampled copies have decorrelated from their
// start (min_moves == max_moves is a fixed count).
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
// log_marginal_likelihood estimates log(integral pi_0 exp(sum_k h_k)), the
// evidence up to whatever parameter-independent constants pi_0 and the h_k
// leave out (a constant c in a leg's h scales that leg's integral by exp(c)
// and moves nothing else).

SMCResult smc_sample(
    const std::vector<std::function<double(const std::vector<double>&)>>& legs,
    const std::function<void(std::vector<double>&, std::mt19937&, double)>& initial_sample,
    const std::function<void(std::vector<double>&, double, std::mt19937&, int)>& mcmc_mutation,
    int dim,
    int n_particles,
    double ess_threshold,
    int min_moves,
    int max_moves,
    unsigned int seed,
    const std::function<double(const std::vector<double>&)>& initial_log_weight = nullptr,
    const std::function<void(const std::vector<std::vector<double>>&, double)>&
        prepare_mutation = nullptr
);

} // namespace tulpa_smc

#endif
