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

// Single SGHMC update, Chen, Fox & Guestrin (2014) eq. (15) written in the
// momentum r = v / epsilon with mass M = I:
//
//   theta_{t+1} = theta_t + epsilon * r_t
//   r_{t+1}     = (1 - alpha) * r_t + epsilon * grad log p(theta_t)
//                 + sqrt(2 * alpha) * N(0, 1)
//
// The paper's position increment is v = epsilon * r and its learning rate is
// eta = epsilon^2 M^-1. Multiplying the momentum recursion through by epsilon
// gives v <- (1 - alpha) v + eta * grad + N(0, 2 * alpha * eta) term for term:
// the gradient term carries eta = epsilon^2, and the injected variance is
// epsilon^2 * 2 * alpha, which is 2 * alpha * eta.
//
// Fluctuation-dissipation then fixes the sampled temperature at 1. The
// recursion's stationary momentum variance is
// 2 * alpha / (1 - (1 - alpha)^2) = 2 / (2 - alpha), which is 1 to O(alpha) --
// the same N(0, I) the sampler resamples r from at the top of each iteration.
// Injecting sqrt(2 * alpha * epsilon) instead puts the chain at temperature
// epsilon, i.e. samples p(theta)^(1/epsilon): posterior means come out roughly
// right while every posterior sd is short by sqrt(epsilon).
//
// Where:
//   theta = parameters
//   r = momentum, stationary N(0, M) = N(0, I)
//   epsilon = step size
//   alpha = epsilon * C, the dimensionless friction applied per step
//   grad log p = gradient of log posterior

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

    // Friction noise standard deviation in the r parameterization: the
    // injected variance is 2 * alpha * eta with eta = epsilon^2, and one
    // factor of epsilon is carried by the position update theta += epsilon * r.
    double noise_sd = std::sqrt(2.0 * alpha);

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

// Walks log_epsilon toward a target acceptance statistic. SGHMC takes no
// Metropolis step -- the friction term stands in for the correction -- so that
// statistic is a log-posterior ratio the sampler never accepts or rejects on,
// and it is not the quantity that bounds the step size here. What bounds it is
// the discretisation error, which grows with epsilon^2 * lambda_max of the
// posterior precision and inflates the sampled marginal SDs above the true
// posterior once that product leaves the small regime. A caller who knows the
// curvature should set config.epsilon and clear config.adapt_epsilon rather
// than let this run; epsilon_min / epsilon_max bound where it can land.
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

SGHMCResult run_sghmc_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const SGHMCConfig& config
);

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
    double epsilon;             // Step size when use_schedule is false
    bool verbose;
    int print_every;
    unsigned int seed;

    // Step size schedule: epsilon_t = a * (b + t)^(-gamma). The schedule
    // supplies the step size on every iteration, so it and `epsilon` are
    // exclusive: with use_schedule set, `epsilon` is not read at all.
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

SGLDResult run_sgld_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const SGLDConfig& config
);

} // namespace tulpa_sghmc

#endif // TULPA_SGHMC_SAMPLER_H
