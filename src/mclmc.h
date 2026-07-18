// mclmc.h
// Microcanonical Langevin Monte Carlo (MCLMC) and Metropolis-Adjusted MCLMC.
//
// MCLMC (unadjusted) integrates a Hamiltonian trajectory with leapfrog and
// applies partial momentum refreshment + energy-shell projection between
// trajectories. Step size is adapted in warmup so that the per-step energy
// error per dimension hits a target (target ~ 1e-3); divergent trajectories
// shrink the ceiling. Accept_rate reports (1 - divergence rate).
//
// MAMCLMC (Metropolis-adjusted) wraps the same leapfrog + refresh + project
// kernel with an MH accept/reject step on the Hamiltonian change, and adapts
// step size via dual averaging targeting acceptance ~0.9.
//
// References:
//   Robnik, De Luca, Silverstein, Seljak (2023). "Microcanonical Hamiltonian
//     Monte Carlo." JMLR.
//   Robnik & Seljak (2024). "Controlled Hamiltonian Monte Carlo / MAMCLMC."

#ifndef TULPA_MCLMC_H
#define TULPA_MCLMC_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

namespace tulpa {

// ============================================================================
// Result struct (consumed by Rcpp wrappers)
// ============================================================================
struct MCLMCResult {
    std::vector<std::vector<double>> draws;   // post-warmup samples
    std::vector<double> lp;                    // log-posterior per sample
    double accept_rate = 0.0;                  // MH rate (adjusted) or
                                               // 1 - divergence rate (unadjusted)
    double step_size_final = 0.0;              // final adapted eps
    double L_final = 0.0;                      // trajectory length used
    long long n_grad_evals = 0;                // total gradient calls
    int n_divergences = 0;                     // trajectories with |dH| > threshold
};

namespace mclmc_detail {

// ----------------------------------------------------------------------------
// Tunable constants (unadjusted MCLMC adaptation)
// ----------------------------------------------------------------------------
constexpr double kSmoothAlpha       = 0.1;     // EMA weight on observed
                                               // |dH|/d (per-iter energy err)
constexpr double kTargetEnergyRatio = 1.0e-3;  // target |dH|/d per step
constexpr double kEpsMin            = 1.0e-4;
constexpr double kEpsMax            = 5.0;
constexpr double kDivergenceThresh  = 1000.0;  // |dH| above this -> divergence

// ----------------------------------------------------------------------------
// Dual-averaging step-size adapter (Hoffman & Gelman 2014).
// Used by MAMCLMC; MCLMC uses the cube-root energy-error scheme instead.
// ----------------------------------------------------------------------------
class DualAverage {
public:
    explicit DualAverage(double eps0)
        : mu_(std::log(10.0 * std::max(1e-10, eps0))),
          gamma_(0.05),
          t0_(10.0),
          kappa_(0.75),
          log_eps_(std::log(std::max(1e-10, eps0))),
          log_eps_bar_(0.0),
          H_bar_(0.0) {}

    void update(double error_stat, int iter) {
        double m = static_cast<double>(iter);
        H_bar_ = (1.0 - 1.0 / (m + t0_)) * H_bar_
               + (1.0 / (m + t0_)) * error_stat;
        log_eps_ = mu_ - (std::sqrt(m) / gamma_) * H_bar_;
        double eta = std::pow(m, -kappa_);
        log_eps_bar_ = eta * log_eps_ + (1.0 - eta) * log_eps_bar_;
    }
    double eps()     const { return std::exp(log_eps_); }
    double eps_bar() const { return std::exp(log_eps_bar_); }
private:
    double mu_, gamma_, t0_, kappa_;
    double log_eps_, log_eps_bar_, H_bar_;
};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
inline void init_momentum(std::vector<double>& p,
                          const std::vector<double>& mass,
                          const std::vector<double>& inv_mass,
                          int dim, std::mt19937& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < dim; i++) {
        p[i] = normal(rng) * std::sqrt(mass[i]);
    }
    // Project onto microcanonical energy shell: KE = dim/2 (equipartition)
    double KE = 0.0;
    for (int i = 0; i < dim; i++) KE += p[i] * p[i] * inv_mass[i];
    if (KE > 1e-300) {
        double scale = std::sqrt(static_cast<double>(dim) / KE);
        for (int i = 0; i < dim; i++) p[i] *= scale;
    }
}

inline double kinetic_energy(const std::vector<double>& p,
                              const std::vector<double>& inv_mass, int dim) {
    double KE = 0.0;
    for (int i = 0; i < dim; i++) KE += 0.5 * p[i] * p[i] * inv_mass[i];
    return KE;
}

// One leapfrog step targeting potential U(q) = -log_prob(q).
// On entry: grad holds dlog_prob/dq at q. On exit: grad holds the new gradient.
// Returns log_prob(q_new) (free byproduct of the gradient evaluation).
inline double leapfrog_step(
    std::vector<double>& q, std::vector<double>& p, std::vector<double>& grad,
    const std::vector<double>& inv_mass, double eps, int dim,
    const std::function<double(const std::vector<double>&,
                               std::vector<double>&)>& log_prob_grad
) {
    // Half-kick (force = +grad of log_prob => +grad of log_prob is -dU/dq)
    for (int i = 0; i < dim; i++) p[i] += 0.5 * eps * grad[i];
    // Drift
    for (int i = 0; i < dim; i++) q[i] += eps * inv_mass[i] * p[i];
    // Recompute gradient at new q
    double lp = log_prob_grad(q, grad);
    // Half-kick
    for (int i = 0; i < dim; i++) p[i] += 0.5 * eps * grad[i];
    return lp;
}

// Partial momentum refreshment: p <- alpha * p + sqrt(1 - alpha^2) * noise
// where noise ~ N(0, mass). Larger alpha => more momentum persistence.
inline void partial_refresh(std::vector<double>& p,
                            const std::vector<double>& mass,
                            double alpha, int dim, std::mt19937& rng) {
    double a = std::max(0.0, std::min(1.0, alpha));
    double sqrt_1ma2 = std::sqrt(std::max(0.0, 1.0 - a * a));
    if (sqrt_1ma2 == 0.0) return;
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < dim; i++) {
        double z = normal(rng) * std::sqrt(mass[i]);
        p[i] = a * p[i] + sqrt_1ma2 * z;
    }
}

// Project momentum onto the microcanonical energy shell (KE == dim/2).
// Pure rescaling — preserves the direction of p.
inline void project_energy(std::vector<double>& p,
                            const std::vector<double>& inv_mass, int dim) {
    double KE = 0.0;
    for (int i = 0; i < dim; i++) KE += p[i] * p[i] * inv_mass[i];
    if (KE < 1e-300) return;
    double scale = std::sqrt(static_cast<double>(dim) / KE);
    for (int i = 0; i < dim; i++) p[i] *= scale;
}

// Mass-matrix setup helper used by both samplers
inline void setup_mass(int dim, const std::vector<double>& mass_diag,
                       std::vector<double>& mass,
                       std::vector<double>& inv_mass) {
    mass.assign(dim, 1.0);
    inv_mass.assign(dim, 1.0);
    if (!mass_diag.empty()) {
        for (int i = 0; i < dim; i++) {
            mass[i] = mass_diag[i];
            inv_mass[i] = 1.0 / mass_diag[i];
        }
    }
}

}  // namespace mclmc_detail

// ============================================================================
// Unadjusted MCLMC: leapfrog + partial refresh, no MH check.
// Step size adapts in warmup using the cube-root energy-error rule:
//   energy error per leapfrog step scales as eps^3, so to move
//   |dH|/d toward kTargetEnergyRatio multiply eps by (target/observed)^(1/3).
// ============================================================================
inline MCLMCResult mclmc_sample(
    const std::function<double(const std::vector<double>&,
                               std::vector<double>&)>& log_prob_grad,
    const std::vector<double>& init,
    int dim,
    int n_iter = 2000,
    int n_warmup = 1000,
    double step_size = 0.0,
    int L = 0,
    const std::vector<double>& mass_diag = {},
    unsigned int seed = 42
) {
    using namespace mclmc_detail;

    std::vector<double> mass, inv_mass;
    setup_mass(dim, mass_diag, mass, inv_mass);

    if (L <= 0) {
        // Need enough leapfrog steps per trajectory to give meaningful
        // exploration even in low dim; sqrt(d) alone gives L=1 for d=1.
        L = std::max(5, static_cast<int>(std::sqrt(static_cast<double>(dim))));
    }
    bool adapt_eps = (step_size <= 0.0);
    if (adapt_eps) {
        // Default: aggressive eps for unadjusted (no MH safety net needed)
        step_size = 0.5 / std::pow(static_cast<double>(dim), 0.25);
    }

    // Langevin partial-refresh factor: alpha = exp(-friction * step) with
    // friction ~ 1/L so the cumulative refresh per trajectory ~ exp(-1).
    // Always in (0, 1); avoids the pathological alpha=1 (no refresh).
    auto compute_alpha = [&](int traj_len) -> double {
        double a = std::exp(-1.0 / static_cast<double>(traj_len));
        return std::max(0.05, std::min(a, 0.99));
    };

    std::mt19937 rng(seed);

    std::vector<double> q(init.begin(), init.end());
    std::vector<double> p(dim);
    std::vector<double> grad(dim, 0.0);

    double lp = log_prob_grad(q, grad);
    long long n_grad_evals = 1;

    init_momentum(p, mass, inv_mass, dim, rng);

    int total_iter = n_warmup + n_iter;
    MCLMCResult result;
    result.draws.reserve(n_iter);
    result.lp.reserve(n_iter);

    int n_divergences = 0;
    double smoothed_error_ratio = kTargetEnergyRatio;
    double eps_div_min = std::numeric_limits<double>::infinity();

    std::vector<double> q_prop(dim);
    std::vector<double> p_prop(dim);
    std::vector<double> grad_prop(dim);

    for (int iter = 1; iter <= total_iter; iter++) {
        bool warmup = (iter <= n_warmup);
        double eps = step_size;
        double alpha = compute_alpha(L);

        // Always accept the leapfrog trajectory (unadjusted).
        std::copy(q.begin(), q.end(), q_prop.begin());
        std::copy(p.begin(), p.end(), p_prop.begin());
        std::copy(grad.begin(), grad.end(), grad_prop.begin());
        double lp_prop = lp;

        double H_old = -lp + kinetic_energy(p_prop, inv_mass, dim);

        for (int step = 0; step < L; step++) {
            lp_prop = leapfrog_step(q_prop, p_prop, grad_prop, inv_mass, eps,
                                    dim, log_prob_grad);
            n_grad_evals++;
        }

        double H_new = -lp_prop + kinetic_energy(p_prop, inv_mass, dim);
        double energy_error = std::abs(H_new - H_old);
        bool diverged = !std::isfinite(lp_prop) || !std::isfinite(H_new) ||
                        energy_error > kDivergenceThresh;

        if (diverged) {
            n_divergences++;
            // Record divergence-derived ceiling and shrink eps immediately.
            if (eps < eps_div_min) eps_div_min = eps;
            step_size = std::max(kEpsMin, 0.5 * step_size);
            // Restart momentum so we can recover from the diverged state
            init_momentum(p, mass, inv_mass, dim, rng);
        } else {
            std::copy(q_prop.begin(), q_prop.end(), q.begin());
            std::copy(p_prop.begin(), p_prop.end(), p.begin());
            std::copy(grad_prop.begin(), grad_prop.end(), grad.begin());
            lp = lp_prop;
        }

        // Canonical Langevin refresh — preserves p ~ N(0, mass).
        // (No energy-shell projection here: doing so would destroy the
        // canonical p-marginal that leapfrog conserves and bias the q-chain.
        // The MH-adjusted variant keeps project_energy because the MH step
        // corrects whatever the proposal kernel does.)
        partial_refresh(p, mass, alpha, dim, rng);

        // Step-size adaptation during warmup:
        // energy error scales as eps^3 for leapfrog, so adjust eps
        // by (target/observed)^{1/3} to move towards the target.
        if (warmup && adapt_eps && !diverged) {
            double observed_ratio = energy_error / static_cast<double>(dim);
            smoothed_error_ratio = kSmoothAlpha * observed_ratio
                                 + (1.0 - kSmoothAlpha) * smoothed_error_ratio;
            if (iter > 10) {
                double ratio = kTargetEnergyRatio / (smoothed_error_ratio + 1e-10);
                double factor = std::pow(std::max(0.2, std::min(ratio, 3.0)),
                                         1.0 / 3.0);
                step_size *= factor;
                step_size = std::max(kEpsMin, std::min(step_size, kEpsMax));
                if (std::isfinite(eps_div_min)) {
                    step_size = std::min(step_size, 0.7 * eps_div_min);
                }
            }
        }

        if (!warmup) {
            result.draws.push_back(q);
            result.lp.push_back(lp);
        }
    }

    // "accept_rate" for unadjusted = fraction of non-divergent trajectories.
    result.accept_rate = static_cast<double>(total_iter - n_divergences)
                         / static_cast<double>(total_iter);
    result.step_size_final = step_size;
    result.L_final = static_cast<double>(L);
    result.n_grad_evals = n_grad_evals;
    result.n_divergences = n_divergences;

    return result;
}

// ============================================================================
// MAMCLMC (adjusted with Metropolis-Hastings correction)
// ============================================================================
inline MCLMCResult mamclmc_sample(
    const std::function<double(const std::vector<double>&,
                               std::vector<double>&)>& log_prob_grad,
    const std::vector<double>& init,
    int dim,
    int n_iter = 2000,
    int n_warmup = 1000,
    double step_size = 0.0,
    int L = 0,
    const std::vector<double>& mass_diag = {},
    unsigned int seed = 42
) {
    using namespace mclmc_detail;

    std::vector<double> mass, inv_mass;
    setup_mass(dim, mass_diag, mass, inv_mass);

    if (L <= 0) {
        // Need enough leapfrog steps per trajectory to give meaningful
        // exploration even in low dim; sqrt(d) alone gives L=1 for d=1.
        L = std::max(5, static_cast<int>(std::sqrt(static_cast<double>(dim))));
    }
    bool adapt_eps = (step_size <= 0.0);
    if (adapt_eps) {
        step_size = 0.3 / std::pow(static_cast<double>(dim), 0.25);
    }

    // Langevin partial-refresh factor: alpha = exp(-friction * step) with
    // friction ~ 1/L so the cumulative refresh per trajectory ~ exp(-1).
    // Always in (0, 1); avoids the pathological alpha=1 (no refresh).
    auto compute_alpha = [&](int traj_len) -> double {
        double a = std::exp(-1.0 / static_cast<double>(traj_len));
        return std::max(0.05, std::min(a, 0.99));
    };

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    std::vector<double> q(init.begin(), init.end());
    std::vector<double> p(dim);
    std::vector<double> grad(dim, 0.0);

    double lp = log_prob_grad(q, grad);
    long long n_grad_evals = 1;

    init_momentum(p, mass, inv_mass, dim, rng);

    DualAverage da(step_size);
    double target_accept = 0.9;  // MAMCLMC targets high acceptance

    int total_iter = n_warmup + n_iter;
    MCLMCResult result;
    result.draws.reserve(n_iter);
    result.lp.reserve(n_iter);

    int n_accept = 0;
    int n_post_warmup = 0;
    int n_divergences = 0;

    std::vector<double> q_prop(dim);
    std::vector<double> p_prop(dim);
    std::vector<double> grad_prop(dim);

    for (int iter = 1; iter <= total_iter; iter++) {
        bool warmup = (iter <= n_warmup);
        double eps = warmup ? da.eps() : step_size;
        double alpha = compute_alpha(L);

        std::copy(q.begin(), q.end(), q_prop.begin());
        std::copy(p.begin(), p.end(), p_prop.begin());
        std::copy(grad.begin(), grad.end(), grad_prop.begin());
        double lp_prop = lp;

        double H_old = -lp + kinetic_energy(p_prop, inv_mass, dim);

        for (int step = 0; step < L; step++) {
            lp_prop = leapfrog_step(q_prop, p_prop, grad_prop, inv_mass, eps,
                                    dim, log_prob_grad);
            n_grad_evals++;
        }

        double H_new = -lp_prop + kinetic_energy(p_prop, inv_mass, dim);

        bool diverged = !std::isfinite(lp_prop) || !std::isfinite(H_new) ||
                        std::abs(H_new - H_old) > kDivergenceThresh;
        if (diverged) {
            n_divergences++;
        }

        double log_accept_prob = -(H_new - H_old);
        if (!std::isfinite(log_accept_prob)) log_accept_prob = -1e300;
        log_accept_prob = std::min(log_accept_prob, 0.0);
        double accept_prob = std::exp(log_accept_prob);

        bool accepted = (uniform(rng) < accept_prob);
        if (accepted) {
            std::copy(q_prop.begin(), q_prop.end(), q.begin());
            std::copy(p_prop.begin(), p_prop.end(), p.begin());
            std::copy(grad_prop.begin(), grad_prop.end(), grad.begin());
            lp = lp_prop;
        } else {
            // Generalized-HMC / partial-refresh reversibility (Horowitz 1991;
            // Neal 2011, sec. 5.3): flip the momentum on rejection so the
            // leapfrog proposal followed by the Langevin refresh leaves the
            // canonical distribution invariant.
            for (int i = 0; i < dim; i++) p[i] = -p[i];
        }

        // Canonical Langevin refresh — preserves p ~ N(0, mass).
        // (project_energy was in the original tail but biases the q-marginal
        // because it destroys the canonical p distribution that leapfrog
        // and refresh both preserve. With true microcanonical dynamics it
        // would be a fixed point; with leapfrog it is not.)
        partial_refresh(p, mass, alpha, dim, rng);

        if (warmup && adapt_eps) {
            double error_stat = target_accept - accept_prob;
            da.update(error_stat, iter);
            step_size = da.eps_bar();
        }

        if (!warmup) {
            result.draws.push_back(q);
            result.lp.push_back(lp);
            if (accepted) n_accept++;
            n_post_warmup++;
        }
    }

    result.accept_rate = (n_post_warmup > 0) ?
        static_cast<double>(n_accept) / n_post_warmup : 0.0;
    result.step_size_final = step_size;
    result.L_final = static_cast<double>(L);
    result.n_grad_evals = n_grad_evals;
    result.n_divergences = n_divergences;

    return result;
}

}  // namespace tulpa

#endif  // TULPA_MCLMC_H
