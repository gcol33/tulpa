// hmc_nuts_find_epsilon.cpp
// Stan-style find_reasonable_epsilon (identity / diagonal / dense mass).

#include <cmath>
#include <random>
#include <vector>

#include "hmc_sampler.h"
#include "linalg_fast.h"

namespace tulpa_hmc {

// =====================================================================
// Unified find_reasonable_epsilon: handles identity, diagonal, and dense mass
// Stan-style algorithm: start at epsilon=1, double or halve until
// acceptance probability crosses 0.5
// =====================================================================

// Helper: compute kinetic energy with mass matrix
static inline double kinetic_energy_mass(
    const double* p, int n,
    const double* inv_mass,          // nullptr = identity
    const DenseMassMatrix* dense_mass // nullptr = not dense
) {
    if (dense_mass) return dense_mass->kinetic_energy(p);
    if (inv_mass) {
        double ke = 0.0;
        for (int i = 0; i < n; i++) ke += p[i] * p[i] * inv_mass[i];
        return 0.5 * ke;
    }
    return 0.5 * tulpa_linalg::norm_squared(p, n);
}

// One integration step respecting the mass matrix type, walking the ACTIVE SIMP
// scheme's op sequence (the same one the trajectory integrator leapfrog_step
// walks) so the seed epsilon is calibrated for the scheme the sampler actually
// uses -- yoshida4/6/8, minerror2, adaptive*, mts -- not a fixed first-order
// leapfrog. leapfrog_step handles the identity/diagonal case; this variant adds
// the dense-mass drift branch, so the two share one op-sequence contract.
static inline LeapfrogResult leapfrog_for_epsilon(
    const std::vector<double>& q, const std::vector<double>& p,
    double epsilon, const ModelData& data, const ParamLayout& layout,
    const double* inv_mass, const DenseMassMatrix* dense_mass
) {
    int n = q.size();
    LeapfrogResult result;
    result.q = q;
    result.p = p;
    result.divergent = false;

    const simp::Scheme& scheme = get_integrator_scheme();
    int last_kick = -1;
    for (int j = 0; j < static_cast<int>(scheme.ops.size()); j++) {
        if (scheme.ops[j].first == simp::Op::Kick) last_kick = j;
    }

    std::vector<double> grad(n);
    std::vector<double> Mp(n);
    for (int j = 0; j < static_cast<int>(scheme.ops.size()); j++) {
        double c = scheme.ops[j].second * epsilon;
        if (scheme.ops[j].first == simp::Op::Kick) {
            if (j == last_kick) {
                compute_gradient(result.q, data, layout, grad, &result.log_prob);
            } else {
                compute_gradient(result.q, data, layout, grad);
            }
            for (int i = 0; i < n; i++) result.p[i] += c * grad[i];
        } else {  // Drift: q += c * M^{-1} * p
            if (dense_mass) {
                dense_mass->inv_mass_times_p(result.p.data(), Mp.data());
                for (int i = 0; i < n; i++) result.q[i] += c * Mp[i];
            } else if (inv_mass) {
                for (int i = 0; i < n; i++) result.q[i] += c * inv_mass[i] * result.p[i];
            } else {
                for (int i = 0; i < n; i++) result.q[i] += c * result.p[i];
            }
        }
    }

    if (!std::isfinite(result.log_prob)) result.divergent = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
            result.divergent = true;
            break;
        }
    }
    return result;
}

// Unified find_reasonable_epsilon: works with identity, diagonal, or dense mass.
// inv_mass_diag = nullptr for identity/dense, mass_dense = nullptr for identity/diagonal.
double find_reasonable_epsilon_impl(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const double* inv_mass_diag,
    const DenseMassMatrix* mass_dense
) {
    int n = q.size();
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> p(n);

    // Sample momentum based on mass type
    if (mass_dense) {
        const_cast<DenseMassMatrix*>(mass_dense)->sample_momentum(p.data(), rng);
    } else if (inv_mass_diag) {
        for (int i = 0; i < n; i++) p[i] = normal(rng) / std::sqrt(inv_mass_diag[i]);
    } else {
        for (int i = 0; i < n; i++) p[i] = normal(rng);
    }

    double log_prob_init;
    std::vector<double> grad_init(n);
    compute_gradient(q, data, layout, grad_init, &log_prob_init);
    double H_init = -log_prob_init + kinetic_energy_mass(p.data(), n, inv_mass_diag, mass_dense);

    double epsilon = 1.0;
    auto lf = leapfrog_for_epsilon(q, p, epsilon, data, layout, inv_mass_diag, mass_dense);

    double delta_H = (-lf.log_prob + kinetic_energy_mass(lf.p.data(), n, inv_mass_diag, mass_dense)) - H_init;

    int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;
    for (int iter = 0; iter < 50; iter++) {
        epsilon *= (direction == 1) ? 2.0 : 0.5;
        if (epsilon < 1e-10 || epsilon > 1e5) break;
        lf = leapfrog_for_epsilon(q, p, epsilon, data, layout, inv_mass_diag, mass_dense);
        double lp_try = lf.log_prob;
        if (!std::isfinite(lp_try)) { if (direction == 1) break; continue; }
        delta_H = (-lp_try + kinetic_energy_mass(lf.p.data(), n, inv_mass_diag, mass_dense)) - H_init;
        if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
        if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
    }
    return std::max(1e-10, std::min(epsilon, 1e3));
}

// Backward-compatible overloads (delegate to impl)
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
) {
    return find_reasonable_epsilon_impl(q, data, layout, rng, nullptr, nullptr);
}

double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
) {
    return find_reasonable_epsilon_impl(q, data, layout, rng, inv_mass.data(), nullptr);
}

double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
) {
    return find_reasonable_epsilon_impl(q, data, layout, rng, nullptr, &mass);
}

}  // namespace tulpa_hmc
