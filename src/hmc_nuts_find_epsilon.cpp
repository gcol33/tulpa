#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// Stan's init_stepsize brackets a one-step acceptance of 0.8, refreshing
// momentum under the active metric at each trial and restoring the position.
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass,
    double epsilon
) {
    std::vector<double> p(q.size());
    auto energy_error = [&]() {
        mass.sample_momentum(p.data(), rng);
        double log_prob;
        std::vector<double> grad(q.size());
        compute_gradient(q, data, layout, grad, &log_prob);
        const double H0 = -log_prob + mass.kinetic_energy(p.data());
        const auto lf = leapfrog_step(q, p, epsilon, data, layout, nullptr, &mass);
        const double H1 = -lf.log_prob + mass.kinetic_energy(lf.p.data());
        const double delta = H0 - H1;
        return std::isfinite(delta) ? delta : -INFINITY;
    };
    const double threshold = std::log(0.8);
    const int direction = energy_error() > threshold ? 1 : -1;
    for (int trial = 0; trial < 100; ++trial) {
        const double delta = energy_error();
        if ((direction == 1 && delta <= threshold) ||
            (direction == -1 && delta >= threshold)) return epsilon;
        epsilon *= direction == 1 ? 2.0 : 0.5;
        if (!(epsilon > 0.0) || !std::isfinite(epsilon) || epsilon > 1e7)
            throw std::runtime_error("No finite reasonable step size could be found");
    }
    throw std::runtime_error("Step-size search did not bracket the acceptance threshold");
}

}  // namespace tulpa_hmc
