#ifndef TULPA_SMC_INTERWEAVE_H
#define TULPA_SMC_INTERWEAVE_H

#include "hmc_gp.h"
#include "smc_modeldata.h"
#include "slice_univariate.h"

namespace tulpa {

inline bool smc_nngp_noncentered(const ModelData& data, const ParamLayout& layout) {
    return layout.is_gp && !layout.is_gp_collapsed && data.has_gp &&
        data.gp_parameterization == 1;
}

inline double smc_nngp_standardize(const double* field, double mean,
    const tulpa_gp::NNGPNCView& view, const tulpa_gp::NNGPNCWorkspace& ws,
    std::vector<double>& theta, int start) {
    double logdet = 0.0;
    for (int i = 0; i < view.n_obs; ++i) {
        const int loc = view.nn_order[i];
        double residual = field[loc] - mean;
        for (int j = 0; j < ws.B_n_nb[i]; ++j)
            residual -= ws.B_flat[i * view.nn + j] *
                (field[ws.nb_idx_flat[i * view.nn + j]] - mean);
        theta[start + loc] = residual / ws.sqrt_d[i];
        logdet += std::log(ws.sqrt_d[i]);
    }
    return logdet;
}

// The physical-field conditional at fixed covariance and other latent blocks.
// The inverse Vecchia map is affine here, so its gradient is its transpose
// applied to the standardized-coordinate gradient.
struct SmcNngpFieldMap {
    std::vector<double> base;
    std::vector<int> beta;
    int start;
    tulpa_gp::NNGPNCView view;
    tulpa_gp::NNGPNCWorkspace ws;

    SmcNngpFieldMap(const std::vector<double>& theta, const ModelData& data,
                    const ParamLayout& layout, const std::vector<int>& indices)
        : base(theta), beta(indices), start(layout.gp_w_start),
          view(tulpa_gp::make_gp_nc_view(data.gp_data)) {
        tulpa_gp::nngp_nc_forward(&theta[start],
            std::exp(theta[layout.log_sigma2_gp_idx]),
            std::exp(theta[layout.log_phi_gp_idx]), view, ws);
    }

    std::vector<double> physical() const {
        std::vector<double> x(beta.size() + view.n_obs);
        for (size_t j = 0; j < beta.size(); ++j) x[j] = base[beta[j]];
        std::copy(ws.w.begin(), ws.w.end(), x.begin() + beta.size());
        return x;
    }

    double standardize(const std::vector<double>& x, std::vector<double>& theta) const {
        theta = base;
        for (size_t j = 0; j < beta.size(); ++j) theta[beta[j]] = x[j];
        return smc_nngp_standardize(x.data() + beta.size(), 0.0, view, ws,
                                    theta, start);
    }

    template<class Target>
    double value_gradient(const std::vector<double>& x, double tau,
                           const Target& target, std::vector<double>& grad) const {
        std::vector<double> theta, full_gradient;
        const double logdet = standardize(x, theta);
        const double lp = target(theta, tau, full_gradient);
        grad.assign(x.size(), 0.0);
        for (size_t j = 0; j < beta.size(); ++j) grad[j] = full_gradient[beta[j]];
        for (int i = 0; i < view.n_obs; ++i) {
            const int loc = view.nn_order[i];
            const double g = full_gradient[start + loc] / ws.sqrt_d[i];
            grad[beta.size() + loc] += g;
            for (int j = 0; j < ws.B_n_nb[i]; ++j)
                grad[beta.size() + ws.nb_idx_flat[i * view.nn + j]] -=
                    ws.B_flat[i * view.nn + j] * g;
        }
        return lp - logdet;
    }
};

// A centred conditional HMC step in the interweaving composition of Yu and
// Meng (2011). It uses the same population-metric HMC as the ancillary step.
struct SmcNngpFieldHmc {
    bool active;
    std::vector<int> beta;
    SmcPopulationHmc hmc;

    static std::vector<int> beta_indices(const ParamLayout& layout) {
        std::vector<int> indices;
        for (size_t p = 0; p < layout.process_beta_start.size(); ++p)
            for (int j = 0; j < layout.process_beta_count[p]; ++j)
                indices.push_back(layout.process_beta_start[p] + j);
        return indices;
    }

    SmcNngpFieldHmc(const ModelData& data, const ParamLayout& layout)
        : active(smc_nngp_noncentered(data, layout)), beta(beta_indices(layout)),
          hmc(active ? beta.size() + data.gp_data.n_obs : 0) {}

    void adapt(const std::vector<std::vector<double>>& particles,
                const ModelData& data, const ParamLayout& layout) {
        if (!active) return;
        std::vector<std::vector<double>> physical;
        physical.reserve(particles.size());
        for (const auto& theta : particles)
            physical.push_back(SmcNngpFieldMap(theta, data, layout, beta).physical());
        hmc.adapt(physical);
    }

    template<class Target>
    void step(std::vector<double>& theta, double tau, std::mt19937& rng,
               int slot, const ModelData& data, const ParamLayout& layout,
               const Target& target) {
        if (!active) return;
        const SmcNngpFieldMap map(theta, data, layout, beta);
        auto physical = map.physical();
        auto conditional = [&](const std::vector<double>& x, double t,
                                std::vector<double>& grad) {
            return map.value_gradient(x, t, target, grad);
        };
        // Covariance and other fixed blocks can change between calls while
        // the physical coordinates stay equal, so their cached target expires.
        if (slot < static_cast<int>(hmc.cache.size())) hmc.cache[slot].valid = false;
        hmc.step(physical, tau, rng, conditional, slot);
        map.standardize(physical, theta);
    }
};

inline int smc_gp_level_index(const ModelData& data, const ParamLayout& layout) {
    if (data.n_processes != 1 || layout.process_beta_count.empty() ||
        layout.process_beta_count[0] == 0) return -1;
    if (!data.sharing.spatial.empty() && !data.sharing.spatial[0]) return -1;
    const auto& p = data.processes[0];
    for (int i = 0; i < data.N; ++i) {
        if (p.X_flat[i * p.p] != 1.0) return -1;
    }
    return layout.process_beta_start[0];
}

// Yu and Meng (2011), JCGS 20:531-570, DOI:10.1198/jcgs.2011.203main:
// interweave the ancillary standardized field with the sufficient physical
// field. Covariance moves at fixed w follow z = L(theta)^-1 w; the intercept
// move holds w + beta0 fixed, the sufficient augmentation for the level.
// The transformed density includes -log|L| for the WHOLE bridge target,
// including its reference component at intermediate temperatures.
template<class LogTarget>
void smc_interweave_nngp(std::vector<double>& theta, double tau,
                         std::mt19937& rng, const ModelData& data,
                         const ParamLayout& layout, const LogTarget& target) {
    if (!smc_nngp_noncentered(data, layout)) return;
    const auto view = tulpa_gp::make_gp_nc_view(data.gp_data);
    tulpa_gp::NNGPNCWorkspace ws;
    tulpa_gp::nngp_nc_forward(&theta[layout.gp_w_start],
        std::exp(theta[layout.log_sigma2_gp_idx]),
        std::exp(theta[layout.log_phi_gp_idx]), view, ws);
    const int level = smc_gp_level_index(data, layout);
    std::vector<double> signal = ws.w;
    if (level >= 0) for (double& value : signal) value += theta[level];
    std::vector<double> proposal(theta), zero(view.n_obs, 0.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    auto uniform = [&]() {
        return std::max(unif(rng), std::numeric_limits<double>::min());
    };
    std::vector<int> coordinates = {layout.log_sigma2_gp_idx, layout.log_phi_gp_idx};
    if (level >= 0) coordinates.push_back(level);
    for (int idx : coordinates) {
        auto conditional = [&](double value) {
            proposal = theta;
            proposal[idx] = value;
            const double sigma2 = std::exp(proposal[layout.log_sigma2_gp_idx]);
            const double range = std::exp(proposal[layout.log_phi_gp_idx]);
            if (!std::isfinite(sigma2) || !std::isfinite(range) ||
                sigma2 <= 0.0 || range <= 0.0)
                return -std::numeric_limits<double>::infinity();
            if (idx != level)
                tulpa_gp::nngp_nc_forward(zero.data(), sigma2, range, view, ws);
            const double mean = level >= 0 ? proposal[level] : 0.0;
            const double logdet = smc_nngp_standardize(signal.data(), mean,
                view, ws, proposal, layout.gp_w_start);
            return target(proposal, tau) - logdet;
        };
        const double value = slice_univariate(theta[idx], conditional, uniform);
        conditional(value);
        theta = proposal;
    }
}

} // namespace tulpa
#endif
