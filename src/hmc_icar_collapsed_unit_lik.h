// hmc_icar_collapsed_unit_lik.h
// Per-unit likelihood + score + obs Hessian (compute_unit_lik).
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_UNIT_LIK_H
#define TULPA_HMC_ICAR_COLLAPSED_UNIT_LIK_H

#include <algorithm>
#include <cmath>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =========================================================================
// Per-spatial-unit likelihood (aggregated over observations at each unit)
// =========================================================================

struct UnitLikResult {
    double ll;          // log-likelihood contribution
    double grad;        // d(ll)/d(spatial_effect)
    double neg_hess;    // -d²(ll)/d(spatial_effect)²
};

// Compute data log-likelihood, gradient, and Hessian at spatial unit s,
// where spatial_eff[s] enters eta for all observations at unit s.
// RE effects are included via re_vals (pre-computed actual RE values).
// Fast version: uses pre-filtered obs list instead of scanning all N observations.
inline UnitLikResult compute_unit_lik(
    int s,                      // spatial unit index (0-based)
    double spatial_eff,         // spatial effect at this unit
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const double* re_vals,      // pre-computed RE values (actual, not z), length n_re_groups or NULL
    const ModelData& data,
    bool is_binomial,
    const int* obs_list = nullptr,  // pre-filtered obs indices at this unit (or NULL)
    int n_obs = -1                  // number of obs at this unit (-1 = scan all)
) {
    UnitLikResult res = {0.0, 0.0, 0.0};
    int N = (n_obs >= 0) ? n_obs : data.N;

    for (int idx = 0; idx < N; idx++) {
        int i = (obs_list != nullptr) ? obs_list[idx] : idx;
        if (obs_list == nullptr && data.spatial_group[i] - 1 != s) continue;

        // Compute eta
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }

        // Add spatial effect
        eta_num_i += spatial_eff;
        if (!is_binomial) eta_denom_i += spatial_eff;  // shared by default

        // Add RE
        if (re_vals != nullptr && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            double re_val = re_vals[data.re_group[i] - 1];
            eta_num_i += re_val;
            if (!is_binomial) eta_denom_i += re_val;
        }

        // Per-family likelihood, gradient, Hessian
        double mu_num = std::exp(std::min(eta_num_i, 20.0));
        int y_num = data.legacy.y_num[i];

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(std::max(y_denom, 1e-10))
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += shape * (y_denom / mu_denom - 1.0);
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    int y_denom = (int)data.legacy.y_denom[i];
                    double r_denom = phi_denom;
                    res.ll += std::lgamma(y_denom + r_denom) - std::lgamma(r_denom) - std::lgamma(y_denom + 1)
                              + y_denom * eta_denom_i - (y_denom + r_denom) * std::log(mu_denom + r_denom)
                              + r_denom * std::log(r_denom);
                    double resid_denom = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                    res.grad += resid_denom;
                    res.neg_hess += mu_denom * r_denom * (y_denom + r_denom) / ((mu_denom + r_denom) * (mu_denom + r_denom));
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));
                // Gamma denominator
                {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(std::max(y_denom, 1e-10))
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += shape * (y_denom / mu_denom - 1.0);
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                res.ll += y_num * eta_num_i - n_trials * std::log(1.0 + std::exp(eta_num_i));
                res.grad += y_num - n_trials * p_i;
                res.neg_hess += n_trials * p_i * (1.0 - p_i);
                break;
            }
            default:
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                break;
        }
    }
    return res;
}

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_UNIT_LIK_H
