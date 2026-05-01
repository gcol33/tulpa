// hmc_likelihood.h
// Legacy ratio-model observation likelihood helpers used by HMC gradients.
//
// The hand-coded gradient paths call these helpers while fusing likelihood and
// gradient work. Keep formulas in sync with compute_log_post() in
// hmc_sampler.cpp; they intentionally return a large negative finite value for
// invalid positive-support parameters to match the legacy sampler behavior.

#ifndef TULPA_HMC_LIKELIHOOD_H
#define TULPA_HMC_LIKELIHOOD_H

#include <cmath>
#include "tulpa/model_data.h"
#include "tulpa/portable_math.h"

namespace tulpa_hmc {

inline double log_lik_binomial(int y, int n, double eta) {
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  }
  return y * eta - n * std::log(1.0 + std::exp(eta));
}

inline double log_lik_negbin(int y, double mu, double phi) {
  if (mu <= 0 || phi <= 0) return -1e10;
  return std::lgamma(y + phi) - std::lgamma(phi) - std::lgamma(y + 1.0)
       + phi * std::log(phi / (mu + phi))
       + y * std::log(mu / (mu + phi));
}

inline double log_lik_poisson(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - std::lgamma(y + 1.0);
}

inline double log_lik_gamma(double y, double shape, double mu) {
  if (y <= 0 || shape <= 0 || mu <= 0) return -1e10;
  const double rate = shape / mu;
  return shape * std::log(rate) + (shape - 1.0) * std::log(y)
       - rate * y - std::lgamma(shape);
}

// Observation log-likelihood for one row of the legacy ratio model. The input
// linear predictors and dispersion parameters are already assembled by the
// caller, which lets vectorized gradients avoid a second O(N) pass.
inline double compute_obs_ll(
    const tulpa::ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom
) {
  if (data.legacy.model_type == tulpa::ModelType::BINOMIAL) {
    return log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], eta_num);
  } else if (data.legacy.model_type == tulpa::ModelType::NEGBIN_NEGBIN) {
    const double mu_num = std::exp(eta_num);
    const double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num)
         + log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
  } else if (data.legacy.model_type == tulpa::ModelType::POISSON_GAMMA) {
    const double mu_num = std::exp(eta_num);
    const double mu_denom = std::exp(eta_denom);
    return log_lik_poisson(data.legacy.y_num[i], mu_num)
         + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.legacy.model_type == tulpa::ModelType::NEGBIN_GAMMA) {
    const double mu_num = std::exp(eta_num);
    const double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num)
         + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.legacy.model_type == tulpa::ModelType::GAMMA_GAMMA) {
    const double mu_num = std::exp(eta_num);
    const double mu_denom = std::exp(eta_denom);
    return log_lik_gamma(data.legacy.y_num_cont[i], phi_num, mu_num)
         + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.legacy.model_type == tulpa::ModelType::LOGNORMAL) {
    const double log_y_num = std::log(data.legacy.y_num_cont[i]);
    const double log_y_denom = std::log(data.legacy.y_denom_cont[i]);
    const double z_num = (log_y_num - eta_num) / phi_num;
    const double z_denom = (log_y_denom - eta_denom) / phi_denom;
    return -log_y_num - std::log(phi_num) - 0.5 * z_num * z_num
           -log_y_denom - std::log(phi_denom) - 0.5 * z_denom * z_denom;
  } else if (data.legacy.model_type == tulpa::ModelType::BETA_BINOMIAL) {
    const double p = 1.0 / (1.0 + std::exp(-eta_num));
    const int y = data.legacy.y_num[i];
    const int n = data.legacy.y_denom[i];
    const double alpha = p * phi_num;
    const double beta_param = (1.0 - p) * phi_num;
    return std::lgamma(y + alpha) + std::lgamma(n - y + beta_param) - std::lgamma(n + phi_num)
         - std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num)
         + tulpa::math::portable_lchoose(n, y);
  }
  return 0.0;
}

} // namespace tulpa_hmc

#endif // TULPA_HMC_LIKELIHOOD_H
