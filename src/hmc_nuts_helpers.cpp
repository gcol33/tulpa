// hmc_nuts_helpers.cpp
// NUTS helper: log-sum-exp.

#include <algorithm>
#include <cmath>
#include <vector>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =====================================================================
// NUTS (No-U-Turn Sampler) helper functions
// =====================================================================

double nuts_log_sum_exp(double a, double b) {
  double m = std::max(a, b);
  if (!std::isfinite(m)) return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

}  // namespace tulpa_hmc
