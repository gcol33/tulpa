// pc_prior.h
// Penalized-complexity (PC) prior on a variance parameter. Single source of
// truth shared by the GP and temporal hyperparameter priors.

#ifndef TULPA_PC_PRIOR_H
#define TULPA_PC_PRIOR_H

#include <cmath>

namespace tulpa {

// PC prior log-density for a variance sigma2. The PC prior places an
// exponential prior on sigma = sqrt(sigma2) with rate -log(alpha)/U (so that
// P(sigma > U) = alpha) and transforms to sigma2 via the Jacobian
// d(sigma)/d(sigma2) = 1/(2*sigma).
inline double pc_prior_log_sigma2(double sigma2, double U, double alpha) {
  double rate = -std::log(alpha) / U;
  double sigma = std::sqrt(sigma2);
  return std::log(rate) - rate * sigma - std::log(2.0 * sigma);
}

}  // namespace tulpa

#endif  // TULPA_PC_PRIOR_H
