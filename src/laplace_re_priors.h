// laplace_re_priors.h
// Random-effect, fixed-effect, and centering helpers for Laplace solvers.

#ifndef TULPA_LAPLACE_RE_PRIORS_H
#define TULPA_LAPLACE_RE_PRIORS_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

// The weak default fixed-effect prior, beta_j ~ N(0, 100^2), in the two
// parameterizations the kernels read it in: the nested-Laplace prior helpers
// take a precision, the spec path takes an SD on ModelData::sigma_beta. Both
// are defined here so a change to the default cannot move only one of them.
inline constexpr double DEFAULT_TAU_BETA   = 1e-4;
inline constexpr double DEFAULT_SIGMA_BETA = 100.0;
static_assert(DEFAULT_SIGMA_BETA * DEFAULT_SIGMA_BETA * DEFAULT_TAU_BETA
                  - 1.0 < 1e-12 &&
              1.0 - DEFAULT_SIGMA_BETA * DEFAULT_SIGMA_BETA * DEFAULT_TAU_BETA
                  < 1e-12,
              "DEFAULT_SIGMA_BETA and DEFAULT_TAU_BETA must describe the same "
              "prior: tau = 1 / sigma^2.");

// Floor added to sigma_re^2 before inverting it into a precision, so a zero or
// near-zero random-effect SD yields a large but finite tau_re instead of a
// division by zero.
inline constexpr double RE_VARIANCE_FLOOR = 1e-10;

// Gaussian prior on the fixed effects beta. With both vectors empty it
// reproduces the historical weak prior beta_j ~ N(0, 1e4) applied
// uniformly. A non-empty `tau` overrides the precision (1/sd^2) per
// coefficient; a non-empty `mean` shifts the prior mean per coefficient.
// When set, each vector must have length p -- the R layer recycles scalar
// mean/sd to full length before constructing this.
struct BetaPrior {
    std::vector<double> mean;  // length p, or empty -> 0
    std::vector<double> tau;   // length p precision, or empty -> DEFAULT_TAU_BETA

    double tau_at(int j) const { return tau.empty() ? DEFAULT_TAU_BETA : tau[j]; }
    double mean_at(int j) const { return mean.empty() ? 0.0 : mean[j]; }
};

// Subtract the mean from x[start, start+length), returning the mean that
// was applied. Single-arm callers ignore the return value; joint drivers
// use it to shift per-arm intercepts so eta is preserved when a rank-
// deficient block is re-centered after a Newton step.
inline double center_effects(Rcpp::NumericVector& x, int start, int length) {
    if (length <= 0) return 0.0;
    double mean = 0.0;
    for (int i = 0; i < length; i++) mean += x[start + i];
    mean /= length;
    for (int i = 0; i < length; i++) x[start + i] -= mean;
    return mean;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_RE_PRIORS_H
