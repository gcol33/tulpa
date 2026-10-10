// family_density.h
// Per-observation log densities of the mu-space families, and the mu clamp the
// fit applies before evaluating them. The Laplace / nested-Laplace kernels and
// downstream pointwise log-likelihoods (WAIC, PSIS-LOO, CPO) call these same
// functions, so a diagnostic scores exactly the density the model was fit with.
//
// `phi` is the dispersion in the SD convention: the residual SD for gaussian
// and lognormal, the precision for beta.

#ifndef TULPA_FAMILY_DENSITY_H
#define TULPA_FAMILY_DENSITY_H

#include <algorithm>
#include <cmath>
#include <string>
#include "tulpa/portable_math.h"

namespace tulpa {

// The one floor on mu, applied identically by the density, the score, the
// Newton working weight and both curvature ladders.
//
// A unit-interval family is held inside (0, 1); every other family with
// positive support is held above the same epsilon; gaussian and lognormal have
// unbounded support and are not clamped at all.
//
// One value, because the ladders differentiate the density: grad_mu is
// proportional to 1 / mu for the binomial and the count families, so a wider
// floor on the score than on the density scales the score by mu / floor while
// the objective the line search reads is still evaluated at the true mu. The
// two are then derivatives of different functions, and the Newton loop stops
// where the clamped score vanishes rather than where the reported objective is
// stationary.
constexpr double kMuFloor = 1e-15;

inline double clamp_mu_unit(double mu) {
    return std::max(std::min(mu, 1.0 - kMuFloor), kMuFloor);
}

inline double clamp_mu_for_family(double mu, const std::string& family) {
    if (family == "binomial" || family == "beta") return clamp_mu_unit(mu);
    if (family == "gaussian" || family == "lognormal") return mu;
    return std::max(mu, kMuFloor);
}

// Inverse logit, evaluated on the side where exp() cannot overflow.
inline double inv_logit_link(double eta) {
    if (eta > 0) return 1.0 / (1.0 + std::exp(-eta));
    double e = std::exp(eta);
    return e / (1.0 + e);
}

// N(mu, phi^2) at y.
inline double log_lik_gaussian(double y, double mu, double phi) {
    double r = y - mu;
    return -0.5 * std::log(2.0 * M_PI * phi * phi) - r * r / (2.0 * phi * phi);
}

// Lognormal density of y, given ly = log(y): N(mu, phi^2) on the log scale plus
// the Jacobian -log(y).
inline double log_lik_lognormal_logy(double ly, double mu, double phi) {
    double r = ly - mu;
    return -ly - 0.5 * std::log(2.0 * M_PI * phi * phi)
           - r * r / (2.0 * phi * phi);
}

inline double log_lik_lognormal(double y, double mu, double phi) {
    return log_lik_lognormal_logy(std::log(std::max(y, 1e-300)), mu, phi);
}

// Beta(mu phi, (1 - mu) phi) at y; mu is taken as already clamped.
inline double log_lik_beta(double y, double mu, double phi) {
    double a = mu * phi;
    double b = (1.0 - mu) * phi;
    return tulpa::math::portable_lgamma(phi) - tulpa::math::portable_lgamma(a)
           - tulpa::math::portable_lgamma(b)
           + (a - 1.0) * std::log(y) + (b - 1.0) * std::log(1.0 - y);
}

// Beta density on the logit linear predictor, with the clamp the fit applies.
inline double log_lik_beta_logit(double y, double eta, double phi) {
    return log_lik_beta(y, clamp_mu_unit(inv_logit_link(eta)), phi);
}

} // namespace tulpa

#endif // TULPA_FAMILY_DENSITY_H
