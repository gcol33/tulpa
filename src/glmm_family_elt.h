// glmm_family_elt.h
// The per-observation GLMM family density behind the compiled GLMM oracle
// (glmm_oracle.h): the family enum, its string map, and the conditional
// log-density with its first two eta-derivatives. Kept in its own header so a
// kernel that needs the density without the oracle's Eigen machinery (the
// Polya-Gamma samplers' recorded log joint) reads the same single source.

#ifndef TULPA_GLMM_FAMILY_ELT_H
#define TULPA_GLMM_FAMILY_ELT_H

#include <Rcpp.h>
#include "laplace_likelihoods.h"   // log_rising_factorial
#include <cmath>
#include <string>
#include <stdexcept>

namespace tulpa {

enum class GLMMFamily { Binomial, Poisson, Gaussian, NegBin };

inline GLMMFamily glmm_family_from_string(const std::string& f) {
    if (f == "binomial")                          return GLMMFamily::Binomial;
    if (f == "poisson")                           return GLMMFamily::Poisson;
    if (f == "gaussian")                          return GLMMFamily::Gaussian;
    if (f == "neg_binomial_2" || f == "negbin")   return GLMMFamily::NegBin;
    throw std::runtime_error("tulpa GLMM oracle: unsupported family '" + f + "'.");
}

// Per-observation conditional log-density and its first two derivatives in the
// linear predictor eta. The full normalizing constants are kept (so `l` is a
// true log-density); only differences / curvature enter the MH ratio and the
// marginal, but keeping them makes the routine self-contained and reusable.
//   phi_var: the family's dispersion read in the VARIANCE convention -- the
//   residual variance for gaussian (sd = sqrt(phi_var)), the size r for
//   neg-binomial, unused by binomial and poisson. laplace_family_link.h reads
//   the same slot in the SD convention; the two readings differ for gaussian,
//   and R converts between them at each boundary (.phi_to_kernel takes the
//   registry's variance to the SD the Laplace kernels want, .phi_to_registry
//   takes it back).
struct GLMMElt { double l, d1, d2; };

// The part of the log-density that does not depend on eta: the binomial
// coefficient, the Poisson and neg-binomial factorials and gamma ratio, the
// gaussian normalizer. It goes through R's math library (lchoose checks R's C
// stack), so a caller running observations across threads evaluates it once
// per observation on the calling thread and adds it to glmm_elt_eta().
inline double glmm_log_const(GLMMFamily fam, double y, double n_trials,
                             double phi_var) {
    switch (fam) {
    case GLMMFamily::Binomial: return R::lchoose(n_trials, y);
    case GLMMFamily::Poisson:  return -R::lgammafn(y + 1.0);
    case GLMMFamily::Gaussian:
        return -0.5 * std::log(2.0 * M_PI) - 0.5 * std::log(phi_var);
    case GLMMFamily::NegBin:
        return log_rising_factorial((int) y, phi_var) - R::lgammafn(y + 1.0);
    }
    return 0.0;   // unreachable
}

// The eta-dependent part of the log-density and its first two eta-derivatives.
// Plain arithmetic only, so it is safe inside a parallel region.
inline GLMMElt glmm_elt_eta(GLMMFamily fam, double eta, double y,
                            double n_trials, double phi_var) {
    switch (fam) {
    case GLMMFamily::Binomial: {
        const double l1p = (eta > 0.0) ? eta + std::log1p(std::exp(-eta))
                                       : std::log1p(std::exp(eta));
        const double mu  = 1.0 / (1.0 + std::exp(-eta));
        return { y * eta - n_trials * l1p, y - n_trials * mu,
                 -n_trials * mu * (1.0 - mu) };
    }
    case GLMMFamily::Poisson: {
        const double lam = std::exp(eta);
        return { y * eta - lam, y - lam, -lam };
    }
    case GLMMFamily::Gaussian: {
        const double r = y - eta;
        return { -0.5 * r * r / phi_var, r / phi_var, -1.0 / phi_var };
    }
    case GLMMFamily::NegBin: {
        // With s = log1p(mu / r) = softplus(eta - log r),
        //   r log(r / (r + mu)) + y log(mu / (r + mu)) = y (eta - log r - s) - r s,
        // whose terms stay exact as r grows (r s -> mu, the Poisson limit).
        const double rsz = phi_var;                 // size
        const double mu  = std::exp(eta);
        const double rm  = rsz + mu;
        const double log_r = std::log(rsz);
        const double x   = eta - log_r;
        const double s   = (x > 0.0) ? x + std::log1p(std::exp(-x))
                                     : std::log1p(std::exp(x));
        const double l   = y * (x - s) - rsz * s;
        const double d1  = y - (rsz + y) * mu / rm;
        const double d2  = -(rsz + y) * rsz * mu / (rm * rm);
        return { l, d1, d2 };
    }
    }
    return { 0.0, 0.0, 0.0 };   // unreachable
}

inline GLMMElt glmm_elt(GLMMFamily fam, double eta, double y,
                        double n_trials, double phi_var) {
    GLMMElt e = glmm_elt_eta(fam, eta, y, n_trials, phi_var);
    e.l += glmm_log_const(fam, y, n_trials, phi_var);
    return e;
}

}  // namespace tulpa

#endif  // TULPA_GLMM_FAMILY_ELT_H
