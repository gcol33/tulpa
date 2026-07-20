// builtin_family_zi.h
// Zero-inflation mixture shared by every compiled path, mirroring R/family_zi.R
// (the R side is the reference the density tests validate against).
//
//   pi_i = inv_logit(logit_zi_i)      structural-zero probability
//   p0_i = P_base(Y = 0 | eta_i)
//
//   y = 0:  ll = log(pi + (1 - pi) p0)
//   y > 0:  ll = log(1 - pi) + ll_base(y | eta)
//
// The two compiled backends reach the ZI linear predictor differently, which is
// why the mixture lives here rather than in either one:
//
//   - The generic sampler path carries it as the `logit_zi` callback argument,
//     built from data.X_zi_flat and the beta_zi parameter block.
//   - The spec-driven Laplace path has no such channel (its shim passes 0.0),
//     so it carries the ZI predictor as process 1 and reads eta[1]. Its
//     curvature contract is a row-major n_processes x n_processes block, which
//     is exactly what the mixture's 2 x 2 needs.
//
// Both call the same mixture below, so there is one implementation of the math
// regardless of which predictor channel supplied logit_zi.
//
// A zero-truncated base has p0 == 0, which collapses the y = 0 branch to
// log(pi) and zeroes the count-side gradient, count-side curvature and the
// cross term -- i.e. the hurdle model, with no truncation-specific branch here.

#ifndef TULPA_BUILTIN_FAMILY_ZI_H
#define TULPA_BUILTIN_FAMILY_ZI_H

#include <cmath>
#include <string>

#include "autodiff_utils.h"
#include "laplace_family_link.h"

namespace tulpa {
namespace zi {

// Families whose compiled kernels supply an observed (not moment-approximated)
// curvature, so the mixture's y = 0 branch -- which differentiates through
// P(Y = 0) -- matches the R reference exactly. Delegating to
// has_observed_curvature() keeps this following the math rather than a second
// hand-maintained list: a family becomes ZI-fittable the moment its observed
// curvature is registered. beta_binomial is excluded by that predicate because
// its compiled weight is the moment weight. The R front door gates on this.
inline bool compiled_zi_supported(const std::string& family) {
    return has_observed_curvature(family);
}

// ---------------------------------------------------------------------------
// Templated pieces (double / arena::Var / fwd::Dual), used by the AD path.
// ---------------------------------------------------------------------------

// log(1 + exp(x)), guarded at both tails.
template<typename T>
inline T log1pexp(const T& x) {
    using tulpa::math::safe_exp;
    using tulpa::math::log1p_fn;
    if (x > T(35.0))  return x;
    if (x < T(-10.0)) return safe_exp(x);
    return log1p_fn(safe_exp(x));
}

// log(inv_logit(z)) and log(1 - inv_logit(z)).
template<typename T>
inline T log_pi(const T& z) { return T(0.0) - log1pexp(T(0.0) - z); }

template<typename T>
inline T log1m_pi(const T& z) { return T(0.0) - log1pexp(z); }

// log(exp(a) + exp(b)) without intermediate overflow.
template<typename T>
inline T logspace_add(const T& a, const T& b) {
    using tulpa::math::safe_exp;
    using tulpa::math::log1p_fn;
    T mx = (a >= b) ? a : b;
    T mn = (a >= b) ? b : a;
    return mx + log1p_fn(safe_exp(mn - mx));
}

// The mixture, given the base log-density evaluated at the realized y and at 0.
//
// `base_truncated` marks a base family with no atom at zero (p0 == 0), i.e. the
// hurdle case. Mathematically the general branch already degenerates there, but
// it degenerates through base_ll_0 = -Inf, and -Inf times a derivative is NaN
// under AD rather than the zero the algebra gives. Taking the limit here keeps
// the hurdle gradient exact instead of merely finite -- and base_ll_0 is then
// never evaluated, so callers may leave it unset.
template<typename T>
inline T mixture_ll(double y, const T& logit_zi,
                    const T& base_ll_y, const T& base_ll_0,
                    bool base_truncated = false) {
    if (y == 0.0) {
        if (base_truncated) return log_pi(logit_zi);
        return logspace_add(log_pi(logit_zi), log1m_pi(logit_zi) + base_ll_0);
    }
    return log1m_pi(logit_zi) + base_ll_y;
}

// ---------------------------------------------------------------------------
// Double path: value and the 2 x 2 eta-space derivative block.
// ---------------------------------------------------------------------------

inline double mixture_ll_double(
    double y, int n_trials, double eta_count, double logit_zi,
    const std::string& family, double phi, double phi2
) {
    if (y != 0.0) {
        return log1m_pi(logit_zi) + log_lik_for_family(y, n_trials, eta_count,
                                                       family, phi, phi2);
    }
    if (is_zero_truncated(family)) return log_pi(logit_zi);   // hurdle
    const double ll_0 = log_lik_for_family(0.0, n_trials, eta_count, family,
                                           phi, phi2);
    return logspace_add(log_pi(logit_zi), log1m_pi(logit_zi) + ll_0);
}

// Fills grad[0..1] = d ll / d(eta_count, logit_zi) and the row-major 2 x 2
// negative Hessian. Derivations mirror zi_score_eta() / zi_neg_hessian() in
// R/family_zi.R; both are validated there against finite differences.
//
// At y > 0 the mixture is additively separable, so the count block is the base
// family's own curvature and the cross term is exactly zero. All coupling lives
// in the y = 0 branch, where either component can have produced the zero.
inline void mixture_eta_weights_double(
    double y, int n_trials, double eta_count, double logit_zi,
    const std::string& family, double phi, double phi2,
    double* grad, double* neg_hess
) {
    const double pi = (logit_zi >= 0.0)
        ? 1.0 / (1.0 + std::exp(-logit_zi))
        : std::exp(logit_zi) / (1.0 + std::exp(logit_zi));

    if (y != 0.0) {
        const GradHess gh = grad_hess_for_family(y, n_trials, eta_count,
                                                 family, phi, phi2);
        grad[0]     = gh.grad;
        grad[1]     = -pi;
        neg_hess[0] = gh.neg_hess;
        neg_hess[1] = 0.0;
        neg_hess[2] = 0.0;
        neg_hess[3] = pi * (1.0 - pi);
        return;
    }

    if (is_zero_truncated(family)) {
        // Hurdle: p0 = 0 collapses the y = 0 branch to log(pi), so the count
        // predictor carries no gradient or curvature here and the cross term
        // vanishes. Taken as a limit rather than through the -Inf density.
        grad[0]     = 0.0;
        grad[1]     = 1.0 - pi;
        neg_hess[0] = 0.0;
        neg_hess[1] = 0.0;
        neg_hess[2] = 0.0;
        neg_hess[3] = pi * (1.0 - pi);
        return;
    }

    // y == 0: differentiate log D, D = pi + (1 - pi) p0, through both
    // predictors. s0 / w0 are the base score and observed curvature at y = 0 --
    // observed, not the Newton working weight, because this branch
    // differentiates the density itself rather than taking an expectation.
    // A zero-truncated base gives p0 = 0 here, which zeroes D_eta, D_ee and
    // D_ez and leaves the hurdle model, so s0 and w0 need only be finite.
    const double p0 = std::exp(log_lik_for_family(0.0, n_trials, eta_count,
                                                  family, phi, phi2));
    const GradHess gh0 = obs_grad_hess_for_family(0.0, n_trials, eta_count,
                                                  family, phi, phi2);
    const double s0 = gh0.grad;
    const double w0 = gh0.neg_hess;

    const double D     = pi + (1.0 - pi) * p0;
    const double D_eta = (1.0 - pi) * p0 * s0;
    const double D_z   = pi * (1.0 - pi) * (1.0 - p0);
    const double D_ee  = (1.0 - pi) * p0 * (s0 * s0 - w0);
    const double D_zz  = (1.0 - p0) * pi * (1.0 - pi) * (1.0 - 2.0 * pi);
    const double D_ez  = -pi * (1.0 - pi) * p0 * s0;

    grad[0] = D_eta / D;
    grad[1] = D_z / D;

    const double nh_ee = -(D_ee / D - (D_eta / D) * (D_eta / D));
    const double nh_zz = -(D_zz / D - (D_z / D) * (D_z / D));
    const double nh_ez = -(D_ez / D - D_eta * D_z / (D * D));

    neg_hess[0] = nh_ee;
    neg_hess[1] = nh_ez;
    neg_hess[2] = nh_ez;
    neg_hess[3] = nh_zz;
}

} // namespace zi
} // namespace tulpa

#endif // TULPA_BUILTIN_FAMILY_ZI_H
