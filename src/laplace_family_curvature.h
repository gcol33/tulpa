// laplace_family_curvature.h
//
// d w / d eta, where w is the per-observation curvature that
// grad_hess_for_family() returns in GradHess::neg_hess.
//
// WHY THIS IS NOT "the third derivative of the log-likelihood". The exact
// gradient of the Laplace objective needs d log|H| / d theta, and H is built
// from the curvature the fitting path ACTUALLY uses:
//
//     H = A' diag(w) A + P,     w_i = grad_hess_for_family(...).neg_hess
//
// For poisson, binomial, neg_binomial_2 and the two truncated families that w
// is the true observed -l''(eta), so dw/deta is the true -l'''(eta). For
// neg_binomial_1, beta_binomial, t, tweedie and everything reaching the generic
// mu-space route, w is a working / expected weight that is deliberately NOT the
// second derivative (see the comments at laplace_family_link.h:352-365, 377-390,
// 392-401, 403-413 and the Fisher form at :427). Differentiating the true log
// density there would produce a gradient of an objective nobody optimizes.
//
// So the contract is: this returns the eta-derivative of whatever
// grad_hess_for_family returns, and it is exact for that. Naming it d3 would be
// a false label for more than half the families; has_curvature_derivative()
// below is the gate, mirroring has_observed_curvature() at
// laplace_family_link.h:541.

#ifndef TULPA_LAPLACE_FAMILY_CURVATURE_H
#define TULPA_LAPLACE_FAMILY_CURVATURE_H

#include "laplace_family_link.h"
#include <string>
#include <cmath>
#include <limits>

namespace tulpa {

// d2 mu / d eta2, the companion to mu_eta() at laplace_family_link.h:103.
inline double mu_eta2(double eta, const std::string& link) {
    if (link == "identity") return 0.0;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") { double e = safe_pos_eta(eta); return 2.0 / (e * e * e); }
    if (link == "logit") {
        double p;
        if (eta > 0) { double e = std::exp(-eta); p = 1.0 / (1.0 + e); }
        else         { double e = std::exp(eta);  p = e / (1.0 + e); }
        return p * (1.0 - p) * (1.0 - 2.0 * p);
    }
    if (link == "probit") return -eta * R::dnorm(eta, 0.0, 1.0, 0);
    if (link == "cauchit") {
        const double d = 1.0 + eta * eta;
        return -2.0 * eta / (M_PI * d * d);
    }
    if (link == "cloglog") {
        const double ee = std::exp(eta);
        return std::exp(eta - ee) * (1.0 - ee);
    }
    if (link == "sqrt") return 2.0;
    if (link == "1mu2") { double e = safe_pos_eta(eta); return 0.75 / (e * e * std::sqrt(e)); }
    return tulpa_linalg::safe_exp(eta);
}

// d V / d mu, the companion to variance_fn() at laplace_family_link.h:141.
inline double dvariance_dmu(double mu, double phi, const std::string& family,
                            int n_trials) {
    if (family == "gaussian") return 0.0;
    if (family == "lognormal") return 0.0;
    // d/dmu of variance_fn's mu (1-mu) / n. Must track that arm exactly: the two
    // are the numerator and denominator of the same quotient rule in
    // curvature_deta_for_family, so a mismatch is a silently wrong gradient.
    if (family == "binomial") return (1.0 - 2.0 * mu) / n_trials;
    if (family == "poisson") return 1.0;
    if (family == "neg_binomial_2") return 1.0 + 2.0 * mu / phi;
    if (family == "gamma") return 2.0 * mu / phi;
    if (family == "inverse_gaussian") return 3.0 * phi * mu * mu;
    if (family == "beta") {
        // V = 1 / (phi^2 tg), tg = trigamma(mu phi) + trigamma((1-mu) phi).
        // d tg / d mu = phi (psi''(mu phi) - psi''((1-mu) phi)), psi'' = tetragamma.
        const double tg = R::trigamma(mu * phi) + R::trigamma((1.0 - mu) * phi);
        const double dtg = phi * (R::psigamma(mu * phi, 2)
                                  - R::psigamma((1.0 - mu) * phi, 2));
        const double V = 1.0 / (phi * phi * tg);
        return -V * dtg / tg;
    }
    unknown_family_stop("dvariance_dmu", family);
}

// Whether curvature_deta_for_family() is exact for this family. Everything
// listed here has a closed-form eta-derivative of the weight the Newton system
// uses; anything else must not silently receive a plausible-looking number.
inline bool has_curvature_derivative(const std::string& family) {
    if (family == "binomial" || family == "poisson" ||
        family == "neg_binomial_2" || family == "neg_binomial_1" ||
        family == "truncated_poisson" || family == "truncated_neg_binomial_2" ||
        family == "beta_binomial" || family == "t" || family == "tweedie") {
        return true;
    }
    // Generic mu-space route: exact whenever both derivative ladders cover the
    // parsed family and link.
    FamilyLink fl = parse_family_link(family);
    const bool fam_ok =
        fl.family == "gaussian" || fl.family == "lognormal" ||
        fl.family == "binomial" || fl.family == "poisson" ||
        fl.family == "neg_binomial_2" || fl.family == "gamma" ||
        fl.family == "inverse_gaussian" || fl.family == "beta";
    const bool link_ok =
        fl.link == "identity" || fl.link == "log" || fl.link == "inverse" ||
        fl.link == "logit" || fl.link == "probit" || fl.link == "cauchit" ||
        fl.link == "cloglog" || fl.link == "sqrt" || fl.link == "1mu2";
    return fam_ok && link_ok;
}

// d(neg_hess)/d eta. Branch order mirrors grad_hess_for_family exactly, so the
// two stay aligned when a family is added.
inline double curvature_deta_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "binomial") {
        // w = n p (1-p);  dp/deta = p(1-p)
        double p;
        if (eta > 0) { double e = std::exp(-eta); p = 1.0 / (1.0 + e); }
        else         { double e = std::exp(eta);  p = e / (1.0 + e); }
        return n_trials * p * (1.0 - p) * (1.0 - 2.0 * p);
    }
    if (family == "poisson") {
        // w = mu
        return tulpa_linalg::safe_exp(eta);
    }
    if (family == "neg_binomial_2") {
        // w = mu phi (y + phi) / (mu + phi)^2
        const double mu = tulpa_linalg::safe_exp(eta);
        const double s  = mu + phi;
        return mu * phi * (y + phi) * (phi - mu) / (s * s * s);
    }
    if (family == "neg_binomial_1") {
        // w = mu / (1 + phi)  (quasi-likelihood IRLS weight, log link)
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        return mu / (1.0 + phi);
    }
    if (family == "truncated_poisson" || family == "truncated_neg_binomial_2") {
        // w = e_weight = da/p - q da^2 / p^2, with q = exp(-a), p = 1 - q,
        // dq/deta = -q da, dp/deta = q da, d(da)/deta = d2a. Only (a, da, d2a)
        // are needed; no third shape derivative appears.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        double a, da, d2a;
        truncation_shape(family, mu, phi, &a, &da, &d2a);
        const double q = std::exp(-a);
        const double p = -std::expm1(-a);
        const double ps = p > 1e-300 ? p : 1e-300;
        const double p2 = ps * ps, p3 = p2 * ps;
        const double term1 = d2a / ps - q * da * da / p2;
        const double term2 = q * da * (2.0 * d2a - da * da) / p2
                             - 2.0 * q * q * da * da * da / p3;
        return term1 - term2;
    }
    if (family == "beta_binomial") {
        // w = n mu (1-mu) / D, D independent of mu; logit link.
        double mu = linkinv(eta, "logit");
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
        const double n = (double)n_trials;
        const double D = 1.0 + (n - 1.0) / (phi + 1.0);
        return n * (1.0 - 2.0 * mu) * mu * (1.0 - mu) / D;
    }
    if (family == "t") {
        // w = (nu+1) / ((nu+3) phi^2), constant in eta.
        return 0.0;
    }
    if (family == "tweedie") {
        if (std::isnan(phi2)) {
            Rcpp::stop("family 'tweedie' needs phi2 (the variance power p).");
        }
        // w = mu^(2-p) / phi, log link
        const double p = phi2;
        const double mu = std::max(std::exp(eta), 1e-10);
        return (2.0 - p) * std::pow(mu, 2.0 - p) / phi;
    }

    // Generic mu-space route: w = dmu^2 / V(mu), so
    //   dw/deta = (2 dmu mu_eta2 V - dmu^3 V'(mu)) / V^2.
    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    const double dmu  = mu_eta(eta, fl.link);
    const double d2mu = mu_eta2(eta, fl.link);
    if (fl.family == "binomial" || fl.family == "beta") {
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
    } else if (fl.family != "gaussian" && fl.family != "lognormal") {
        mu = std::max(mu, 1e-10);
    }
    const double V  = variance_fn(mu, phi, fl.family, n_trials);
    const double dV = dvariance_dmu(mu, phi, fl.family, n_trials);
    return (2.0 * dmu * d2mu * V - dmu * dmu * dmu * dV) / (V * V);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_CURVATURE_H
