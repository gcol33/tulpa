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

// d3 mu / d eta3, the companion to mu_eta2() above. Needed by the generic
// mu-space route of curvature_deta2_for_family().
inline double mu_eta3(double eta, const std::string& link) {
    if (link == "identity") return 0.0;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") { double e = safe_pos_eta(eta); return -6.0 / (e * e * e * e); }
    if (link == "logit") {
        double p;
        if (eta > 0) { double e = std::exp(-eta); p = 1.0 / (1.0 + e); }
        else         { double e = std::exp(eta);  p = e / (1.0 + e); }
        return (1.0 - 6.0 * p + 6.0 * p * p) * p * (1.0 - p);
    }
    if (link == "probit") return (eta * eta - 1.0) * R::dnorm(eta, 0.0, 1.0, 0);
    if (link == "cauchit") {
        const double d = 1.0 + eta * eta;
        return 2.0 * (3.0 * eta * eta - 1.0) / (M_PI * d * d * d);
    }
    if (link == "cloglog") {
        const double ee = std::exp(eta);
        return std::exp(eta - ee) * (1.0 - 3.0 * ee + ee * ee);
    }
    if (link == "sqrt") return 0.0;
    if (link == "1mu2") { double e = safe_pos_eta(eta); return -1.875 / (e * e * e * std::sqrt(e)); }
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

// d2 V / d mu2, the companion to dvariance_dmu() above. Same family arms.
inline double d2variance_dmu2(double mu, double phi, const std::string& family,
                              int n_trials) {
    if (family == "gaussian") return 0.0;
    if (family == "lognormal") return 0.0;
    if (family == "binomial") return -2.0 / n_trials;
    if (family == "poisson") return 0.0;
    if (family == "neg_binomial_2") return 2.0 / phi;
    if (family == "gamma") return 2.0 / phi;
    if (family == "inverse_gaussian") return 6.0 * phi * mu;
    if (family == "beta") {
        // V = 1 / (phi^2 tg); reuse dvariance_dmu's tg, tg1 and add the next
        // pentagamma rung tg2 = d2 tg / d mu2.
        const double tg  = R::trigamma(mu * phi) + R::trigamma((1.0 - mu) * phi);
        const double tg1 = phi * (R::psigamma(mu * phi, 2)
                                  - R::psigamma((1.0 - mu) * phi, 2));
        const double tg2 = phi * phi * (R::psigamma(mu * phi, 3)
                                        + R::psigamma((1.0 - mu) * phi, 3));
        const double cc = 1.0 / (phi * phi);
        return -cc * tg2 / (tg * tg) + 2.0 * cc * tg1 * tg1 / (tg * tg * tg);
    }
    unknown_family_stop("d2variance_dmu2", family);
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

// Whether curvature_deta2_for_family() is exact for this family. The mirror of
// has_curvature_derivative(): the truncated pair carry the third truncation-shape
// derivative d3a from truncation_shape(), so their second eta derivative is
// closed-form alongside every other family that has the first.
inline bool has_curvature_2nd_derivative(const std::string& family) {
    return has_curvature_derivative(family);
}

// Whether the exact analytic mode Jacobian dx_hat/dtheta can be formed for this
// family. From the true score stationarity of the inner solve the Jacobian is
// -(A' diag(W_obs) A + P)^-1 (dP/dtheta) x_hat, governed by the OBSERVED
// curvature W_obs = -l''(eta) -- NOT the Newton working weight H is built from,
// which only equals it for canonical or constant-curvature families. It is exact
// when W_obs is available: either explicitly (has_observed_curvature) or because
// the working weight already is it (gaussian/lognormal constant curvature, or a
// canonical link). Families where the two differ and no exact observed form
// exists -- beta_binomial, t, tweedie, non-canonical generic -- return false, so
// the marginal correction differences the mode instead of trusting a working-
// weight Jacobian.
inline bool has_exact_mode_jacobian(const std::string& family) {
    if (has_observed_curvature(family)) return true;
    FamilyLink fl = parse_family_link(family);
    if (fl.family == "gaussian" || fl.family == "lognormal") return true;
    // Canonical links, where the Fisher working weight (dmu/deta)^2 / V is -l''.
    return (fl.family == "poisson" && fl.link == "log") ||
           (fl.family == "binomial" && fl.link == "logit") ||
           (fl.family == "gamma" && fl.link == "inverse") ||
           (fl.family == "inverse_gaussian" && fl.link == "1mu2");
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

// d2(neg_hess)/d eta2, the sibling of curvature_deta_for_family. Branch order
// mirrors it exactly, including the truncated pair, whose second eta derivative
// is assembled from the shape derivatives (a, da, d2a, d3a).
inline double curvature_deta2_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "binomial") {
        // dw/deta = n p(1-p)(1-2p);  d/deta of that = n (1-6p+6p^2) p(1-p)
        double p;
        if (eta > 0) { double e = std::exp(-eta); p = 1.0 / (1.0 + e); }
        else         { double e = std::exp(eta);  p = e / (1.0 + e); }
        return n_trials * (1.0 - 6.0 * p + 6.0 * p * p) * p * (1.0 - p);
    }
    if (family == "poisson") {
        // dw/deta = mu, so d2w/deta2 = mu
        return tulpa_linalg::safe_exp(eta);
    }
    if (family == "neg_binomial_2") {
        // dw/deta = mu phi (y+phi)(phi-mu)/s^3, s = mu+phi
        const double mu = tulpa_linalg::safe_exp(eta);
        const double s  = mu + phi;
        return phi * (y + phi) * mu * (mu * mu - 4.0 * mu * phi + phi * phi)
               / (s * s * s * s);
    }
    if (family == "neg_binomial_1") {
        // dw/deta = mu / (1+phi), so d2w/deta2 = mu / (1+phi)
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        return mu / (1.0 + phi);
    }
    if (family == "truncated_poisson" || family == "truncated_neg_binomial_2") {
        // w = f(a, da) = da/p - q da^2/p^2, q = e^{-a}, p = 1 - q, so w depends
        // on eta only through the shape a and its derivatives. The composite
        // second derivative (dev_notes/proto_truncated_curvature.R):
        //   d2w = f_aa da^2 + 2 f_ada da d2a + f_dada d2a^2 + f_a d2a + f_da d3a
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        double a, da, d2a, d3a;
        truncation_shape(family, mu, phi, &a, &da, &d2a, &d3a);
        const double q = std::exp(-a);
        const double p = -std::expm1(-a);
        const double ps = p > 1e-300 ? p : 1e-300;
        const double p2 = ps * ps, p3 = p2 * ps, p4 = p3 * ps;
        const double f_a    = -da * q / p2 + da * da * q * (p + 2.0 * q) / p3;
        const double f_da   = 1.0 / ps - 2.0 * q * da / p2;
        const double f_dada = -2.0 * q / p2;
        const double f_ada  = -q / p2 + 2.0 * da * q * (p + 2.0 * q) / p3;
        const double R_a    = -q * (p + 2.0 * q) / p3
                              - 2.0 * q * q * (2.0 * p + 3.0 * q) / p4;
        const double f_aa   = da * q * (p + 2.0 * q) / p3 + da * da * R_a;
        return f_aa * da * da + 2.0 * f_ada * da * d2a + f_dada * d2a * d2a
               + f_a * d2a + f_da * d3a;
    }
    if (family == "beta_binomial") {
        // same logit shape as binomial, with n -> n/D
        double mu = linkinv(eta, "logit");
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
        const double n = (double)n_trials;
        const double D = 1.0 + (n - 1.0) / (phi + 1.0);
        return n * (1.0 - 6.0 * mu + 6.0 * mu * mu) * mu * (1.0 - mu) / D;
    }
    if (family == "t") {
        // w constant in eta
        return 0.0;
    }
    if (family == "tweedie") {
        if (std::isnan(phi2)) {
            Rcpp::stop("family 'tweedie' needs phi2 (the variance power p).");
        }
        // dw/deta = (2-p) mu^(2-p)/phi, so d2w/deta2 = (2-p)^2 mu^(2-p)/phi
        const double a = 2.0 - phi2;
        const double mu = std::max(std::exp(eta), 1e-10);
        return a * a * std::pow(mu, a) / phi;
    }

    // Generic mu-space route: w = dmu^2 / V(mu). With u = mu_eta, u1 = mu_eta2,
    // u2 = mu_eta3, Vm = dvariance_dmu, Vmm = d2variance_dmu2,
    //   d2w/deta2 = 2(u1^2 + u u2)/V - (5 u^2 u1 Vm + u^4 Vmm)/V^2
    //               + 2 u^4 Vm^2 / V^3.
    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    const double u  = mu_eta(eta, fl.link);
    const double u1 = mu_eta2(eta, fl.link);
    const double u2 = mu_eta3(eta, fl.link);
    if (fl.family == "binomial" || fl.family == "beta") {
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
    } else if (fl.family != "gaussian" && fl.family != "lognormal") {
        mu = std::max(mu, 1e-10);
    }
    const double V   = variance_fn(mu, phi, fl.family, n_trials);
    const double Vm  = dvariance_dmu(mu, phi, fl.family, n_trials);
    const double Vmm = d2variance_dmu2(mu, phi, fl.family, n_trials);
    const double u2p = u * u, u4 = u2p * u2p;
    return 2.0 * (u1 * u1 + u * u2) / V
           - (5.0 * u2p * u1 * Vm + u4 * Vmm) / (V * V)
           + 2.0 * u4 * Vm * Vm / (V * V * V);
}

// ---------------------------------------------------------------------------
// d(W_obs - w)/deta: the eta-derivative of the observed-minus-working curvature.
//
// The closed outer Hessian needs it because the mode motion is governed by the
// TRUE curvature H_true = H + A' diag(W_obs - w) A while log|H| is the working
// one. Differentiating a quantity formed on H_true^-1 therefore needs
// dH_true/dtheta, which is dH/dtheta plus A' diag(d(W_obs - w)/deta * eta_dot) A.
//
// The difference is identically zero for every family whose Newton weight
// already IS the observed curvature, so a zero derivative is exact there and
// the branch below is only about the two families where it is not.
inline bool has_obs_curvature_delta_derivative(const std::string& family) {
    return has_curvature_derivative(family);
}

// The same for the SECOND eta-derivative of the difference, which the coupled
// y = 0 branch of an untruncated mixture needs (it differentiates log P(Y = 0)
// a fourth time). Everything whose difference is identically zero has it
// trivially; neg_binomial_1 carries it through the pentagamma rung. The
// truncated pair is the one gap: d2 log p / deta2 differentiated twice more
// would need a fourth truncation-shape derivative, and truncation_shape() stops
// at d3a. That gap is unreachable in practice -- a zero-truncated base makes the
// mixture a hurdle, whose y = 0 branch never forms this.
inline bool has_obs_curvature_delta_2nd_derivative(const std::string& family) {
    if (family == "truncated_neg_binomial_2") return false;
    return has_curvature_2nd_derivative(family);
}

inline double obs_curvature_delta_deta_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "neg_binomial_1") {
        // W_obs = -s + r^2 T1 with r = mu/phi, dr/deta = r,
        //   s  = r (psi(y+r) - psi(r) - log1p(phi)),  ds/deta = s - r^2 T1,
        //   T1 = psi'(r) - psi'(y+r),  T2 = psi''(r) - psi''(y+r),
        // so d(r^2 T1)/deta = 2 r^2 T1 + r^3 T2 and
        //   dW_obs/deta = -s + 3 r^2 T1 + r^3 T2.
        // The working weight mu/(1+phi) is subtracted through its own registered
        // derivative, so the two halves of the difference cannot drift apart.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        const double r  = mu / phi;
        const double s  = r * (tulpa::math::portable_digamma(y + r)
                               - tulpa::math::portable_digamma(r)
                               - std::log1p(phi));
        const double T1 = tulpa::math::portable_trigamma(r)
                          - tulpa::math::portable_trigamma(y + r);
        const double T2 = tulpa::math::portable_tetragamma(r)
                          - tulpa::math::portable_tetragamma(y + r);
        return -s + 3.0 * r * r * T1 + r * r * r * T2
             - curvature_deta_for_family(y, n_trials, eta, family, phi, phi2);
    }
    if (family == "truncated_neg_binomial_2") {
        // W_obs = (untruncated NB2 observed curvature) + d2 log p, and
        // w = e_weight, so the difference differentiates to the NB2 curvature
        // derivative plus d3 log p minus the truncated working one. Both
        // curvature derivatives are already registered above; only d3 log p is
        // new, and truncation_shape() already supplies the d3a it needs.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        double a, da, d2a, d3a;
        truncation_shape(family, mu, phi, &a, &da, &d2a, &d3a);
        const double q  = std::exp(-a);
        const double p  = -std::expm1(-a);
        const double ps = p > 1e-300 ? p : 1e-300;
        // p = 1 - e^-a, so its eta-derivatives follow from (a, da, d2a, d3a).
        const double dp  = q * da;
        const double d2p = q * (d2a - da * da);
        const double d3p = q * (d3a - 3.0 * da * d2a + da * da * da);
        // d3 log p = p'''/p - 3 p' p''/p^2 + 2 p'^3/p^3.
        const double d3log_p = d3p / ps
            - 3.0 * dp * d2p / (ps * ps)
            + 2.0 * dp * dp * dp / (ps * ps * ps);
        return curvature_deta_for_family(y, n_trials, eta, "neg_binomial_2",
                                         phi, phi2)
             + d3log_p
             - curvature_deta_for_family(y, n_trials, eta, family, phi, phi2);
    }
    // W_obs == w identically: the difference is the zero function.
    return 0.0;
}

// d2(W_obs - w)/deta2, the next rung. Same contract as above: zero wherever the
// two weights coincide, and gated by has_obs_curvature_delta_2nd_derivative().
inline double obs_curvature_delta_deta2_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "neg_binomial_1") {
        // Differentiating dW_obs/deta = -s + 3 r^2 T1 + r^3 T2 once more, with
        // dr/deta = r, ds/deta = s - r^2 T1 and T3 = psi'''(r) - psi'''(y+r):
        //   d2W_obs/deta2 = -s + 7 r^2 T1 + 6 r^3 T2 + r^4 T3.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        const double r  = mu / phi;
        const double s  = r * (tulpa::math::portable_digamma(y + r)
                               - tulpa::math::portable_digamma(r)
                               - std::log1p(phi));
        const double T1 = tulpa::math::portable_trigamma(r)
                          - tulpa::math::portable_trigamma(y + r);
        const double T2 = tulpa::math::portable_tetragamma(r)
                          - tulpa::math::portable_tetragamma(y + r);
        const double T3 = tulpa::math::portable_pentagamma(r)
                          - tulpa::math::portable_pentagamma(y + r);
        const double r2 = r * r;
        return -s + 7.0 * r2 * T1 + 6.0 * r2 * r * T2 + r2 * r2 * T3
             - curvature_deta2_for_family(y, n_trials, eta, family, phi, phi2);
    }
    return 0.0;
}

// The TRUE eta-derivatives of the observed curvature, as opposed to the working
// weight's. H is built from the working weight, so the objective's log|H| moves
// with curvature_deta_for_family; but a mixture's y = 0 branch differentiates
// the DENSITY (through P(Y = 0)), and that needs these. The two coincide for
// every family whose Newton weight already is the observed curvature, which is
// why this distinction stayed invisible until neg_binomial_1.
inline double obs_curvature_deta_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    return curvature_deta_for_family(y, n_trials, eta, family, phi, phi2)
         + obs_curvature_delta_deta_for_family(y, n_trials, eta, family, phi,
                                               phi2);
}

inline double obs_curvature_deta2_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    return curvature_deta2_for_family(y, n_trials, eta, family, phi, phi2)
         + obs_curvature_delta_deta2_for_family(y, n_trials, eta, family, phi,
                                                phi2);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_CURVATURE_H
