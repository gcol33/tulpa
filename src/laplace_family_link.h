// laplace_family_link.h
// Family/link dispatch and per-observation likelihood utilities.

#ifndef TULPA_LAPLACE_FAMILY_LINK_H
#define TULPA_LAPLACE_FAMILY_LINK_H

#include "laplace_likelihoods.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

// Default degrees of freedom for the Student-t (robust) family, used when no
// phi2 is supplied through the second dispersion channel; nu = 4 is the common
// heavy-tailed drop-in for Gaussian.
constexpr double kStudentTDf = 4.0;

struct FamilyLink {
    std::string family;
    std::string link;
};

inline FamilyLink parse_family_link(const std::string& code) {
    FamilyLink fl;
    static const std::pair<std::string, std::string> defaults[] = {
        {"gaussian", "identity"},
        {"binomial", "logit"},
        {"poisson", "log"},
        {"neg_binomial_2", "log"},
        {"gamma", "log"},
        {"inverse_gaussian", "log"},
        {"beta", "logit"},
        {"lognormal", "identity"},
    };
    for (auto& [fam, def_link] : defaults) {
        if (code == fam) { fl.family = fam; fl.link = def_link; return fl; }
        std::string prefix = fam + "_";
        if (code.substr(0, prefix.size()) == prefix && code.size() > prefix.size()) {
            std::string suffix = code.substr(prefix.size());
            static const char* links[] = {
                "identity", "log", "inverse", "logit", "probit",
                "cauchit", "cloglog", "sqrt", "1mu2", nullptr
            };
            for (int i = 0; links[i]; i++) {
                if (suffix == links[i]) {
                    fl.family = fam;
                    fl.link = suffix;
                    return fl;
                }
            }
        }
    }
    fl.family = code;
    for (auto& [fam, def_link] : defaults) {
        if (code == fam) { fl.link = def_link; return fl; }
    }
    fl.link = "log";
    return fl;
}

// Floor for links whose inverse is singular at eta <= 0 (inverse: 1/eta;
// 1mu2: 1/sqrt(eta)). The mean of the families using these links is positive,
// so eta is clamped to a small positive value rather than producing Inf/NaN
// from 1/eta or sqrt of a negative argument when Newton's unconstrained eta
// crosses zero -- the analogue of the safe_exp guard the log link already uses

inline double safe_pos_eta(double eta) {
    constexpr double kEtaFloor = 1e-10;
    return eta < kEtaFloor ? kEtaFloor : eta;
}

inline double linkinv(double eta, const std::string& link) {
    if (link == "identity") return eta;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") return 1.0 / safe_pos_eta(eta);
    if (link == "logit") {
        if (eta > 0) return 1.0 / (1.0 + std::exp(-eta));
        double e = std::exp(eta);
        return e / (1.0 + e);
    }
    if (link == "probit") return R::pnorm(eta, 0.0, 1.0, 1, 0);
    if (link == "cauchit") return 0.5 + std::atan(eta) / M_PI;
    if (link == "cloglog") return 1.0 - std::exp(-std::exp(eta));
    if (link == "sqrt") return eta * eta;
    if (link == "1mu2") return 1.0 / std::sqrt(safe_pos_eta(eta));
    return tulpa_linalg::safe_exp(eta);
}

inline double mu_eta(double eta, const std::string& link) {
    if (link == "identity") return 1.0;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") { double e = safe_pos_eta(eta); return -1.0 / (e * e); }
    if (link == "logit") {
        double p;
        if (eta > 0) {
            double e = std::exp(-eta);
            p = 1.0 / (1.0 + e);
        } else {
            double e = std::exp(eta);
            p = e / (1.0 + e);
        }
        return p * (1.0 - p);
    }
    if (link == "probit") return R::dnorm(eta, 0.0, 1.0, 0);
    if (link == "cauchit") return 1.0 / (M_PI * (1.0 + eta * eta));
    if (link == "cloglog") return std::exp(eta - std::exp(eta));
    if (link == "sqrt") return 2.0 * eta;
    if (link == "1mu2") { double e = safe_pos_eta(eta); return -0.5 / (e * std::sqrt(e)); }
    return tulpa_linalg::safe_exp(eta);
}

inline double variance_fn(double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") return phi * phi;
    if (family == "lognormal") return phi * phi;
    if (family == "binomial") return n_trials * mu * (1.0 - mu);
    if (family == "poisson") return mu;
    if (family == "neg_binomial_2") return mu + mu * mu / phi;
    if (family == "gamma") return mu * mu / phi;
    if (family == "inverse_gaussian") return phi * mu * mu * mu;
    if (family == "beta") {
        // Working variance: V s.t. dmu^2 / V = Fisher info per obs on eta.
        // Fisher info = phi^2 * (trigamma(mu*phi) + trigamma((1-mu)*phi)) * dmu^2
        // (Ferrari & Cribari-Neto 2004).
        double tg = R::trigamma(mu * phi) + R::trigamma((1.0 - mu) * phi);
        return 1.0 / (phi * phi * tg);
    }
    return mu;
}

inline double grad_mu(double y, double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") return (y - mu) / (phi * phi);
    if (family == "lognormal") {
        // eta = E[log Y]; gradient wrt eta equals gradient wrt mu under
        // identity link. log p(y|eta) = -log(y) - 0.5*log(2pi phi^2)
        //                              - (log(y) - eta)^2 / (2 phi^2)
        // d/d eta = (log(y) - eta) / phi^2.
        double ly = std::log(std::max(y, 1e-300));
        return (ly - mu) / (phi * phi);
    }
    if (family == "binomial") return ((int)y - n_trials * mu) / (mu * (1.0 - mu));
    if (family == "poisson") return (int)y / mu - 1.0;
    if (family == "neg_binomial_2") return (int)y / mu - ((int)y + phi) / (mu + phi);
    if (family == "gamma") return phi * (y - mu) / (mu * mu);
    if (family == "inverse_gaussian") return (y - mu) / (phi * mu * mu * mu);
    if (family == "beta") {
        // d log f / d mu = phi * (y* - mu*) with y* = logit(y),
        // mu* = digamma(mu*phi) - digamma((1-mu)*phi).
        double y_star  = std::log(y) - std::log(1.0 - y);
        double mu_star = R::digamma(mu * phi) - R::digamma((1.0 - mu) * phi);
        return phi * (y_star - mu_star);
    }
    return (int)y / mu - 1.0;
}

inline double log_lik_mu(double y, double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") {
        double r = y - mu;
        return -0.5 * std::log(2.0 * M_PI * phi * phi) - r * r / (2.0 * phi * phi);
    }
    if (family == "lognormal") {
        double ly = std::log(std::max(y, 1e-300));
        double r  = ly - mu;
        return -ly - 0.5 * std::log(2.0 * M_PI * phi * phi)
               - r * r / (2.0 * phi * phi);
    }
    if (family == "binomial") {
        double p = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
        // lchoose keeps this a true log-density, matching the poisson arm below
        // (which keeps its lgamma(y+1)), dbinom(), and the other kernels.
        return (int)y * std::log(p) + (n_trials - (int)y) * std::log(1.0 - p)
               + R::lchoose((double) n_trials, y);
    }
    if (family == "poisson") {
        double safe_mu = std::max(mu, 1e-15);
        return (int)y * std::log(safe_mu) - safe_mu - R::lgammafn((int)y + 1.0);
    }
    if (family == "neg_binomial_2") {
        double safe_mu = std::max(mu, 1e-15);
        return R::lgammafn((int)y + phi) - R::lgammafn(phi) - R::lgammafn((int)y + 1.0)
               + phi * std::log(phi / (safe_mu + phi))
               + (int)y * std::log(safe_mu / (safe_mu + phi));
    }
    if (family == "gamma") {
        return phi * std::log(phi) - R::lgammafn(phi) + (phi - 1.0) * std::log(y)
               - phi * std::log(mu) - phi * y / mu;
    }
    if (family == "inverse_gaussian") {
        double r = y - mu;
        return -0.5 * std::log(2.0 * M_PI * phi * y * y * y)
               - r * r / (2.0 * phi * mu * mu * y);
    }
    if (family == "beta") {
        double a = mu * phi;
        double b = (1.0 - mu) * phi;
        return R::lgammafn(phi) - R::lgammafn(a) - R::lgammafn(b)
               + (a - 1.0) * std::log(y) + (b - 1.0) * std::log(1.0 - y);
    }
    double safe_mu = std::max(mu, 1e-15);
    return (int)y * std::log(safe_mu) - safe_mu - R::lgammafn((int)y + 1.0);
}

// Tweedie compound Poisson-gamma log-density (1 < p < 2), log link upstream:
// mean mu, dispersion phi, power p. Exact zero mass exp(-lambda); positive y
// via the event-count series (Dunn & Smyth 2005), log-sum-exp'd from the
// dominating index j_max = y^(2-p) / (phi (2-p)) until terms fall 37 nats
// below the running peak. Mirrors .tweedie_loglik in R/family_loglik.R.
inline double log_lik_tweedie(double y, double mu, double phi, double p) {
    mu = std::max(mu, 1e-10);
    const double lam = std::pow(mu, 2.0 - p) / (phi * (2.0 - p));
    if (y < 0.0) return R_NegInf;
    if (y <= 0.0) return -lam;
    const double a  = (2.0 - p) / (p - 1.0);
    const double b  = std::pow(mu, 1.0 - p) / (phi * (p - 1.0));
    const double la = std::log(lam), lb = std::log(b), ly = std::log(y);
    auto logterm = [&](double n) {
        return n * la - R::lgammafn(n + 1.0) + n * a * lb
             + (n * a - 1.0) * ly - R::lgammafn(n * a);
    };
    const double jmax = std::pow(y, 2.0 - p) / (phi * (2.0 - p));
    const int n0 = std::max(1, (int)std::lround(jmax));
    std::vector<double> terms;
    double lmax = logterm((double)n0);
    terms.push_back(lmax);
    for (int n = n0 + 1; ; ++n) {
        const double lt = logterm((double)n);
        terms.push_back(lt);
        if (lt > lmax) lmax = lt;
        if (lt < lmax - 37.0) break;
    }
    for (int n = n0 - 1; n >= 1; --n) {
        const double lt = logterm((double)n);
        terms.push_back(lt);
        if (lt > lmax) lmax = lt;
        if (lt < lmax - 37.0) break;
    }
    double s = 0.0;
    for (double lt : terms) s += std::exp(lt - lmax);
    return lmax + std::log(s) - lam - b * y;
}

struct GradHess {
    double grad;
    double neg_hess;
};

inline GradHess grad_hess_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "binomial") {
        return {grad_log_lik_binomial((int)y, n_trials, eta),
                neg_hess_log_lik_binomial((int)y, n_trials, eta)};
    }
    if (family == "poisson") {
        return {grad_log_lik_poisson((int)y, eta),
                neg_hess_log_lik_poisson((int)y, eta)};
    }
    if (family == "neg_binomial_2") {
        return {grad_log_lik_negbin((int)y, eta, phi),
                neg_hess_log_lik_negbin((int)y, eta, phi)};
    }
    if (family == "beta_binomial") {
        // Beta-binomial (logit link, mu = P(success), phi = precision a + b).
        // Score is exact; the working weight is the moment-based Fisher weight
        // n mu(1-mu) / D with the overdispersion factor D = 1 + (n-1)/(phi+1)
        // (D -> 1 recovers the binomial weight as phi -> Inf). Always positive.
        double mu = linkinv(eta, "logit");
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
        double a = mu * phi, b = (1.0 - mu) * phi;
        double dmu = mu * (1.0 - mu);
        double n = (double)n_trials;
        double grad = phi * (R::digamma(y + a) - R::digamma(a)
                             - R::digamma(n - y + b) + R::digamma(b)) * dmu;
        double D = 1.0 + (n - 1.0) / (phi + 1.0);
        return {grad, n * mu * (1.0 - mu) / D};
    }
    if (family == "t") {
        // Student-t location-scale (identity link): y ~ eta + phi * t_nu, with
        // nu = phi2 (NaN => the robust default kStudentTDf) and phi the scale.
        // Score is exact; the working weight is the constant Fisher information
        // (nu+1)/((nu+3) phi^2), which is positive and needs no Fisher fallback
        // (unlike the redescending observed information of the heavy tails).
        const double nu = std::isnan(phi2) ? kStudentTDf : phi2;
        double resid = y - eta;
        double grad = (nu + 1.0) * resid / (nu * phi * phi + resid * resid);
        return {grad, (nu + 1.0) / ((nu + 3.0) * phi * phi)};
    }
    if (family == "tweedie") {
        // Compound Poisson-gamma (log link), phi2 = power p in (1, 2). EDM
        // score through the log link, (y - mu)/(phi mu^(p-1)); expected
        // Fisher weight mu^(2-p)/phi. Always positive, no fallback needed.
        if (std::isnan(phi2)) {
            Rcpp::stop("family 'tweedie' needs phi2 (the variance power p).");
        }
        const double p = phi2;
        double mu = std::max(std::exp(eta), 1e-10);
        return {(y - mu) / (phi * std::pow(mu, p - 1.0)),
                std::pow(mu, 2.0 - p) / phi};
    }

    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    double dmu = mu_eta(eta, fl.link);
    if (fl.family == "binomial" || fl.family == "beta") {
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
    } else if (fl.family != "gaussian" && fl.family != "lognormal") {
        mu = std::max(mu, 1e-10);
    }

    double g = grad_mu(y, mu, phi, fl.family, n_trials);
    double V = variance_fn(mu, phi, fl.family, n_trials);
    return {g * dmu, dmu * dmu / V};
}

inline double log_lik_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "binomial") return log_lik_binomial((int)y, n_trials, eta);
    if (family == "poisson") return log_lik_poisson((int)y, eta);
    if (family == "neg_binomial_2") return log_lik_negbin((int)y, eta, phi);
    if (family == "beta_binomial") {
        double mu = linkinv(eta, "logit");
        mu = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
        double a = mu * phi, b = (1.0 - mu) * phi;
        double n = (double)n_trials, yi = (double)(int)y;
        return R::lchoose(n, yi)
             + R::lgammafn(yi + a) + R::lgammafn(n - yi + b) - R::lgammafn(n + a + b)
             - R::lgammafn(a) - R::lgammafn(b) + R::lgammafn(a + b);
    }
    if (family == "t") {
        const double nu = std::isnan(phi2) ? kStudentTDf : phi2;
        double r = (y - eta) / phi;
        return R::lgammafn((nu + 1.0) / 2.0) - R::lgammafn(nu / 2.0)
             - 0.5 * std::log(nu * M_PI * phi * phi)
             - 0.5 * (nu + 1.0) * std::log1p(r * r / nu);
    }
    if (family == "tweedie") {
        if (std::isnan(phi2)) {
            Rcpp::stop("family 'tweedie' needs phi2 (the variance power p).");
        }
        return log_lik_tweedie(y, std::exp(eta), phi, phi2);
    }

    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    if (fl.family == "binomial" || fl.family == "beta") {
        mu = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
    } else if (fl.family != "gaussian" && fl.family != "lognormal") {
        mu = std::max(mu, 1e-15);
    }
    return log_lik_mu(y, mu, phi, fl.family, n_trials);
}

// Grouped beta sufficient statistics. A set of n exchangeable Beta(mu*phi,
// (1-mu)*phi) observations sharing the SAME linear predictor (hence the same mu)
// enters the likelihood only through (n, sum log y, sum log(1-y)) -- the beta
// log-density is linear in log(y) and log(1-y). Collapsing them to one row
// carrying those sufficient statistics leaves the log-likelihood, gradient and
// (Fisher) Hessian pointwise unchanged. With n = 1, slog_y = log(y),
// slog_1my = log(1-y) these reduce exactly to the per-observation beta branch of
// log_lik_mu / grad_hess_for_family (same mu clamps), so the ungrouped path is
// byte-identical.
inline double log_lik_beta_grouped(double slog_y, double slog_1my, int n,
                                   double eta, double phi) {
    double mu = linkinv(eta, "logit");
    mu = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
    double a = mu * phi;
    double b = (1.0 - mu) * phi;
    return (a - 1.0) * slog_y + (b - 1.0) * slog_1my
           + (double)n * (R::lgammafn(phi) - R::lgammafn(a) - R::lgammafn(b));
}

inline GradHess grad_hess_beta_grouped(double slog_y, double slog_1my, int n,
                                       double eta, double phi) {
    double mu  = linkinv(eta, "logit");
    double dmu = mu_eta(eta, "logit");
    mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
    double mu_star = R::digamma(mu * phi) - R::digamma((1.0 - mu) * phi);
    double g_mu = phi * ((slog_y - slog_1my) - (double)n * mu_star);
    double V = variance_fn(mu, phi, "beta", 1);
    return { g_mu * dmu, (double)n * dmu * dmu / V };
}

// Interval-censored Gaussian latent (ordered-probit with KNOWN thresholds). The
// latent value is Normal(eta, sigma^2) and the observation records only that it
// fell in the half-open interval (lower, upper]; lower = -Inf / upper = +Inf are
// the open outer classes. With phi = sigma this is the discrete-class sibling of
// the gaussian arm: the log-density is the probability MASS of the observed
// class, P = Phi((upper - eta)/sigma) - Phi((lower - eta)/sigma), so the score is
// a genuine PMF over classes with no change-of-variable Jacobian. P(eta) is
// log-concave in eta (a convolution of the interval indicator with a log-concave
// density, Prekopa), so -d2 logP/d eta2 >= 0 and Newton needs no Fisher fallback.
//
//   d logP/d eta = (phi(zl) - phi(zu)) / (sigma * P)
//   -d2 logP/d eta2 = g^2 - (zl phi(zl) - zu phi(zu)) / (sigma^2 P)
//
// with zl = (lower - eta)/sigma, zu = (upper - eta)/sigma, phi the standard
// normal density (0 at +/-Inf, as is z phi(z)). The mass P is differenced in the
// accurate tail to avoid catastrophic cancellation when eta sits far from the
// class.
struct IntervalGaussian {
    double ll;        // log P
    double grad;      // d logP / d eta
    double neg_hess;  // -d2 logP / d eta2  (>= 0)
};

inline IntervalGaussian interval_gaussian_core(double lower, double upper,
                                               double eta, double sigma) {
    const double inv_s = 1.0 / sigma;
    const bool lo_open = !R_finite(lower);   // -Inf
    const bool hi_open = !R_finite(upper);   // +Inf
    const double zl = lo_open ? R_NegInf : (lower - eta) * inv_s;
    const double zu = hi_open ? R_PosInf : (upper - eta) * inv_s;

    // Mass P = Phi(zu) - Phi(zl), differenced in the accurate tail.
    double P;
    if (lo_open && hi_open) {
        P = 1.0;
    } else if (lo_open) {
        P = R::pnorm(zu, 0.0, 1.0, 1, 0);              // Phi(zu)
    } else if (hi_open) {
        P = R::pnorm(zl, 0.0, 1.0, 0, 0);              // 1 - Phi(zl)
    } else if (zu <= 0.0) {
        P = R::pnorm(zu, 0.0, 1.0, 1, 0) - R::pnorm(zl, 0.0, 1.0, 1, 0);
    } else if (zl >= 0.0) {
        P = R::pnorm(zl, 0.0, 1.0, 0, 0) - R::pnorm(zu, 0.0, 1.0, 0, 0);
    } else {
        P = R::pnorm(zu, 0.0, 1.0, 1, 0) - R::pnorm(zl, 0.0, 1.0, 1, 0);
    }
    const double Psafe = P > 1e-300 ? P : 1e-300;

    const double pl = lo_open ? 0.0 : R::dnorm(zl, 0.0, 1.0, 0);
    const double pu = hi_open ? 0.0 : R::dnorm(zu, 0.0, 1.0, 0);
    // z * phi(z) -> 0 as |z| -> Inf. Guard the Inf * 0 = NaN that arises when a
    // FINITE bound sits far from eta: zl / zu overflow to +/-Inf while the
    // density underflows to 0. The analytic limit of the product is 0.
    const double zpl = (lo_open || pl == 0.0 || !R_finite(zl)) ? 0.0 : zl * pl;
    const double zpu = (hi_open || pu == 0.0 || !R_finite(zu)) ? 0.0 : zu * pu;

    const double Pprime = (pl - pu) * inv_s;                       // dP/d eta
    const double Pdd    = (zpl - zpu) * inv_s * inv_s;             // d2P/d eta2
    const double g  = Pprime / Psafe;
    double nh = g * g - Pdd / Psafe;
    if (nh < 1e-12) nh = 1e-12;   // log-concave; guard roundoff at the flat tail

    return { std::log(Psafe), g, nh };
}

inline double log_lik_interval_gaussian(double lower, double upper,
                                        double eta, double sigma) {
    return interval_gaussian_core(lower, upper, eta, sigma).ll;
}

inline GradHess grad_hess_interval_gaussian(double lower, double upper,
                                            double eta, double sigma) {
    const IntervalGaussian r = interval_gaussian_core(lower, upper, eta, sigma);
    return { r.grad, r.neg_hess };
}

// Upper-truncated Gaussian latent. The latent response is
// Normal(eta, sigma^2) CONDITIONED on y <= u, a known upper bound on the response
// scale (+Inf => no truncation). This is a truncated GAUSSIAN: it operates on the
// response it is given, exactly as the plain gaussian family does, so a consumer
// gets a truncated-LOGNORMAL by feeding it log(cover) with u = log(ceiling) (the
// same way the cover hurdle already fits its lognormal arm with family="gaussian"
// on log-cover; the -log y Jacobian is the consumer's, added outside the engine).
// It is the continuous-density counterpart of interval_gaussian: the interval
// family scores the probability MASS of a censored class, this scores a truncated
// DENSITY -- the Gaussian density divided by the retained mass Phi((u - eta)/sigma).
// Distinct objects: truncation conditions the sampling law, censoring records an
// interval.
//
//   z = (y - eta)/sigma, a = (u - eta)/sigma, lambda = phi(a)/Phi(a)
//   ll            = -0.5 log(2 pi sigma^2) - 0.5 z^2 - log Phi(a)
//   d logf/d eta  = (z + lambda) / sigma
//   -d2/d eta2    = (1 - lambda (a + lambda)) / sigma^2
//
// As u -> +Inf (a -> +Inf, lambda -> 0) this reduces EXACTLY to the gaussian arm.
// 0 < lambda(a + lambda) < 1 for finite a (the truncated-normal variance factor),
// so -d2 logf/d eta2 > 0 and Newton needs no Fisher fallback -- the same
// log-concavity interval_gaussian relies on. lambda is formed in log space so it
// stays finite in deep truncation (a -> -Inf, predicted mean far above the
// bound); the curvature is floored at the flat far tail, mirroring
// interval_gaussian_core.
struct TruncatedGaussian {
    double ll;        // log density of the truncated Gaussian (no response Jacobian)
    double grad;      // d logf / d eta
    double neg_hess;  // -d2 logf / d eta2  (>= 0)
};

inline TruncatedGaussian truncated_gaussian_core(double y, double u_upper,
                                                 double eta, double sigma) {
    const double inv_s = 1.0 / sigma;
    const double z = (y - eta) * inv_s;

    double a, logPhi_a, lambda;
    if (!R_finite(u_upper)) {            // +Inf bound => untruncated gaussian
        a = R_PosInf;
        logPhi_a = 0.0;
        lambda   = 0.0;
    } else {
        a = (u_upper - eta) * inv_s;
        logPhi_a = R::pnorm(a, 0.0, 1.0, 1, 1);                   // log Phi(a)
        lambda   = std::exp(R::dnorm(a, 0.0, 1.0, 1) - logPhi_a); // phi(a)/Phi(a), stable
    }

    const double ll = -0.5 * std::log(2.0 * M_PI * sigma * sigma)
                      - 0.5 * z * z - logPhi_a;
    const double grad = (z + lambda) * inv_s;
    // lambda * (a + lambda): guard the 0 * Inf at a = +Inf (lambda = 0 there).
    const double curv_term = (lambda == 0.0) ? 0.0 : lambda * (a + lambda);
    double nh = (1.0 - curv_term) * inv_s * inv_s;
    if (nh < 1e-12) nh = 1e-12;   // log-concave; guard roundoff at the flat tail

    return { ll, grad, nh };
}

inline double log_lik_truncated_gaussian(double y, double u_upper,
                                         double eta, double sigma) {
    return truncated_gaussian_core(y, u_upper, eta, sigma).ll;
}

inline GradHess grad_hess_truncated_gaussian(double y, double u_upper,
                                             double eta, double sigma) {
    const TruncatedGaussian r = truncated_gaussian_core(y, u_upper, eta, sigma);
    return { r.grad, r.neg_hess };
}

inline double compute_total_log_lik(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericVector& eta, int N,
    const std::string& family, double phi, int n_threads
) {
    // This entry carries no phi2, so tweedie (which requires the variance
    // power) can never evaluate here: stop on the calling thread BEFORE the
    // parallel region — the per-observation stop inside an omp reduction
    // body would escape the structured block, which is std::terminate.
    if (family == "tweedie") {
        Rcpp::stop("family 'tweedie' needs phi2 (the variance power p); "
                   "route it through the LikelihoodSpec path, which carries "
                   "phi2.");
    }
    double log_lik = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:log_lik) schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N; i++) {
        log_lik += log_lik_for_family(y[i], n_trials[i], eta[i], family, phi);
    }
    return log_lik;
}

// Data log-likelihood as a functor of the current linear predictor eta.
//
// The shared Newton loop (laplace_newton_solve_ll) reads the data log-lik
// only through a callable `double(const Rcpp::NumericVector& eta)`, so the
// likelihood is no longer baked into the loop as a family enum. This functor
// is the built-in-family value of that callable: a single source of truth that
// the family-enum mode finders pass while the LikelihoodSpec path passes its
// own spec.ll_double-backed functor. The borrowed pointers must outlive the
// fit.
struct FamilyLogLik {
    const Rcpp::NumericVector* y = nullptr;
    const Rcpp::IntegerVector* n_trials = nullptr;
    int N = 0;
    std::string family;
    double phi = 1.0;
    int n_threads = 1;

    double operator()(const Rcpp::NumericVector& eta) const {
        return compute_total_log_lik(*y, *n_trials, eta, N, family, phi,
                                     n_threads);
    }
};

} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_LINK_H
