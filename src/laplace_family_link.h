// laplace_family_link.h
// Family/link dispatch and per-observation likelihood utilities.

#ifndef TULPA_LAPLACE_FAMILY_LINK_H
#define TULPA_LAPLACE_FAMILY_LINK_H

#include "laplace_likelihoods.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

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

inline double linkinv(double eta, const std::string& link) {
    if (link == "identity") return eta;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") return 1.0 / eta;
    if (link == "logit") {
        if (eta > 0) return 1.0 / (1.0 + std::exp(-eta));
        double e = std::exp(eta);
        return e / (1.0 + e);
    }
    if (link == "probit") return R::pnorm(eta, 0.0, 1.0, 1, 0);
    if (link == "cauchit") return 0.5 + std::atan(eta) / M_PI;
    if (link == "cloglog") return 1.0 - std::exp(-std::exp(eta));
    if (link == "sqrt") return eta * eta;
    if (link == "1mu2") return 1.0 / std::sqrt(eta);
    return tulpa_linalg::safe_exp(eta);
}

inline double mu_eta(double eta, const std::string& link) {
    if (link == "identity") return 1.0;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") return -1.0 / (eta * eta);
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
    if (link == "1mu2") return -0.5 / (eta * std::sqrt(eta));
    return tulpa_linalg::safe_exp(eta);
}

inline double variance_fn(double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") return phi * phi;
    if (family == "binomial") return n_trials * mu * (1.0 - mu);
    if (family == "poisson") return mu;
    if (family == "neg_binomial_2") return mu + mu * mu / phi;
    if (family == "gamma") return mu * mu / phi;
    if (family == "inverse_gaussian") return phi * mu * mu * mu;
    if (family == "beta") {
        // Working variance: V s.t. dmu^2 / V = Fisher info per obs on eta.
        // Fisher info = phi^2 * (trigamma(mu*phi) + trigamma((1-mu)*phi)) * dmu^2
        // (Ferrari & Cribari-Neto 2004); see dev_notes/beta_likelihood.md.
        double tg = R::trigamma(mu * phi) + R::trigamma((1.0 - mu) * phi);
        return 1.0 / (phi * phi * tg);
    }
    return mu;
}

inline double grad_mu(double y, double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") return (y - mu) / (phi * phi);
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
    if (family == "binomial") {
        double p = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
        return (int)y * std::log(p) + (n_trials - (int)y) * std::log(1.0 - p);
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

struct GradHess {
    double grad;
    double neg_hess;
};

inline GradHess grad_hess_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi
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

    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    double dmu = mu_eta(eta, fl.link);
    if (fl.family == "binomial" || fl.family == "beta") {
        mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
    } else if (fl.family != "gaussian") {
        mu = std::max(mu, 1e-10);
    }

    double g = grad_mu(y, mu, phi, fl.family, n_trials);
    double V = variance_fn(mu, phi, fl.family, n_trials);
    return {g * dmu, dmu * dmu / V};
}

inline double log_lik_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi
) {
    if (family == "binomial") return log_lik_binomial((int)y, n_trials, eta);
    if (family == "poisson") return log_lik_poisson((int)y, eta);
    if (family == "neg_binomial_2") return log_lik_negbin((int)y, eta, phi);

    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    if (fl.family == "binomial" || fl.family == "beta") {
        mu = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
    } else if (fl.family != "gaussian") {
        mu = std::max(mu, 1e-15);
    }
    return log_lik_mu(y, mu, phi, fl.family, n_trials);
}

inline double compute_total_log_lik(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericVector& eta, int N,
    const std::string& family, double phi, int n_threads
) {
    double log_lik = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:log_lik) schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N; i++) {
        log_lik += log_lik_for_family(y[i], n_trials[i], eta[i], family, phi);
    }
    return log_lik;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_LINK_H
