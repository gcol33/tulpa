// laplace_family_link.h
// Family/link dispatch and per-observation likelihood utilities.
//
// `phi` throughout this header is the family's dispersion read in the SD
// convention: the residual SD for gaussian and lognormal, and the family's own
// shape / size / precision / scale elsewhere. glmm_oracle.h reads the same slot
// in the VARIANCE convention, so the two disagree wherever .phi_is_variance()
// is true (gaussian, lognormal); R converts at each boundary with
// .phi_to_kernel() and .phi_to_registry().

#ifndef TULPA_LAPLACE_FAMILY_LINK_H
#define TULPA_LAPLACE_FAMILY_LINK_H

#include "laplace_likelihoods.h"
#include "linalg_fast.h"
#include "omp_threads.h"          // tulpa_parallel_sum (serial route at one thread)
#include "tulpa/portable_math.h"   // thread-safe digamma / trigamma
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

// P(Y > 0) held off zero. Every ratio in TruncationTerm divides by it, and a
// zero-truncated family reaches p = 0 in the limit of a vanishing mean.
constexpr double kProbFloor = 1e-300;

// mu floor on the tweedie log link, which reads mu^(p-1) in a denominator and
// mu^(2-p) in the weight; at mu = 0 one of the two diverges for every p in
// (1, 2).
constexpr double kTweedieMuFloor = 1e-10;

// Width in log space of the tweedie event-count series: summation stops once a
// term falls this many nats below the running peak, where it can no longer move
// the log-sum-exp at double precision.
constexpr double kTweedieSeriesNats = 37.0;

// Curvature floor on the censored / truncated gaussian arms. Both cores are
// log-concave in eta, so a non-positive reading is roundoff at a flat tail; the
// floor is what keeps the Newton Hessian PD there.
constexpr double kCensoredCurvFloor = 1e-12;

// The Student-t degrees of freedom in force: phi2 where the second dispersion
// channel carries one, kStudentTDf where it does not.
inline double student_t_df(double phi2) {
    return std::isnan(phi2) ? kStudentTDf : phi2;
}

// The tweedie variance power. Required: mu^p has no sensible default, so a
// missing phi2 is an error rather than a substitution.
inline double tweedie_power(double phi2) {
    if (std::isnan(phi2)) {
        Rcpp::stop("family 'tweedie' needs phi2 (the variance power p).");
    }
    return phi2;
}

// The tweedie power together with the floored mean at eta, which is what every
// branch reading the log link needs.
struct TweedieParams {
    double p;
    double mu;
};

inline TweedieParams tweedie_params(double phi2, double eta) {
    return {tweedie_power(phi2), std::max(std::exp(eta), kTweedieMuFloor)};
}

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

// Which branch of the per-observation family ladder a family code selects.
// GENERIC is the mu-space route that reads FamilyResolved::fl.
enum class FamilyKind : int {
    BINOMIAL, POISSON, NEG_BINOMIAL_2, NEG_BINOMIAL_1,
    TRUNCATED_POISSON, TRUNCATED_NEG_BINOMIAL_2, BETA_BINOMIAL,
    STUDENT_T, TWEEDIE, GENERIC
};

// A family code resolved once, so the per-observation entry points below carry
// no string work at all (gcol33/tulpa#372). The ladders they replace compared
// the code against every special family in turn and then called
// parse_family_link, which builds two std::strings and a concatenated prefix
// per call -- per observation, per objective evaluation. Both entry points take
// this struct; the std::string overloads resolve it and forward, so the ladder
// itself is written once.
struct FamilyResolved {
    FamilyKind kind = FamilyKind::GENERIC;
    FamilyLink fl;              // the mu-space route's family and link
    std::string code;           // the family code as supplied
    bool positive_eta_domain = false;   // link_has_positive_eta_domain(fl.link)
};

// Links carried on the open half-line eta > 0:
//
//   inverse   mu = 1/eta          singular at eta = 0
//   1mu2      mu = 1/sqrt(eta)    singular at eta = 0
//   sqrt      mu = eta^2          finite at eta = 0, but folds the two branches
//                                 of the parabola onto the same mean: eta and
//                                 -eta are observationally identical, so the
//                                 mode comes in mirror pairs and mu_eta(0) = 0
//                                 makes the Hessian singular between them.
//
// All three are fitted on eta > 0 -- for the first two because mu is undefined
// otherwise, for sqrt because that is the branch that is identified. This is the
// branch glm()'s starting values put you on for each of them.
inline bool link_has_positive_eta_domain(const std::string& link) {
    return link == "inverse" || link == "1mu2" || link == "sqrt";
}

// Which branch of the ladders below a family code selects. Allocation-free:
// this is the same sequence of string comparisons the ladders opened with, so a
// caller that classifies once and then evaluates per observation pays it once,
// and a caller that still passes a string pays exactly what it paid before.
inline FamilyKind family_kind(const std::string& code) {
    if (code == "binomial")                      return FamilyKind::BINOMIAL;
    if (code == "poisson")                       return FamilyKind::POISSON;
    if (code == "neg_binomial_2")                return FamilyKind::NEG_BINOMIAL_2;
    if (code == "neg_binomial_1")                return FamilyKind::NEG_BINOMIAL_1;
    if (code == "truncated_poisson")             return FamilyKind::TRUNCATED_POISSON;
    if (code == "truncated_neg_binomial_2")      return FamilyKind::TRUNCATED_NEG_BINOMIAL_2;
    if (code == "beta_binomial")                 return FamilyKind::BETA_BINOMIAL;
    if (code == "t")                             return FamilyKind::STUDENT_T;
    if (code == "tweedie")                       return FamilyKind::TWEEDIE;
    return FamilyKind::GENERIC;
}

inline FamilyResolved resolve_family(const std::string& code) {
    FamilyResolved r;
    r.kind = family_kind(code);
    r.fl = parse_family_link(code);
    r.code = code;
    r.positive_eta_domain = link_has_positive_eta_domain(r.fl.link);
    return r;
}

// The eta-independent per-observation term of the families whose log-density
// splits into `kernel + const` (laplace_likelihoods.h), and zero for every
// other kind. A fit evaluates this once per observation and hands the value
// back to log_lik_for_family, which adds it in the same place and the same
// order the full density did (gcol33/tulpa#372).
inline double log_lik_const_for_kind(double y, int n_trials, FamilyKind kind) {
    switch (kind) {
    case FamilyKind::BINOMIAL:
        return log_lik_binomial_const((int)y, n_trials);
    case FamilyKind::POISSON:
    case FamilyKind::TRUNCATED_POISSON:   // its base density is the Poisson one
        return log_lik_poisson_const((int)y);
    default:
        return 0.0;
    }
}

// Whether eta lies in the link's domain.
//
// The data log-likelihood is -Inf outside it (log_lik_for_family below), which
// turns the domain into a barrier the Newton loop cannot step across: the
// penalized objective is -Inf there, the line-search acceptance test rejects it,
// and line_search_backtrack refuses a non-finite objective even at its final
// trial. Every iterate the solver holds is therefore interior, so linkinv /
// mu_eta / mu_eta2 are only ever evaluated at eta > 0.
//
// Feasibility of the FIRST iterate is not automatic -- the default latent start
// is x = 0, hence eta = 0, which is on the boundary for all three of these
// links. make_start_feasible() in laplace_newton_loop.h shifts the process
// intercepts so the solve begins interior.
inline bool link_eta_in_domain(double eta, const std::string& link) {
    return eta > 0.0 || !link_has_positive_eta_domain(link);
}

// Defensive floor for the singular links, NOT a modelling choice. The barrier
// keeps every accepted iterate interior, so this is reached only where the
// engine steps without consulting the objective -- the failed-factorization
// fallback in laplace_newton.h. It is here so that path yields a large finite
// number rather than propagating Inf/NaN into a Cholesky.
//
// The value returned is not meaningful: below the floor mu is CONSTANT in eta
// while mu_eta and mu_eta2 report -1e20 and 2e30, so the value and its reported
// derivatives describe different functions. Optimizing against this region reads
// garbage, which is precisely what the barrier exists to prevent -- do not treat
// the floor as an extension of the link.
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
    if (link == "probit") return tulpa::math::portable_pnorm(eta);
    if (link == "cauchit") return 0.5 + std::atan(eta) / M_PI;
    // -expm1(-a) keeps full relative accuracy for every a > 0; 1 - exp(-a)
    // cancels as a -> 0 and rounds to exactly 0 once exp(eta) falls below the
    // double spacing at 1, which is mu = 0, outside the support of every
    // consumer of this link.
    if (link == "cloglog") return -std::expm1(-std::exp(eta));
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
    if (link == "probit") return tulpa::math::portable_dnorm(eta);
    if (link == "cauchit") return 1.0 / (M_PI * (1.0 + eta * eta));
    if (link == "cloglog") return std::exp(eta - std::exp(eta));
    if (link == "sqrt") return 2.0 * eta;
    if (link == "1mu2") { double e = safe_pos_eta(eta); return -0.5 / (e * std::sqrt(e)); }
    return tulpa_linalg::safe_exp(eta);
}

// d2 mu / d eta2, the companion to mu_eta() above.
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
    if (link == "probit") return -eta * tulpa::math::portable_dnorm(eta);
    if (link == "cauchit") {
        const double d = 1.0 + eta * eta;
        return -2.0 * eta / (M_PI * d * d);
    }
    if (link == "cloglog") {
        const double ee = std::exp(eta);
        return std::exp(eta - ee) * (-std::expm1(eta));
    }
    if (link == "sqrt") return 2.0;
    if (link == "1mu2") { double e = safe_pos_eta(eta); return 0.75 / (e * e * std::sqrt(e)); }
    return tulpa_linalg::safe_exp(eta);
}

// d3 mu / d eta3, the companion to mu_eta2() above. Needed by the generic
// mu-space routes of curvature_deta2_for_family() and the observed curvature's
// eta-derivative.
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
    if (link == "probit") return (eta * eta - 1.0) * tulpa::math::portable_dnorm(eta);
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

// A family name that reaches the mu-space dispatch below without matching a
// branch is a programming error, not a data condition: every fitting path is
// gated by the R registry, so an unmatched name means a family was registered
// in R without a compiled counterpart. These three functions used to end in
// the Poisson branch, which made that mistake silent -- the model fitted, and
// fitted the wrong likelihood. Failing here turns it into a build-time-visible
// error the first time the family is exercised.
[[noreturn]] inline void unknown_family_stop(const char* fn,
                                             const std::string& family) {
    Rcpp::stop("%s: no compiled implementation for family '%s'. Families "
               "defined in the R registry must be added to "
               "laplace_family_link.h before they can be fitted.",
               fn, family.c_str());
}

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
// The families the mu-space ladders below implement. Every one of variance_fn,
// grad_mu, dgrad_mu_dmu, d2grad_mu_dmu2 and log_lik_mu carries exactly these
// branches and calls unknown_family_stop on anything else, so the list lives
// here rather than being restated by each caller that has to ask in advance.
inline bool mu_space_family_supported(const std::string& base) {
    return base == "gaussian" || base == "lognormal" || base == "binomial" ||
           base == "poisson" || base == "neg_binomial_2" || base == "gamma" ||
           base == "inverse_gaussian" || base == "beta";
}

constexpr double kMuFloor = 1e-15;

inline double clamp_mu_unit(double mu) {
    return std::max(std::min(mu, 1.0 - kMuFloor), kMuFloor);
}

inline double clamp_mu_for_family(double mu, const std::string& family) {
    if (family == "binomial" || family == "beta") return clamp_mu_unit(mu);
    if (family == "gaussian" || family == "lognormal") return mu;
    return std::max(mu, kMuFloor);
}

// The WORKING variance: the V for which dmu^2 / V is the Fisher information per
// observation on eta. This is not Var(y) wherever the two differ -- see the
// binomial and beta arms below -- and nothing consumes it as a response
// variance; its only callers are the dmu^2 / V compositions in
// grad_hess_for_family, log_lik_beta_grouped's weight, and the quotient rule in
// curvature_deta_for_family.
inline double variance_fn(double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") return phi * phi;
    if (family == "lognormal") return phi * phi;
    // y ~ Bin(n, mu) has Var(y) = n mu (1-mu), but the Fisher information on eta
    // is n (dmu/deta)^2 / (mu (1-mu)), so the V that dmu^2 / V must divide by is
    // mu (1-mu) / n -- the proportion-scale variance function, as in glm's
    // binomial()$variance on a proportion response. Returning the RESPONSE
    // variance here made the working weight a factor of n^2 too small for every
    // non-canonical binomial link (probit, cloglog, cauchit, log) at n > 1. The
    // canonical logit link never reached this: grad_hess_for_family answers
    // binomial from neg_hess_log_lik_binomial before the generic route, which is
    // why only the suffixed forms were affected, and n = 1 hid it there.
    if (family == "binomial") return mu * (1.0 - mu) / n_trials;
    if (family == "poisson") return mu;
    if (family == "neg_binomial_2") return mu + mu * mu / phi;
    if (family == "gamma") return mu * mu / phi;
    if (family == "inverse_gaussian") return phi * mu * mu * mu;
    if (family == "beta") {
        // Working variance: V s.t. dmu^2 / V = Fisher info per obs on eta.
        // Fisher info = phi^2 * (trigamma(mu*phi) + trigamma((1-mu)*phi)) * dmu^2
        // (Ferrari & Cribari-Neto 2004).
        double tg = tulpa::math::portable_trigamma(mu * phi) + tulpa::math::portable_trigamma((1.0 - mu) * phi);
        return 1.0 / (phi * phi * tg);
    }
    unknown_family_stop("variance_fn", family);
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
        double mu_star = tulpa::math::portable_digamma(mu * phi) - tulpa::math::portable_digamma((1.0 - mu) * phi);
        return phi * (y_star - mu_star);
    }
    unknown_family_stop("grad_mu", family);
}

// d grad_mu / d mu and its next derivative: the mu-space curvature ladder that
// turns the score above into the OBSERVED curvature of the same log density.
//
// With l(eta) = L(mu(eta)), L' = grad_mu and u = mu_eta,
//
//     -l''  = -(L'' u^2 + L' u1)
//     -l''' = -(L''' u^3 + 3 L'' u u1 + L' u2)
//
// so these two are all the generic mu-space route needs on top of what it
// already carries. They differentiate grad_mu ITSELF, not the working weight
// dmu^2 / V that grad_hess_for_family returns -- for a family whose score is
// the exponential-dispersion form the two ladders meet in expectation, and for
// beta they do not meet at all.
inline double dgrad_mu_dmu(double y, double mu, double phi,
                           const std::string& family, int n_trials) {
    if (family == "gaussian" || family == "lognormal") {
        return -1.0 / (phi * phi);
    }
    if (family == "binomial") {
        const double P = mu * (1.0 - mu), Pp = 1.0 - 2.0 * mu;
        const double r = (double)(int)y - (double)n_trials * mu;
        return -(double)n_trials / P - r * Pp / (P * P);
    }
    if (family == "poisson") return -(double)(int)y / (mu * mu);
    if (family == "neg_binomial_2") {
        const double yi = (double)(int)y;
        return -yi / (mu * mu) + (yi + phi) / ((mu + phi) * (mu + phi));
    }
    if (family == "gamma") return -phi * (2.0 * y - mu) / (mu * mu * mu);
    if (family == "inverse_gaussian") {
        const double m2 = mu * mu;
        return -(3.0 * y - 2.0 * mu) / (phi * m2 * m2);
    }
    if (family == "beta") {
        return -phi * phi * (tulpa::math::portable_trigamma(mu * phi)
                             + tulpa::math::portable_trigamma((1.0 - mu) * phi));
    }
    unknown_family_stop("dgrad_mu_dmu", family);
}

inline double d2grad_mu_dmu2(double y, double mu, double phi,
                             const std::string& family, int n_trials) {
    if (family == "gaussian" || family == "lognormal") return 0.0;
    if (family == "binomial") {
        // P = mu(1-mu), P' = 1-2mu, P'' = -2; g = (y - n mu) / P.
        const double P = mu * (1.0 - mu), Pp = 1.0 - 2.0 * mu;
        const double r = (double)(int)y - (double)n_trials * mu;
        return 2.0 * (double)n_trials * Pp / (P * P)
             - r * (-2.0 * P - 2.0 * Pp * Pp) / (P * P * P);
    }
    if (family == "poisson") return 2.0 * (double)(int)y / (mu * mu * mu);
    if (family == "neg_binomial_2") {
        const double yi = (double)(int)y, s = mu + phi;
        return 2.0 * yi / (mu * mu * mu) - 2.0 * (yi + phi) / (s * s * s);
    }
    if (family == "gamma") {
        const double m2 = mu * mu;
        return phi * (6.0 * y - 2.0 * mu) / (m2 * m2);
    }
    if (family == "inverse_gaussian") {
        const double m2 = mu * mu;
        return (12.0 * y - 6.0 * mu) / (phi * m2 * m2 * mu);
    }
    if (family == "beta") {
        const double p3 = phi * phi * phi;
        return -p3 * (tulpa::math::portable_tetragamma(mu * phi)
                      - tulpa::math::portable_tetragamma((1.0 - mu) * phi));
    }
    unknown_family_stop("d2grad_mu_dmu2", family);
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
        double p = clamp_mu_unit(mu);
        // lchoose keeps this a true log-density, matching the poisson arm below
        // (which keeps its lgamma(y+1)), dbinom(), and the other kernels.
        return (int)y * std::log(p) + (n_trials - (int)y) * std::log(1.0 - p)
               + tulpa::math::portable_lchoose((double) n_trials, y);
    }
    if (family == "poisson") {
        double safe_mu = std::max(mu, kMuFloor);
        return (int)y * std::log(safe_mu) - safe_mu - tulpa::math::portable_lgamma((int)y + 1.0);
    }
    if (family == "neg_binomial_2") {
        double safe_mu = std::max(mu, kMuFloor);
        return tulpa::math::portable_lgamma((int)y + phi) - tulpa::math::portable_lgamma(phi) - tulpa::math::portable_lgamma((int)y + 1.0)
               + phi * std::log(phi / (safe_mu + phi))
               + (int)y * std::log(safe_mu / (safe_mu + phi));
    }
    if (family == "gamma") {
        return phi * std::log(phi) - tulpa::math::portable_lgamma(phi) + (phi - 1.0) * std::log(y)
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
        return tulpa::math::portable_lgamma(phi) - tulpa::math::portable_lgamma(a) - tulpa::math::portable_lgamma(b)
               + (a - 1.0) * std::log(y) + (b - 1.0) * std::log(1.0 - y);
    }
    unknown_family_stop("log_lik_mu", family);
}

// Tweedie compound Poisson-gamma log-density (1 < p < 2), log link upstream:
// mean mu, dispersion phi, power p. Exact zero mass exp(-lambda); positive y
// via the event-count series (Dunn & Smyth 2005), log-sum-exp'd from the
// dominating index j_max = y^(2-p) / (phi (2-p)) until terms fall 37 nats
// below the running peak. Mirrors .tweedie_loglik in R/family_loglik.R.
inline double log_lik_tweedie(double y, double mu, double phi, double p) {
    mu = std::max(mu, kTweedieMuFloor);
    const double lam = std::pow(mu, 2.0 - p) / (phi * (2.0 - p));
    if (y < 0.0) return R_NegInf;
    if (y <= 0.0) return -lam;
    const double a  = (2.0 - p) / (p - 1.0);
    const double b  = std::pow(mu, 1.0 - p) / (phi * (p - 1.0));
    const double la = std::log(lam), lb = std::log(b), ly = std::log(y);
    auto logterm = [&](double n) {
        return n * la - tulpa::math::portable_lgamma(n + 1.0) + n * a * lb
             + (n * a - 1.0) * ly - tulpa::math::portable_lgamma(n * a);
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
        if (lt < lmax - kTweedieSeriesNats) break;
    }
    for (int n = n0 - 1; n >= 1; --n) {
        const double lt = logterm((double)n);
        terms.push_back(lt);
        if (lt > lmax) lmax = lt;
        if (lt < lmax - kTweedieSeriesNats) break;
    }
    double s = 0.0;
    for (double lt : terms) s += std::exp(lt - lmax);
    return lmax + std::log(s) - lam - b * y;
}

// ---------------------------------------------------------------------------
// Zero-truncated count families.
//
// Both shipped truncated families condition an untruncated count law on y >= 1,
// so both subtract the same retained-mass term log P(Y > 0) from the density
// and the same derivatives of it from the score and curvature. Writing
// P(Y > 0) = 1 - exp(-a) makes the two differ only in a and its eta-derivatives:
//
//   truncated_poisson         a = mu,                 da = mu,   d2a = mu,
//                                                     d3a = mu
//   truncated_neg_binomial_2  a = phi log1p(mu/phi),  da = phi mu / (phi + mu),
//                                                     d2a = phi^2 mu / (phi + mu)^2,
//                                                     d3a = phi^2 mu (phi - mu) / (phi + mu)^3
//
// so the pieces below are computed once from (a, da, d2a) and reused by the
// density, the working weight and the observed curvature. That keeps the three
// from drifting apart, which is the failure mode a per-family copy invites.
// Mirrors truncated_poisson / truncated_neg_binomial_2 in R/family_loglik.R.
struct TruncationTerm {
    double q;         // P(Y = 0) = exp(-a)
    double p;         // P(Y > 0) = 1 - exp(-a)
    double p_safe;    // p floored at kProbFloor: the denominator of every ratio
    double dp;        // d P(Y > 0) / d eta
    double d2p;       // d2 P(Y > 0) / d eta2
    double log_p;     // log P(Y > 0)
    double dlog_p;    // d log P(Y > 0) / d eta
    double d2log_p;   // d2 log P(Y > 0) / d eta2
    double e_weight;  // expected curvature contribution of the truncation
};

inline bool is_zero_truncated(const std::string& family) {
    return family == "truncated_poisson" ||
           family == "truncated_neg_binomial_2";
}

inline TruncationTerm truncation_term(double a, double da, double d2a) {
    TruncationTerm t;
    t.q = std::exp(-a);                         // P(Y = 0)
    t.p = -std::expm1(-a);
    t.p_safe = t.p > kProbFloor ? t.p : kProbFloor;
    const double q = t.q;
    const double psafe = t.p_safe;
    t.log_p = tulpa::math::log1m_exp(a);
    const double dp  = q * da;
    const double d2p = q * (d2a - da * da);
    t.dp  = dp;
    t.d2p = d2p;
    t.dlog_p  = dp / psafe;
    t.d2log_p = (d2p * psafe - dp * dp) / (psafe * psafe);
    // Var(y | y > 0) through the link: positive for every a > 0, so the Newton
    // Hessian stays PD without a Fisher fallback (unlike the observed form,
    // which carries y).
    t.e_weight = da / psafe - q * da * da / (psafe * psafe);
    return t;
}

// d3 log P(Y > 0) / d eta3, the next rung on the same (q, p) the term carries.
// Separate from TruncationTerm because only the observed-curvature ladder needs
// it and it is the one piece that reads the third shape derivative d3a.
inline double truncation_d3log_p(const TruncationTerm& t,
                                 double da, double d2a, double d3a) {
    const double ps  = t.p_safe;
    const double d3p = t.q * (d3a - 3.0 * da * d2a + da * da * da);
    return d3p / ps
         - 3.0 * t.dp * t.d2p / (ps * ps)
         + 2.0 * t.dp * t.dp * t.dp / (ps * ps * ps);
}

// (a, da, d2a) for a truncated family at the current eta, and the third
// eta-derivative d3a when a non-null pointer is passed. `mu` is the untruncated
// mean; phi is the NB size and is ignored by the Poisson arm.
inline void truncation_shape(FamilyKind kind, double mu, double phi,
                             double* a, double* da, double* d2a,
                             double* d3a = nullptr) {
    if (kind == FamilyKind::TRUNCATED_POISSON) {
        *a = mu; *da = mu; *d2a = mu;
        if (d3a) *d3a = mu;
        return;
    }
    const double s = phi + mu;
    *a   = phi * std::log1p(mu / phi);
    *da  = phi * mu / s;
    *d2a = phi * phi * mu / (s * s);
    if (d3a) *d3a = phi * phi * mu * (phi - mu) / (s * s * s);
}

inline void truncation_shape(const std::string& family, double mu, double phi,
                             double* a, double* da, double* d2a,
                             double* d3a = nullptr) {
    truncation_shape(family == "truncated_poisson"
                         ? FamilyKind::TRUNCATED_POISSON
                         : FamilyKind::TRUNCATED_NEG_BINOMIAL_2,
                     mu, phi, a, da, d2a, d3a);
}

struct GradHess {
    double grad;
    double neg_hess;
};

// Score + Newton working weight, with the family already classified. `fl` is
// read only on the GENERIC (mu-space) branch and may be null for every other
// kind; it is what the two public entry points below differ in supplying --
// the resolved one hands over a FamilyLink parsed once per fit, the string one
// parses it here, and the ladder itself is written only here.
inline GradHess grad_hess_for_family_core(
    double y, int n_trials, double eta, FamilyKind kind, const FamilyLink* fl,
    double phi, double phi2
) {
    if (kind == FamilyKind::BINOMIAL) {
        return {grad_log_lik_binomial((int)y, n_trials, eta),
                neg_hess_log_lik_binomial((int)y, n_trials, eta)};
    }
    if (kind == FamilyKind::POISSON) {
        return {grad_log_lik_poisson((int)y, eta),
                neg_hess_log_lik_poisson((int)y, eta)};
    }
    if (kind == FamilyKind::NEG_BINOMIAL_2) {
        return {grad_log_lik_negbin((int)y, eta, phi),
                neg_hess_log_lik_negbin((int)y, eta, phi)};
    }
    if (kind == FamilyKind::NEG_BINOMIAL_1) {
        // Score is exact. The working weight is the quasi-likelihood IRLS
        // weight (dmu/deta)^2 / V(mu) = mu / (1 + phi), the form glmmTMB uses;
        // it is positive everywhere, so Newton needs no Fisher fallback. It is
        // NOT the Fisher information -- because the shape r = mu/phi moves with
        // the mean, the exact expected curvature is r^2 sum_{k>=0} P(y>k)/(r+k)^2,
        // a series with no elementary closed form. The observed curvature is
        // available exactly, through obs_grad_hess_for_family below.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        const double r  = mu / phi;
        const double grad = r * (tulpa::math::portable_digamma(y + r)
                                 - tulpa::math::portable_digamma(r)
                                 - std::log1p(phi));
        return {grad, mu / (1.0 + phi)};
    }
    if (kind == FamilyKind::TRUNCATED_POISSON ||
        kind == FamilyKind::TRUNCATED_NEG_BINOMIAL_2) {
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        double a, da, d2a;
        truncation_shape(kind, mu, phi, &a, &da, &d2a);
        const TruncationTerm t = truncation_term(a, da, d2a);
        const double base_grad = (kind == FamilyKind::TRUNCATED_POISSON)
            ? grad_log_lik_poisson((int)y, eta)
            : grad_log_lik_negbin((int)y, eta, phi);
        return {base_grad - t.dlog_p, t.e_weight};
    }
    if (kind == FamilyKind::BETA_BINOMIAL) {
        // Beta-binomial (logit link, mu = P(success), phi = precision a + b).
        // Score is exact; the working weight is the moment-based Fisher weight
        // n mu(1-mu) / D with the overdispersion factor D = 1 + (n-1)/(phi+1)
        // (D -> 1 recovers the binomial weight as phi -> Inf). Always positive.
        double mu = linkinv(eta, "logit");
        mu = clamp_mu_unit(mu);
        double a = mu * phi, b = (1.0 - mu) * phi;
        double dmu = mu * (1.0 - mu);
        double n = (double)n_trials;
        double grad = phi * (tulpa::math::portable_digamma(y + a) - tulpa::math::portable_digamma(a)
                             - tulpa::math::portable_digamma(n - y + b) + tulpa::math::portable_digamma(b)) * dmu;
        double D = 1.0 + (n - 1.0) / (phi + 1.0);
        return {grad, n * mu * (1.0 - mu) / D};
    }
    if (kind == FamilyKind::STUDENT_T) {
        // Student-t location-scale (identity link): y ~ eta + phi * t_nu, with
        // nu = phi2 (NaN => the robust default kStudentTDf) and phi the scale.
        // Score is exact; the working weight is the constant Fisher information
        // (nu+1)/((nu+3) phi^2), which is positive and needs no Fisher fallback
        // (unlike the redescending observed information of the heavy tails).
        const double nu = student_t_df(phi2);
        double resid = y - eta;
        double grad = (nu + 1.0) * resid / (nu * phi * phi + resid * resid);
        return {grad, (nu + 1.0) / ((nu + 3.0) * phi * phi)};
    }
    if (kind == FamilyKind::TWEEDIE) {
        // Compound Poisson-gamma (log link), phi2 = power p in (1, 2). EDM
        // score through the log link, (y - mu)/(phi mu^(p-1)); expected
        // Fisher weight mu^(2-p)/phi. Always positive, no fallback needed.
        const TweedieParams tw = tweedie_params(phi2, eta);
        const double p = tw.p, mu = tw.mu;
        return {(y - mu) / (phi * std::pow(mu, p - 1.0)),
                std::pow(mu, 2.0 - p) / phi};
    }

    double mu = linkinv(eta, fl->link);
    double dmu = mu_eta(eta, fl->link);
    mu = clamp_mu_for_family(mu, fl->family);

    double g = grad_mu(y, mu, phi, fl->family, n_trials);
    double V = variance_fn(mu, phi, fl->family, n_trials);
    return {g * dmu, dmu * dmu / V};
}

// The resolved entry: no string work at all, the family having been classified
// and its link parsed once at spec construction.
inline GradHess grad_hess_for_family(
    double y, int n_trials, double eta, const FamilyResolved& fr,
    double phi, double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    return grad_hess_for_family_core(y, n_trials, eta, fr.kind, &fr.fl,
                                     phi, phi2);
}

inline GradHess grad_hess_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    const FamilyKind kind = family_kind(family);
    if (kind != FamilyKind::GENERIC) {
        return grad_hess_for_family_core(y, n_trials, eta, kind, nullptr,
                                         phi, phi2);
    }
    const FamilyLink fl = parse_family_link(family);
    return grad_hess_for_family_core(y, n_trials, eta, kind, &fl, phi, phi2);
}

// Per-observation log-density, with the family already classified and its
// eta-independent term already evaluated. `ll_const` is
// log_lik_const_for_kind(y, n_trials, kind) -- the binomial lchoose, the
// Poisson -lgamma(y+1), zero elsewhere -- and it is ADDED in the same position
// and the same order the full densities in laplace_likelihoods.h add it, so a
// caller holding a precomputed value reproduces them bit for bit
// (gcol33/tulpa#372). `fl` and `pos_eta_domain` are read only on the GENERIC
// branch; every other kind may pass a null `fl`.
inline double log_lik_for_family_core(
    double y, int n_trials, double eta, FamilyKind kind, const FamilyLink* fl,
    bool pos_eta_domain, double phi, double phi2, double ll_const
) {
    if (kind == FamilyKind::BINOMIAL) {
        return log_lik_binomial_kernel((int)y, n_trials, eta) + ll_const;
    }
    if (kind == FamilyKind::POISSON) {
        return log_lik_poisson_kernel((int)y, eta) + ll_const;
    }
    if (kind == FamilyKind::NEG_BINOMIAL_2) return log_lik_negbin((int)y, eta, phi);
    if (kind == FamilyKind::NEG_BINOMIAL_1) {
        // Log link, variance mu (1 + phi): in NB(r, p) form the shape moves
        // with the mean, r = mu / phi, and p = 1 / (1 + phi) is constant.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        const double r  = mu / phi;
        const double yi = (double)(int)y;
        return tulpa::math::portable_lgamma(yi + r) - tulpa::math::portable_lgamma(r) - tulpa::math::portable_lgamma(yi + 1.0)
             - (yi + r) * std::log1p(phi) + yi * std::log(phi);
    }
    if (kind == FamilyKind::TRUNCATED_POISSON ||
        kind == FamilyKind::TRUNCATED_NEG_BINOMIAL_2) {
        const double yi = (double)(int)y;
        if (yi < 1.0) return R_NegInf;   // support is y >= 1
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        const double base = (kind == FamilyKind::TRUNCATED_POISSON)
            ? log_lik_poisson_kernel((int)y, eta) + ll_const
            : log_lik_negbin((int)y, eta, phi);
        double a, da, d2a;
        truncation_shape(kind, mu, phi, &a, &da, &d2a);
        return base - truncation_term(a, da, d2a).log_p;
    }
    if (kind == FamilyKind::BETA_BINOMIAL) {
        double mu = linkinv(eta, "logit");
        mu = clamp_mu_unit(mu);
        double a = mu * phi, b = (1.0 - mu) * phi;
        double n = (double)n_trials, yi = (double)(int)y;
        return tulpa::math::portable_lchoose(n, yi)
             + tulpa::math::portable_lgamma(yi + a) + tulpa::math::portable_lgamma(n - yi + b) - tulpa::math::portable_lgamma(n + a + b)
             - tulpa::math::portable_lgamma(a) - tulpa::math::portable_lgamma(b) + tulpa::math::portable_lgamma(a + b);
    }
    if (kind == FamilyKind::STUDENT_T) {
        const double nu = student_t_df(phi2);
        double r = (y - eta) / phi;
        return tulpa::math::portable_lgamma((nu + 1.0) / 2.0) - tulpa::math::portable_lgamma(nu / 2.0)
             - 0.5 * std::log(nu * M_PI * phi * phi)
             - 0.5 * (nu + 1.0) * std::log1p(r * r / nu);
    }
    if (kind == FamilyKind::TWEEDIE) {
        return log_lik_tweedie(y, std::exp(eta), phi, tweedie_power(phi2));
    }

    // Domain barrier for the eta > 0 links. -Inf here is what stops the Newton
    // line search from stepping out of the domain; see link_eta_in_domain.
    if (pos_eta_domain && eta <= 0.0) return R_NegInf;
    double mu = linkinv(eta, fl->link);
    mu = clamp_mu_for_family(mu, fl->family);
    return log_lik_mu(y, mu, phi, fl->family, n_trials);
}

// The resolved entry: no string work, and the caller supplies the constant it
// precomputed once per fit.
inline double log_lik_for_family(
    double y, int n_trials, double eta, const FamilyResolved& fr,
    double phi, double phi2, double ll_const
) {
    return log_lik_for_family_core(y, n_trials, eta, fr.kind, &fr.fl,
                                   fr.positive_eta_domain, phi, phi2, ll_const);
}

inline double log_lik_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    const FamilyKind kind = family_kind(family);
    if (kind != FamilyKind::GENERIC) {
        return log_lik_for_family_core(y, n_trials, eta, kind, nullptr, false,
                                       phi, phi2,
                                       log_lik_const_for_kind(y, n_trials, kind));
    }
    const FamilyLink fl = parse_family_link(family);
    return log_lik_for_family_core(y, n_trials, eta, kind, &fl,
                                   link_has_positive_eta_domain(fl.link),
                                   phi, phi2, 0.0);
}

// Whether a family code reaches the generic mu-space route with BOTH ladders
// covering it -- grad_mu / dgrad_mu_dmu / d2grad_mu_dmu2 for the family and
// mu_eta / mu_eta2 / mu_eta3 for the link. Every generic-route derivative gates
// on this one predicate, so adding a family or a link to the ladders opens all
// of them at once instead of each carrying its own copy of the list.
inline bool generic_mu_route_exact(const std::string& family) {
    FamilyLink fl = parse_family_link(family);
    const bool fam_ok = mu_space_family_supported(fl.family);
    const bool link_ok =
        fl.link == "identity" || fl.link == "log" || fl.link == "inverse" ||
        fl.link == "logit" || fl.link == "probit" || fl.link == "cauchit" ||
        fl.link == "cloglog" || fl.link == "sqrt" || fl.link == "1mu2";
    return fam_ok && link_ok;
}

// Whether obs_grad_hess_for_family returns the exact observed curvature for
// this family, rather than delegating to a working weight that only
// approximates it.
//
// Everything the generic mu-space route covers is exact through the grad_mu
// ladder above, so this is the same predicate that gates the curvature
// derivatives -- one condition, not a second hand-kept list that can drift from
// the branches it describes.
//
// `t` is included even though its observed information redescends and goes
// NEGATIVE in the tails. That property is what keeps the Fisher form as the
// NEWTON weight, in grad_hess_for_family; nothing reads this function for a
// Newton step. What reads it is the mode-JACOBIAN solve, which is governed by
// the true curvature at the mode -- positive definite there because the mode is
// a local maximum, whatever individual observations contribute.
inline bool has_observed_curvature(const std::string& family) {
    return family == "poisson" || family == "binomial" ||
           family == "neg_binomial_2" || family == "neg_binomial_1" ||
           family == "truncated_poisson" ||
           family == "truncated_neg_binomial_2" ||
           family == "beta_binomial" || family == "tweedie" ||
           family == "t" || generic_mu_route_exact(family);
}

// Whether the Newton working weight grad_hess_for_family returns already IS the
// observed curvature, so W_obs - w is the zero FUNCTION and every quantity built
// on the difference is exactly zero rather than merely small. True for a
// curvature that carries no y (poisson, binomial, the truncated poisson), for
// neg_binomial_2's explicit branch (which returns the observed form), and for
// the canonical-link and constant-curvature members of the generic route. A
// non-canonical link breaks it even for those families: gaussian_log has a
// y-carrying observed curvature where gaussian_identity does not.
inline bool working_weight_is_observed(const std::string& family) {
    if (family == "poisson" || family == "binomial" ||
        family == "neg_binomial_2" || family == "truncated_poisson") {
        return true;
    }
    FamilyLink fl = parse_family_link(family);
    return (fl.family == "gaussian" && fl.link == "identity") ||
           (fl.family == "lognormal" && fl.link == "identity") ||
           (fl.family == "poisson" && fl.link == "log") ||
           (fl.family == "binomial" && fl.link == "logit") ||
           (fl.family == "gamma" && fl.link == "inverse") ||
           (fl.family == "inverse_gaussian" && fl.link == "1mu2");
}

// Whether the family is a discrete count distribution, so log_lik_for_family at
// y = 0 is a log PROBABILITY and the zero-inflation mixture's y = 0 branch means
// something. The zero-truncated pair belongs here: P(Y = 0) is exactly zero,
// which is what degenerates the mixture into the hurdle model.
//
// Separate from has_observed_curvature(). The two coincided while only count
// families carried an observed curvature, and reading P(Y = 0) off a continuous
// density -- a finite log-DENSITY for gaussian, an infinite one for gamma or
// beta -- is the failure that coincidence was hiding.
// The discrete bases, as a list rather than a disjunction, so a caller that has
// to REPORT the set can enumerate the same thing the predicate tests instead of
// restating it. Everything zero inflation can be compiled over is in here by
// construction, since compiled_zi_supported() requires has_discrete_mass().
inline const std::vector<std::string>& discrete_mass_families() {
    static const std::vector<std::string> bases = {
        "poisson", "binomial", "neg_binomial_2", "neg_binomial_1",
        "beta_binomial", "truncated_poisson", "truncated_neg_binomial_2"};
    return bases;
}

inline bool has_discrete_mass(const std::string& family) {
    const std::string base = parse_family_link(family).family;
    const std::vector<std::string>& bases = discrete_mass_families();
    return std::find(bases.begin(), bases.end(), base) != bases.end();
}

// Score plus OBSERVED curvature -d2 log f / d eta2 at the realized y.
//
// grad_hess_for_family returns the Newton WORKING weight, which is chosen for
// positive-definiteness and so is the expected/moment form wherever the two
// differ. The zero-inflation mixture cannot use that: its y = 0 branch
// differentiates through P(Y = 0), so it needs the curvature of the actual
// log-density, not a Fisher stand-in. This is the C++ counterpart of
// .family_obs_weight() in R/family_loglik.R and is validated against it.
//
// For a family whose curvature carries no y -- poisson, binomial, the
// truncated pair -- observed and expected coincide identically and the
// delegation below is exact, not an approximation. neg_binomial_2 also
// coincides: neg_hess_log_lik_negbin already returns (y + phi) phi mu /
// (mu + phi)^2, the observed form.
// Whether a family code reaches a compiled branch at all. A code that does not
// is the programming error unknown_family_stop names, and the point of asking
// in advance is WHERE the error is raised: Rcpp::stop throws, and an exception
// leaving an OpenMP structured block is std::terminate rather than an R error,
// so every entry that puts a family dispatch inside tulpa_parallel_sum resolves
// the family and asks this on the calling thread first.
inline bool family_has_compiled_impl(const std::string& code) {
    if (family_kind(code) != FamilyKind::GENERIC) return true;
    return mu_space_family_supported(parse_family_link(code).family);
}

inline GradHess obs_grad_hess_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi,
    double phi2 = std::numeric_limits<double>::quiet_NaN()
) {
    if (family == "neg_binomial_1") {
        // Differentiating the score s = r (psi(y+r) - psi(r) - log1p(phi))
        // with dr/deta = r gives -ds/deta = -s + r^2 (psi'(r) - psi'(y+r)).
        // Unlike neg_binomial_2 this does not agree with the working weight
        // even in expectation -- see the note under its grad_hess branch.
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        const double r  = mu / phi;
        const double s  = r * (tulpa::math::portable_digamma(y + r)
                               - tulpa::math::portable_digamma(r)
                               - std::log1p(phi));
        return {s, -s + r * r * (tulpa::math::portable_trigamma(r)
                                 - tulpa::math::portable_trigamma(y + r))};
    }
    if (family == "truncated_neg_binomial_2") {
        // The truncated density is the NB2 density minus log P(Y > 0), so the
        // curvature is the untruncated OBSERVED curvature plus d2 log P / d eta2.
        // (truncated_poisson's untruncated curvature carries no y, so its
        // observed form already equals the working weight and falls through.)
        const double mu = std::max(tulpa_linalg::safe_exp(eta), 1e-15);
        double a, da, d2a;
        truncation_shape(family, mu, phi, &a, &da, &d2a);
        const TruncationTerm t = truncation_term(a, da, d2a);
        return {grad_log_lik_negbin((int)y, eta, phi) - t.dlog_p,
                neg_hess_log_lik_negbin((int)y, eta, phi) + t.d2log_p};
    }
    if (family == "beta_binomial") {
        // Differentiating the exact score phi (psi(y+a) - psi(a) - psi(n-y+b)
        // + psi(b)) dmu, with a = mu phi, b = (1-mu) phi and a + b = phi free of
        // eta. The working weight is the moment form n mu(1-mu) / D instead, so
        // the two differ per observation AND in expectation.
        double mu = linkinv(eta, "logit");
        mu = clamp_mu_unit(mu);
        const double a = mu * phi, b = (1.0 - mu) * phi;
        const double dmu = mu * (1.0 - mu), n = (double)n_trials;
        const double grad = phi * (tulpa::math::portable_digamma(y + a) - tulpa::math::portable_digamma(a)
                                   - tulpa::math::portable_digamma(n - y + b) + tulpa::math::portable_digamma(b)) * dmu;
        const double tg = tulpa::math::portable_trigamma(a) + tulpa::math::portable_trigamma(b)
                          - tulpa::math::portable_trigamma(y + a) - tulpa::math::portable_trigamma(n - y + b);
        return {grad, phi * phi * tg * dmu * dmu - (1.0 - 2.0 * mu) * grad};
    }
    if (family == "t") {
        // Differentiating the exact score (nu+1) d / D with d = y - eta and
        // D = nu phi^2 + d^2 gives (nu+1)(nu phi^2 - d^2) / D^2 -- the
        // redescending observed information, negative once |d| exceeds the
        // scale. grad_hess_for_family keeps the constant Fisher form for Newton
        // for exactly that reason; this is the true curvature the mode Jacobian
        // is governed by.
        const double nu = student_t_df(phi2);
        const double d = y - eta;
        const double D = nu * phi * phi + d * d;
        return {(nu + 1.0) * d / D,
                (nu + 1.0) * (nu * phi * phi - d * d) / (D * D)};
    }
    if (family == "tweedie") {
        // EDM score (y - mu) mu^(1-p) / phi through the log link; differentiating
        // it gives mu^(1-p) [mu + (p-1)(y - mu)] / phi, which is
        // mu(2-p) + (p-1) y over phi mu^(p-1) -- positive for every y >= 0 with
        // p in (1, 2). At y = mu it collapses to the registered working weight.
        const TweedieParams tw = tweedie_params(phi2, eta);
        const double p = tw.p, mu = tw.mu;
        const double m1p = std::pow(mu, 1.0 - p);
        return {(y - mu) * m1p / phi,
                m1p * (mu + (p - 1.0) * (y - mu)) / phi};
    }
    if (generic_mu_route_exact(family) && !working_weight_is_observed(family)) {
        // -l'' = -(L'' u^2 + L' u1) at the realized y, against the working
        // weight u^2 / V the Fisher route returns. Where the link is canonical
        // or the curvature carries no y the two are the same function, and the
        // delegation below is both exact and cheaper -- taking this branch there
        // would leave the difference at rounding noise rather than at zero.
        FamilyLink fl = parse_family_link(family);
        double mu = linkinv(eta, fl.link);
        mu = clamp_mu_for_family(mu, fl.family);
        const double u  = mu_eta(eta, fl.link);
        const double u1 = mu_eta2(eta, fl.link);
        const double g  = grad_mu(y, mu, phi, fl.family, n_trials);
        const double gp = dgrad_mu_dmu(y, mu, phi, fl.family, n_trials);
        return {g * u, -(gp * u * u + g * u1)};
    }
    return grad_hess_for_family(y, n_trials, eta, family, phi, phi2);
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
    mu = clamp_mu_unit(mu);
    double a = mu * phi;
    double b = (1.0 - mu) * phi;
    return (a - 1.0) * slog_y + (b - 1.0) * slog_1my
           + (double)n * (tulpa::math::portable_lgamma(phi) - tulpa::math::portable_lgamma(a) - tulpa::math::portable_lgamma(b));
}

inline GradHess grad_hess_beta_grouped(double slog_y, double slog_1my, int n,
                                       double eta, double phi) {
    double mu  = linkinv(eta, "logit");
    double dmu = mu_eta(eta, "logit");
    mu = clamp_mu_unit(mu);
    double mu_star = tulpa::math::portable_digamma(mu * phi) - tulpa::math::portable_digamma((1.0 - mu) * phi);
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
// normal density (0 at +/-Inf, as is z phi(z)). Everything is formed in log
// space: the mass is differenced in whichever tail keeps its exponent, and the
// two ratios are exponentiated only after the subtraction, so an eta many sigma
// from the class still reports the gradient that points back toward it.
// A real number carried as a sign and a log magnitude, so a difference of two
// quantities that both underflow on the natural scale can still be formed and
// divided by a third that underflows with them.
struct SignedLog {
    double sign;     // -1, 0 or +1
    double log_mag;  // log |value|; -Inf when the value is zero
};

// exp(la) - exp(lb).
inline SignedLog signed_log_diff(double la, double lb) {
    if (la == lb) return SignedLog{0.0, R_NegInf};
    const bool a_bigger = la > lb;
    const double hi = a_bigger ? la : lb;
    const double lo = a_bigger ? lb : la;
    return SignedLog{a_bigger ? 1.0 : -1.0,
                     hi + tulpa::math::log1m_exp(hi - lo)};
}

inline SignedLog signed_log_add(const SignedLog& a, const SignedLog& b) {
    if (a.sign == 0.0) return b;
    if (b.sign == 0.0) return a;
    if (a.sign == b.sign) {
        const double hi = a.log_mag > b.log_mag ? a.log_mag : b.log_mag;
        const double lo = a.log_mag > b.log_mag ? b.log_mag : a.log_mag;
        return SignedLog{a.sign, hi + std::log1p(std::exp(lo - hi))};
    }
    const SignedLog d = signed_log_diff(a.log_mag, b.log_mag);
    return SignedLog{d.sign * a.sign, d.log_mag};
}

struct IntervalGaussian {
    double ll;        // log P
    double grad;      // d logP / d eta
    double neg_hess;  // -d2 logP / d eta2  (>= 0)
};

inline IntervalGaussian interval_gaussian_core(double lower, double upper,
                                               double eta, double sigma) {
    using tulpa::math::portable_dnorm_log;
    using tulpa::math::portable_pnorm;
    using tulpa::math::portable_pnorm_log;

    const double inv_s = 1.0 / sigma;
    const bool lo_open = !R_finite(lower);   // -Inf
    const bool hi_open = !R_finite(upper);   // +Inf
    const double zl = lo_open ? R_NegInf : (lower - eta) * inv_s;
    const double zu = hi_open ? R_PosInf : (upper - eta) * inv_s;

    // log P = log(Phi(zu) - Phi(zl)), differenced in whichever tail keeps its
    // exponent: below the median both pieces are read from the lower tail,
    // above it both from the upper, and a straddling interval carries mass of
    // order one in both so the natural-scale difference is well conditioned.
    // Differencing the probabilities themselves and flooring the result gives a
    // finite log, a zero gradient and a floored curvature once both tails
    // underflow -- a plateau of the floor, which the convergence test reads as
    // a mode and the line search accepts because it is finite.
    double logP;
    if (lo_open && hi_open) {
        logP = 0.0;
    } else if (lo_open) {
        logP = portable_pnorm_log(zu);
    } else if (hi_open) {
        logP = portable_pnorm_log(-zl);
    } else if (zu <= 0.0) {
        logP = signed_log_diff(portable_pnorm_log(zu),
                               portable_pnorm_log(zl)).log_mag;
    } else if (zl >= 0.0) {
        logP = signed_log_diff(portable_pnorm_log(-zl),
                               portable_pnorm_log(-zu)).log_mag;
    } else {
        const double P = portable_pnorm(zu) - portable_pnorm(zl);
        logP = P > 0.0 ? std::log(P) : R_NegInf;
    }

    // An interval carrying no mass -- upper <= lower -- is outside the support.
    // -Inf is what the line search refuses, so the solve backtracks off it
    // instead of reporting it as a mode.
    if (!(logP > R_NegInf)) return { R_NegInf, 0.0, kCensoredCurvFloor };

    const double lpl = lo_open ? R_NegInf : portable_dnorm_log(zl);
    const double lpu = hi_open ? R_NegInf : portable_dnorm_log(zu);

    // dP/deta = (phi(zl) - phi(zu)) / sigma, formed from the logs so the ratio
    // to P stays finite where both densities underflow individually.
    const SignedLog dP = signed_log_diff(lpl, lpu);
    const double g = (dP.sign == 0.0)
        ? 0.0
        : dP.sign * std::exp(dP.log_mag - logP) * inv_s;

    // d2P/deta2 = (zl phi(zl) - zu phi(zu)) / sigma^2, with the minus of the
    // second term carried in its sign. z phi(z) -> 0 as |z| -> Inf, the limit
    // an open bound takes.
    const SignedLog t_l = (lo_open || !R_finite(zl) || zl == 0.0)
        ? SignedLog{0.0, R_NegInf}
        : SignedLog{zl > 0.0 ? 1.0 : -1.0, std::log(std::fabs(zl)) + lpl};
    const SignedLog t_u = (hi_open || !R_finite(zu) || zu == 0.0)
        ? SignedLog{0.0, R_NegInf}
        : SignedLog{zu > 0.0 ? -1.0 : 1.0, std::log(std::fabs(zu)) + lpu};
    const SignedLog d2P = signed_log_add(t_l, t_u);
    const double Pdd_over_P = (d2P.sign == 0.0)
        ? 0.0
        : d2P.sign * std::exp(d2P.log_mag - logP) * inv_s * inv_s;

    double nh = g * g - Pdd_over_P;
    if (nh < kCensoredCurvFloor) nh = kCensoredCurvFloor;

    return { logP, g, nh };
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
        logPhi_a = tulpa::math::portable_pnorm_log(a);            // log Phi(a)
        lambda   = std::exp(tulpa::math::portable_dnorm_log(a)
                            - logPhi_a);                          // phi(a)/Phi(a), stable
    }

    const double ll = -0.5 * std::log(2.0 * M_PI * sigma * sigma)
                      - 0.5 * z * z - logPhi_a;
    const double grad = (z + lambda) * inv_s;
    // lambda * (a + lambda): guard the 0 * Inf at a = +Inf (lambda = 0 there).
    const double curv_term = (lambda == 0.0) ? 0.0 : lambda * (a + lambda);
    double nh = (1.0 - curv_term) * inv_s * inv_s;
    if (nh < kCensoredCurvFloor) nh = kCensoredCurvFloor;

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
    // Both stops below run on the calling thread BEFORE the parallel region —
    // the per-observation stop inside an omp reduction body would escape the
    // structured block, which is std::terminate. This entry carries no phi2, so
    // tweedie (which requires the variance power) can never evaluate here; an
    // unregistered family would reach unknown_family_stop from log_lik_mu.
    if (family == "tweedie") {
        Rcpp::stop("family 'tweedie' needs phi2 (the variance power p); "
                   "route it through the LikelihoodSpec path, which carries "
                   "phi2.");
    }
    if (!family_has_compiled_impl(family)) {
        unknown_family_stop("compute_total_log_lik", family);
    }
    // Resolving once also takes the family_kind ladder and the link parse out
    // of the per-observation body.
    const FamilyResolved fr = resolve_family(family);
    const double phi2 = std::numeric_limits<double>::quiet_NaN();
    return tulpa_parallel_sum(n_threads, N, [&](int i) {
        return log_lik_for_family(
            y[i], n_trials[i], eta[i], fr, phi, phi2,
            log_lik_const_for_kind(y[i], n_trials[i], fr.kind));
    });
}

// The same sum with the family resolved and its eta-independent per-observation
// terms already evaluated (gcol33/tulpa#372). This is the form the Newton line
// search reaches, so nothing in the loop body touches a string or an lgamma
// that does not move with eta; FamilyLogLik below owns the precompute.
inline double compute_total_log_lik(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericVector& eta, int N,
    const FamilyResolved& fr, const double* ll_const, double phi, int n_threads
) {
    // Raised on the calling thread, ahead of the parallel region: see the
    // entry above.
    if (fr.kind == FamilyKind::TWEEDIE) {
        Rcpp::stop("family 'tweedie' needs phi2 (the variance power p); "
                   "route it through the LikelihoodSpec path, which carries "
                   "phi2.");
    }
    if (!family_has_compiled_impl(fr.code)) {
        unknown_family_stop("compute_total_log_lik", fr.code);
    }
    const double phi2 = std::numeric_limits<double>::quiet_NaN();
    // This is the sum the Newton line search reaches, so it is evaluated once
    // per objective evaluation: at one thread tulpa_parallel_sum reduces
    // serially rather than entering libgomp for a team of one. Index order,
    // and therefore the sum, is the same on both routes.
    return tulpa_parallel_sum(n_threads, N, [&](int i) {
        return log_lik_for_family(y[i], n_trials[i], eta[i], fr, phi, phi2,
                                  ll_const[i]);
    });
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
    // Resolved once by prepare(); until it runs the string ladder is taken, so
    // a construction site that forgets the call gets the same arithmetic and
    // the old speed rather than a wrong constant.
    FamilyResolved fam;
    std::vector<double> ll_const;
    bool prepared = false;

    void prepare() {
        if (!y || N <= 0) return;
        fam = resolve_family(family);
        ll_const.resize((size_t)N);
        for (int i = 0; i < N; i++) {
            const int nt = n_trials ? (*n_trials)[i] : 1;
            ll_const[(size_t)i] = log_lik_const_for_kind((*y)[i], nt, fam.kind);
        }
        prepared = true;
    }

    double operator()(const Rcpp::NumericVector& eta) const {
        if (!prepared) {
            return compute_total_log_lik(*y, *n_trials, eta, N, family, phi,
                                         n_threads);
        }
        return compute_total_log_lik(*y, *n_trials, eta, N, fam,
                                     ll_const.data(), phi, n_threads);
    }
};

} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_LINK_H
