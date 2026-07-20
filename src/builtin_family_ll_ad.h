// builtin_family_ll_ad.h
// Autodiff-templated per-observation log-likelihood for the built-in GLM
// families, so the ModelData sampler kernels get an analytic reverse-/forward-
// mode gradient through the full latent vector (fixed effects + RE + spatial +
// temporal) instead of the numerical fallback. The double path's value still
// comes from builtin_family_ll_double (laplace_builtin_family_spec.h); this
// header supplies spec.ll_arena / spec.ll_fwd, which differ from that value only
// by an eta-independent constant (the combinatorial / normalising terms), so the
// gradient stays consistent with the Hamiltonian's potential.
//
// Coverage: the default-link families whose log-density is a clean, AD-friendly
// closed form built from the templated primitives in autodiff_utils.h --
// gaussian (identity), poisson (log), binomial (logit), neg_binomial_2 (log),
// gamma (log), inverse_gaussian (log), lognormal (identity), beta (logit) and
// beta_binomial (logit).
// Each branch is the full log-density, so its value matches
// builtin_family_ll_double exactly (not merely up to an eta-independent
// constant). builtin_family_has_ad() gates which families set the AD callbacks;
// any other family (or non-default link, e.g. binomial_probit -- the link suffix
// is part of the family string) keeps ll_arena / ll_fwd null and falls back to
// the numerical gradient.

#ifndef TULPA_BUILTIN_FAMILY_LL_AD_H
#define TULPA_BUILTIN_FAMILY_LL_AD_H

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_builtin_family_spec.h"  // BuiltinFamilyResponse
#include "autodiff_utils.h"               // tulpa::math templated densities
#include "builtin_family_zi.h"            // tulpa::zi mixture (shared with Laplace)
#include "tulpa/types.h"                  // ZIType
#include <Rcpp.h>
#include <cmath>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

// Whether the AD likelihood below covers this family (exact, default-link only).
inline bool builtin_family_has_ad(const std::string& family) {
    return family == "gaussian" || family == "poisson" ||
           family == "binomial" || family == "neg_binomial_2" ||
           family == "gamma" || family == "inverse_gaussian" ||
           family == "lognormal" || family == "beta" ||
           family == "beta_binomial" || family == "t" ||
           family == "neg_binomial_1" ||
           family == "truncated_poisson" ||
           family == "truncated_neg_binomial_2";
}

// Unweighted base log-density at an arbitrary response value, templated over
// the AD type. Split out from builtin_family_ll_ad so the zero-inflation
// mixture can evaluate the same density at both the realized y and at 0
// without a second copy of the family dispatch.
template<typename T>
inline T builtin_family_base_ll_ad(
    double yv, int nt, const BuiltinFamilyResponse* r, const T& eta0
) {
    using namespace tulpa::math;
    const double phi = r->phi;
    const std::string& fam = r->family;
    const T* eta = &eta0;

    if (fam == "poisson") {
        return log_lik_poisson((int)yv, safe_exp(eta[0]));
    }
    if (fam == "binomial") {
        return log_lik_binomial((int)yv, nt, inv_logit(eta[0]));
    }
    if (fam == "neg_binomial_2") {
        return log_lik_negbin((int)yv, safe_exp(eta[0]), T(phi));
    }
    if (fam == "neg_binomial_1") {
        // Log link, variance mu (1 + phi). Shape r = mu / phi moves with the
        // mean, so r carries the eta dependence and the lgammas differentiate
        // through it; the log1p(phi) and log(phi) terms are eta-independent.
        T r = safe_exp(eta[0]) * T(1.0 / phi);
        return (lgamma_fn(T(yv) + r) - lgamma_fn(r)
             - T(std::lgamma(yv + 1.0))
             - (T(yv) + r) * T(std::log1p(phi))
             + T(yv * std::log(phi)));
    }
    if (fam == "truncated_poisson") {
        // Untruncated density minus the retained mass log(1 - exp(-mu)).
        T mu = safe_exp(eta[0]);
        return log_lik_poisson((int)yv, mu) - log1m_exp_fn(mu);
    }
    if (fam == "truncated_neg_binomial_2") {
        // Untruncated density minus log(1 - exp(-a)), a = phi log1p(mu/phi).
        T mu = safe_exp(eta[0]);
        T a  = log1p_fn(mu * T(1.0 / phi)) * T(phi);
        return log_lik_negbin((int)yv, mu, T(phi)) - log1m_exp_fn(a);
    }
    if (fam == "beta_binomial") {
        // Logit link, mu = P(success), phi = precision a + b. Exact AD log-lik.
        return log_lik_beta_binomial((int)yv, nt, inv_logit(eta[0]), T(phi));
    }
    if (fam == "t") {
        // Student-t location-scale (identity link): y ~ eta + phi * t_nu,
        // nu = phi2 (NaN => the robust default kStudentTDf), phi the scale.
        const double nu = std::isnan(r->phi2) ? kStudentTDf : r->phi2;
        T r = (T(yv) - eta[0]) * T(1.0 / phi);
        return (T(std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
                         - 0.5 * std::log(nu * M_PI * phi * phi))
             - T(0.5 * (nu + 1.0)) * log1p_fn(r * r * T(1.0 / nu)));
    }
    if (fam == "gaussian") {
        // Identity link: y ~ N(eta, phi^2). phi is the residual SD (held fixed).
        T resid = T(yv) - eta[0];
        return (T(-0.5 * std::log(2.0 * M_PI * phi * phi))
             - resid * resid * T(1.0 / (2.0 * phi * phi)));
    }
    if (fam == "lognormal") {
        // Identity link on the log scale: log(y) ~ N(eta, phi^2). phi is the
        // log-scale SD. The -log(y) Jacobian is eta-independent (data constant).
        const double ly = std::log(std::max(yv, 1e-300));
        T resid = T(ly) - eta[0];
        return (T(-ly - 0.5 * std::log(2.0 * M_PI * phi * phi))
             - resid * resid * T(1.0 / (2.0 * phi * phi)));
    }
    if (fam == "gamma") {
        // Log link: y ~ Gamma(shape = phi, mean = mu = exp(eta)).
        // ll = phi log phi - lgamma(phi) + (phi-1) log y - phi*eta - phi*y*exp(-eta).
        const double c = phi * std::log(phi) - std::lgamma(phi)
                       + (phi - 1.0) * std::log(std::max(yv, 1e-300));
        return (T(c) - T(phi) * eta[0] - T(phi * yv) * safe_exp(-eta[0]));
    }
    if (fam == "inverse_gaussian") {
        // Log link: y ~ IG with mean mu = exp(eta) and variance phi*mu^3.
        // ll = -0.5 log(2 pi phi y^3) - (y-mu)^2 / (2 phi y mu^2).
        T mu = safe_exp(eta[0]);
        T resid = T(yv) - mu;
        const double c = -0.5 * std::log(2.0 * M_PI * phi * yv * yv * yv);
        return (T(c)
             - resid * resid * T(1.0 / (2.0 * phi * yv)) / (mu * mu));
    }
    if (fam == "beta") {
        // Logit link: y ~ Beta(a, b) with a = mu*phi, b = (1-mu)*phi,
        // mu = inv_logit(eta), phi the precision.
        T mu = inv_logit(eta[0]);
        T a = mu * T(phi);
        T b = (T(1.0) - mu) * T(phi);
        const double ly  = std::log(yv);
        const double l1y = std::log(1.0 - yv);
        return (T(std::lgamma(phi)) - lgamma_fn(a) - lgamma_fn(b)
             + (a - T(1.0)) * T(ly) + (b - T(1.0)) * T(l1y));
    }
    Rcpp::stop("builtin_family_ll_ad: no AD likelihood for family '%s'. "
               "builtin_family_has_ad() gates which families reach here, so "
               "the two have fallen out of step.", fam.c_str());
}

// LikelihoodFn<T>: per-obs log-likelihood for a built-in family, templated over
// the AD type so the generic log-posterior differentiates straight through it.
//
// When the model carries zero inflation the sampler paths supply the ZI linear
// predictor through the `logit_zi` argument (built from data.X_zi_flat and the
// beta_zi block), and the base density is wrapped in the structural-zero
// mixture. The gradient flows through both predictors because the mixture is
// built from the same templated primitives as the density.
template<typename T>
inline T builtin_family_ll_ad(
    int i, const T* eta, const T& logit_zi, const T& /*logit_oi*/,
    const std::vector<T>& /*params*/, const ModelData& data,
    const ParamLayout& /*layout*/, const void* model_data
) {
    const auto* r   = static_cast<const BuiltinFamilyResponse*>(model_data);
    const double yv = r->y[i];
    const int    nt = r->n_trials ? r->n_trials[i] : 1;
    // Per-obs likelihood weight, matching builtin_family_ll_double / the score /
    // Fisher Hessian so the potential and its gradient stay consistent under a
    // weighted likelihood. A no-op when weights are absent.
    const double w = r->weights ? r->weights[i] : 1.0;

    if (data.zi_type == ZIType::NONE) {
        return T(w) * builtin_family_base_ll_ad<T>(yv, nt, r, eta[0]);
    }
    // A zero-truncated base is the hurdle case, where the y = 0 branch is
    // log(pi) alone; mixture_ll takes that limit from the flag rather than
    // through a density at 0 the truncated family does not define.
    const bool trunc = is_zero_truncated(r->family);
    if (trunc && yv == 0.0) {
        return T(w) * zi::mixture_ll(yv, logit_zi, T(0.0), T(0.0), true);
    }
    return T(w) * zi::mixture_ll(
        yv, logit_zi,
        builtin_family_base_ll_ad<T>(yv, nt, r, eta[0]),
        trunc ? T(0.0) : builtin_family_base_ll_ad<T>(0.0, nt, r, eta[0]),
        trunc);
}

} // namespace tulpa

#endif // TULPA_BUILTIN_FAMILY_LL_AD_H
