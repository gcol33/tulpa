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
           family == "beta_binomial" || family == "t";
}

// LikelihoodFn<T>: per-obs log-likelihood for a built-in family, templated over
// the AD type so the generic log-posterior differentiates straight through it.
template<typename T>
inline T builtin_family_ll_ad(
    int i, const T* eta, const T& /*logit_zi*/, const T& /*logit_oi*/,
    const std::vector<T>& /*params*/, const ModelData& /*data*/,
    const ParamLayout& /*layout*/, const void* model_data
) {
    using namespace tulpa::math;
    const auto* r   = static_cast<const BuiltinFamilyResponse*>(model_data);
    const double yv = r->y[i];
    const int    nt = r->n_trials ? r->n_trials[i] : 1;
    const double phi = r->phi;
    const std::string& fam = r->family;
    // Per-obs likelihood weight, matching builtin_family_ll_double / the score /
    // Fisher Hessian so the potential and its gradient stay consistent under a
    // weighted likelihood (gcol33/tulpa#108). A no-op when weights are absent.
    const double w = r->weights ? r->weights[i] : 1.0;

    if (fam == "poisson") {
        return T(w) * log_lik_poisson((int)yv, safe_exp(eta[0]));
    }
    if (fam == "binomial") {
        return T(w) * log_lik_binomial((int)yv, nt, inv_logit(eta[0]));
    }
    if (fam == "neg_binomial_2") {
        return T(w) * log_lik_negbin((int)yv, safe_exp(eta[0]), T(phi));
    }
    if (fam == "beta_binomial") {
        // Logit link, mu = P(success), phi = precision a + b. Exact AD log-lik.
        return T(w) * log_lik_beta_binomial((int)yv, nt, inv_logit(eta[0]), T(phi));
    }
    if (fam == "t") {
        // Student-t location-scale (identity link): y ~ eta + phi * t_nu,
        // nu = phi2 (NaN => the robust default kStudentTDf), phi the scale.
        const double nu = std::isnan(r->phi2) ? kStudentTDf : r->phi2;
        T r = (T(yv) - eta[0]) * T(1.0 / phi);
        return T(w) * (T(std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
                         - 0.5 * std::log(nu * M_PI * phi * phi))
             - T(0.5 * (nu + 1.0)) * log1p_fn(r * r * T(1.0 / nu)));
    }
    if (fam == "gaussian") {
        // Identity link: y ~ N(eta, phi^2). phi is the residual SD (held fixed).
        T resid = T(yv) - eta[0];
        return T(w) * (T(-0.5 * std::log(2.0 * M_PI * phi * phi))
             - resid * resid * T(1.0 / (2.0 * phi * phi)));
    }
    if (fam == "lognormal") {
        // Identity link on the log scale: log(y) ~ N(eta, phi^2). phi is the
        // log-scale SD. The -log(y) Jacobian is eta-independent (data constant).
        const double ly = std::log(std::max(yv, 1e-300));
        T resid = T(ly) - eta[0];
        return T(w) * (T(-ly - 0.5 * std::log(2.0 * M_PI * phi * phi))
             - resid * resid * T(1.0 / (2.0 * phi * phi)));
    }
    if (fam == "gamma") {
        // Log link: y ~ Gamma(shape = phi, mean = mu = exp(eta)).
        // ll = phi log phi - lgamma(phi) + (phi-1) log y - phi*eta - phi*y*exp(-eta).
        const double c = phi * std::log(phi) - std::lgamma(phi)
                       + (phi - 1.0) * std::log(std::max(yv, 1e-300));
        return T(w) * (T(c) - T(phi) * eta[0] - T(phi * yv) * safe_exp(-eta[0]));
    }
    if (fam == "inverse_gaussian") {
        // Log link: y ~ IG with mean mu = exp(eta) and variance phi*mu^3.
        // ll = -0.5 log(2 pi phi y^3) - (y-mu)^2 / (2 phi y mu^2).
        T mu = safe_exp(eta[0]);
        T resid = T(yv) - mu;
        const double c = -0.5 * std::log(2.0 * M_PI * phi * yv * yv * yv);
        return T(w) * (T(c)
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
        return T(w) * (T(std::lgamma(phi)) - lgamma_fn(a) - lgamma_fn(b)
             + (a - T(1.0)) * T(ly) + (b - T(1.0)) * T(l1y));
    }
    Rcpp::stop("builtin_family_ll_ad: AD likelihood covers gaussian / poisson / "
               "binomial(logit) / neg_binomial_2 / gamma / inverse_gaussian / "
               "lognormal / beta; got '%s'.", fam.c_str());
}

} // namespace tulpa

#endif // TULPA_BUILTIN_FAMILY_LL_AD_H
