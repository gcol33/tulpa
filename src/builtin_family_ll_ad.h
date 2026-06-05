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
// Coverage: the four default-link families whose log-density is a clean,
// AD-friendly closed form built from the templated primitives in
// autodiff_utils.h -- gaussian (identity), poisson (log), binomial (logit), and
// neg_binomial_2 (log). builtin_family_has_ad() gates which families set the AD
// callbacks; any other family (or non-default link, e.g. binomial_probit) keeps
// ll_arena / ll_fwd null and falls back to the numerical gradient.

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
           family == "binomial" || family == "neg_binomial_2";
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

    if (fam == "poisson") {
        return log_lik_poisson((int)yv, safe_exp(eta[0]));
    }
    if (fam == "binomial") {
        return log_lik_binomial((int)yv, nt, inv_logit(eta[0]));
    }
    if (fam == "neg_binomial_2") {
        return log_lik_negbin((int)yv, safe_exp(eta[0]), T(phi));
    }
    if (fam == "gaussian") {
        // Identity link: y ~ N(eta, phi^2). phi is the residual SD (held fixed).
        T resid = T(yv) - eta[0];
        return T(-0.5 * std::log(2.0 * M_PI * phi * phi))
             - resid * resid * T(1.0 / (2.0 * phi * phi));
    }
    Rcpp::stop("builtin_family_ll_ad: AD likelihood only covers gaussian / "
               "poisson / binomial(logit) / neg_binomial_2; got '%s'.", fam.c_str());
}

} // namespace tulpa

#endif // TULPA_BUILTIN_FAMILY_LL_AD_H
