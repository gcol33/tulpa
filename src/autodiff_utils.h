// autodiff_utils.h
// Templated math functions that work with double, ad::Var (tape), fwd::Dual (forward),
// and arena::Var (arena reverse-mode)
// Enables single implementation of compute_log_post for both evaluation and gradient

#ifndef TULPA_AUTODIFF_UTILS_H
#define TULPA_AUTODIFF_UTILS_H

#include <algorithm>
#include <cmath>
#include <vector>
#include <Rcpp.h>
#include "autodiff.h"
#include "tulpa/autodiff_fwd.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/ad_scalar_math.h"

namespace tulpa {
namespace math {

// The tape type is src-only, so its trait specialisation and its overloads
// of the functions in tulpa/ad_scalar_math.h live here.
template<>
struct is_ad_var<ad::Var> : std::true_type {};

// Portable math functions are in portable_math.h (included by autodiff headers)

// ============================================================================
// Basic math functions - dispatch to std:: or ad::
// ============================================================================

// exp - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_exp(const T& x) {
    return ad::exp(x);
}

// log - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_log(const T& x) {
    return ad::log(x);
}

// sqrt - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_sqrt(const T& x) {
    return ad::sqrt(x);
}

// max - ad::Var (tape) version
// Note: for tape autodiff, max is non-differentiable at a==b,
// but we use subgradient (gradient of the active branch)
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_max(const T& a, const T& b) {
    if (a.val() >= b.val()) {
        return a;
    } else {
        return b;
    }
}

// lgamma (log of gamma function) - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
lgamma_fn(T x) {
    return std::lgamma(x);  // thread-safe, no R stack checking
}

// lgamma - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
lgamma_fn(const T& x) {
    return ad::lgamma(x);
}

// lgamma - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
lgamma_fn(const T& x) {
    // d(lgamma(x)) = digamma(x)
    return fwd::Dual(std::lgamma(x.val), portable_digamma(x.val) * x.grad);
}

// lgamma - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
lgamma_fn(const T& x) {
    return arena::lgamma(x);
}

// digamma (derivative of lgamma) - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
digamma_fn(T x) {
    return portable_digamma(x);  // thread-safe, no R stack checking
}

// digamma - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
digamma_fn(const T& x) {
    // d(digamma(x)) = trigamma(x)
    return fwd::Dual(portable_digamma(x.val), portable_trigamma(x.val) * x.grad);
}

// For ad::Var, digamma is handled internally by lgamma's backward pass

// digamma - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
digamma_fn(const T& x) {
    return arena::digamma(x);
}

// inv_logit - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
inv_logit(const T& x) {
    return ad::inv_logit(x);
}

// Bounded parameterization for a hyperparameter confined to (lower, upper).
//
// The bound belongs in the map, not in a rejection: HMC integrates a smooth
// density and needs a finite gradient everywhere it can step. Sampling the
// quantity directly and returning -INFINITY outside the box leaves the
// sampler no gradient to recover from, and every such step is reported as a
// divergence. So the sampler carries an unconstrained u and the density is
// evaluated at
//
//   phi = lower + (upper - lower) * inv_logit(u)
//
// which cannot leave the interval. The map contributes
//
//   dphi/du = (phi - lower) * (upper - phi) / (upper - lower)
//
// to the change of variables; log_jacobian_bounded returns its log. Callers
// add it to whatever density they place on phi.
template<typename T>
inline T bounded_from_logit(const T& u, double lower, double upper) {
    return T(lower) + T(upper - lower) * inv_logit(u);
}

template<typename T>
inline T log_jacobian_bounded(const T& phi, double lower, double upper) {
    return safe_log(phi - T(lower))
         + safe_log(T(upper) - phi)
         - T(std::log(upper - lower));
}

// -log(upper - lower): the Uniform(lower, upper) density a caller places on
// phi when it adds log_jacobian_bounded alone.
inline double log_prior_uniform_log_normalizer(double lower, double upper) {
    return -std::log(upper - lower);
}

// log1p - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
log1p_fn(T x) {
    return std::log1p(x);
}

// log1p - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
log1p_fn(const T& x) {
    return ad::log1p(x);
}

// log1p - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
log1p_fn(const T& x) {
    return fwd::log1p(x);
}

// log1p - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
log1p_fn(const T& x) {
    return arena::log1p(x);
}

// expm1 - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
expm1_fn(T x) {
    return std::expm1(x);
}

// expm1 - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
expm1_fn(const T& x) {
    return ad::expm1(x);
}

// expm1 - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
expm1_fn(const T& x) {
    return fwd::expm1(x);
}

// expm1 - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
expm1_fn(const T& x) {
    return arena::expm1(x);
}

// log(1 - exp(-a)) for a > 0, the retained-mass term of the zero-truncated and
// hurdle count families. Both naive forms lose precision at one end: log1p(-exp(-a))
// cancels as a -> 0, log(-expm1(-a)) underflows as a -> Inf. Splitting at
// log 2 keeps the accurate form on each side (Machler 2012), and both branches
// differentiate, so this is exact under AD rather than a fallback.
template<typename T>
inline T log1m_exp_fn(const T& a) {
    if (a <= T(0.693147180559945)) return safe_log(T(0.0) - expm1_fn(T(0.0) - a));
    return log1p_fn(T(0.0) - safe_exp(T(0.0) - a));
}

// log(1 + exp(x)), finite for every finite x. Splitting at zero keeps the
// exponent negative in both branches, so log1p always sees a small argument and
// the leading x is carried outside the log rather than through it: the naive
// log(1 + exp(x)) overflows to +Inf at x > 709.78 where the value is close to
// x, and log(1 + exp(-|x|)) at the other end is exact where exp(x) alone would
// drop the -exp(2x)/2 term. Its derivative is the logistic sigmoid, so it
// differentiates as accurately as it evaluates.
template<typename T>
inline T softplus(const T& x) {
    if (x > T(0.0)) return x + log1p_fn(safe_exp(T(0.0) - x));
    return log1p_fn(safe_exp(x));
}

// The logistic sigmoid, held strictly positive: inv_logit(x) is
// exp(-softplus(-x)), so inv_logit_pos(x) gives mu and inv_logit_pos(-x) gives
// 1 - mu, and a beta shape pair (mu phi, (1 - mu) phi) built from two calls has
// neither member exactly zero. Forming 1 / (1 + exp(-x)) instead rounds mu to
// exactly 1 above x = 36.74, making 1 - mu exactly 0, which is lgamma(0) = +Inf
// in the density and digamma(0) = -Inf in its gradient. The exponent is bounded
// at -700, which is where the double form's safe_exp already bounds it and
// where the AD exponentials would otherwise underflow to zero.
template<typename T>
inline T inv_logit_pos(const T& x) {
    T s = softplus(T(0.0) - x);
    if (s > T(700.0)) s = T(700.0);
    return safe_exp(T(0.0) - s);
}

// ============================================================================
// Value extraction (for getting double from T)
// ============================================================================

// double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, double>::type
get_value(const T& x) {
    return x;
}

// ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, double>::type
get_value(const T& x) {
    return x.val();
}

// fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, double>::type
get_value(const T& x) {
    return x.val;
}

// arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, double>::type
get_value(const T& x) {
    return x.val();
}

// log(exp(a) + exp(b)), shifted by the larger argument so neither exponent
// overflows. Both arguments -inf -- two mixture components that both underflow
// -- makes every difference below inf - inf, i.e. NaN, so a non-finite maximum
// short-circuits to itself, which is what the double core in linalg_fast.h
// returns there. This is the one copy: the zero- and one-inflated densities
// below all reduce their two-component mixture through it.
template<typename T>
inline T log_sum_exp_fn(const T& a, const T& b) {
    const T max_ab = (get_value(a) > get_value(b)) ? a : b;
    if (!std::isfinite(get_value(max_ab))) return max_ab;
    return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
}


// ============================================================================
// Dot product that works with all types
// ============================================================================

template<typename T>
inline T dot_product(const std::vector<T>& x, const std::vector<T>& y) {
    T sum = T(0.0);
    for (size_t i = 0; i < x.size(); i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// Dot product with double design matrix and T coefficients
template<typename T>
inline T dot_product_mixed(const double* x, const std::vector<T>& y, int n) {
    T sum = T(0.0);
    for (int i = 0; i < n; i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// Simpler: dot product where x is double array, y is T vector starting at offset
template<typename T>
inline T dot_product_mixed(const double* x, const T* y, int n) {
    T sum = T(0.0);
    for (int i = 0; i < n; i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// ============================================================================
// Likelihood functions (templated)
// ============================================================================

template<typename T>
inline T log_lik_poisson(int y, const T& mu) {
    // y * log(mu) - mu - lgamma(y+1)
    // Note: lgamma(y+1) is constant w.r.t. parameters, but we include it for correctness
    return y * safe_log(mu) - mu - lgamma_fn(T(y + 1.0));
}

// Beta density under the logit link, given the precision phi and the response's
// two logs. The single source of truth for every beta arm in the package: the
// generic family kernel, the beta sampler and the SPDE sampler all read it, so
// the shapes are held strictly positive by inv_logit_pos in exactly one place.
// log_y and log_1my arrive precomputed because every caller floors and caches
// them once per fit rather than per gradient evaluation.
template<typename T>
inline T log_lik_beta_logit(const T& eta, const T& phi,
                            double log_y, double log_1my) {
    const T a = inv_logit_pos(eta) * phi;
    const T b = inv_logit_pos(T(0.0) - eta) * phi;
    return lgamma_fn(phi) - lgamma_fn(a) - lgamma_fn(b)
         + (a - T(1.0)) * T(log_y)
         + (b - T(1.0)) * T(log_1my);
}

template<typename T>
inline T log_lik_gamma(double y, const T& shape, const T& mu) {
    // Gamma(y | shape, rate) where rate = shape/mu (mean parameterization)
    // = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape)
    T rate = shape / mu;
    return shape * safe_log(rate) + (shape - 1.0) * std::log(y) - rate * y - lgamma_fn(shape);
}

// log Gamma(y + phi) - log Gamma(phi) for a count y >= 0 and size phi > 0.
// As a difference of two lgammas it loses about eps * phi log(phi) to
// cancellation: 1e-12 at phi = 1e3, past 1 at phi = 1e15. From phi = 1e3 it is
// formed from Stirling's series instead,
//   phi log1p(y / phi) - y + y log(phi + y) - log1p(y / phi) / 2
//     + S(phi + y) - S(phi),   S(x) = 1/(12x) - 1/(360x^3) + 1/(1260x^5),
// whose terms are each O(y log phi) and whose truncation error is below
// 1/(1680 phi^7). Both branches differentiate, so the score in phi is exact
// too (the digamma difference cancels the same way).
template<typename T>
inline T lgamma_ratio_count(int y, const T& phi) {
    if (y == 0) return T(0.0);
    if (phi < T(1e3)) return lgamma_fn(T(y) + phi) - lgamma_fn(phi);
    const T yy = T(y);
    const T l1 = log1p_fn(yy / phi);
    auto stirling = [](const T& x) {
        const T x2 = x * x;
        return T(1.0 / 12.0) / x - T(1.0 / 360.0) / (x * x2)
             + T(1.0 / 1260.0) / (x * x2 * x2);
    };
    return phi * l1 - yy + yy * (safe_log(phi) + l1) - T(0.5) * l1
         + stirling(phi + yy) - stirling(phi);
}

// Negative binomial (type 2) with log mean eta and log size log_phi:
// var = mu + mu^2 / phi. With s = log1p(mu / phi) = softplus(eta - log_phi),
//   log p = lgamma_ratio_count(y, phi) - lgamma(y + 1)
//           + y (eta - log_phi - s) - phi s,
// every term finite and free of cancellation as phi -> Inf, where it reaches
// the Poisson density (phi s -> mu). The naive phi log(phi / (mu + phi))
// forms log(phi) - log(mu + phi) at a resolution of eps log(phi) and
// multiplies the rounding by phi: at log phi = 40 it returns a density near
// +5.2 for a Poisson -1.65 (y = 4, eta = 1.3).
template<typename T>
inline T log_lik_negbin_log(int y, const T& eta, const T& log_phi) {
    const T phi = safe_exp(log_phi);
    const T s = softplus(eta - log_phi);
    return lgamma_ratio_count(y, phi) - lgamma_fn(T(y + 1.0))
         + T(y) * (eta - log_phi - s) - phi * s;
}

// The same density from the mean mu and the size phi.
template<typename T>
inline T log_lik_negbin(int y, const T& mu, const T& phi) {
    return log_lik_negbin_log(y, safe_log(mu), safe_log(phi));
}

template<typename T>
inline T log_lik_binomial(int y, int n, const T& p) {
    // y * log(p) + (n-y) * log(1-p) + lchoose(n, y)
    // lchoose is constant w.r.t. p
    T log_lik = y * safe_log(p) + (n - y) * safe_log(T(1.0) - p);
    log_lik = log_lik + portable_lchoose(n, y);  // constant
    return log_lik;
}

// ============================================================================
// ZI/OI/Hurdle binomial log-likelihoods (templated)
// ============================================================================

// Zero-inflated binomial: P(Y=0) = zi + (1-zi) * Binom(0|n,p)
template<typename T>
inline T log_lik_zi_binomial(int y, int n, const T& p, const T& logit_zi) {
    T log_zi = -safe_log(T(1.0) + safe_exp(-logit_zi));  // log(sigmoid(logit_zi))
    T log_1m_zi = -safe_log(T(1.0) + safe_exp(logit_zi)); // log(1 - sigmoid(logit_zi))

    if (y == 0) {
        // log(zi + (1-zi) * (1-p)^n)
        T log_binom_zero = n * safe_log(T(1.0) - p);
        // log-sum-exp: log(exp(log_zi) + exp(log_1m_zi + log_binom_zero))
        T a = log_zi;
        T b = log_1m_zi + log_binom_zero;
        return log_sum_exp_fn(a, b);
    } else {
        // log((1-zi) * Binom(y|n,p))
        return log_1m_zi + log_lik_binomial(y, n, p);
    }
}

// One-inflated binomial: P(Y=n) = oi + (1-oi) * Binom(n|n,p)
template<typename T>
inline T log_lik_oi_binomial(int y, int n, const T& p, const T& logit_oi) {
    T log_oi = -safe_log(T(1.0) + safe_exp(-logit_oi));  // log(sigmoid(logit_oi))
    T log_1m_oi = -safe_log(T(1.0) + safe_exp(logit_oi)); // log(1 - sigmoid(logit_oi))

    if (y == n) {
        // log(oi + (1-oi) * p^n)
        T log_binom_n = n * safe_log(p);
        T a = log_oi;
        T b = log_1m_oi + log_binom_n;
        return log_sum_exp_fn(a, b);
    } else {
        // log((1-oi) * Binom(y|n,p))
        return log_1m_oi + log_lik_binomial(y, n, p);
    }
}

// Zero-and-one inflated binomial (ZOIB)
template<typename T>
inline T log_lik_zoib(int y, int n, const T& p, const T& logit_zi, const T& logit_oi) {
    T log_zi = -safe_log(T(1.0) + safe_exp(-logit_zi));
    T log_1m_zi = -safe_log(T(1.0) + safe_exp(logit_zi));
    T log_oi = -safe_log(T(1.0) + safe_exp(-logit_oi));
    T log_1m_oi = -safe_log(T(1.0) + safe_exp(logit_oi));

    if (y == 0) {
        // P(Y=0) = zi + (1-zi) * (1-oi) * Binom(0|n,p)
        T log_binom_zero = n * safe_log(T(1.0) - p);
        T a = log_zi;
        T b = log_1m_zi + log_1m_oi + log_binom_zero;
        return log_sum_exp_fn(a, b);
    } else if (y == n) {
        // P(Y=n) = (1-zi) * (oi + (1-oi) * Binom(n|n,p))
        T log_binom_n = n * safe_log(p);
        T a = log_1m_zi + log_oi;
        T b = log_1m_zi + log_1m_oi + log_binom_n;
        return log_sum_exp_fn(a, b);
    } else {
        // P(Y=y) = (1-zi) * (1-oi) * Binom(y|n,p)
        return log_1m_zi + log_1m_oi + log_lik_binomial(y, n, p);
    }
}

// Hurdle binomial: separate zero vs non-zero
template<typename T>
inline T log_lik_hurdle_binomial(int y, int n, const T& p, const T& logit_theta) {
    T log_theta = -safe_log(T(1.0) + safe_exp(-logit_theta));  // P(Y > 0)
    T log_1m_theta = -safe_log(T(1.0) + safe_exp(logit_theta)); // P(Y = 0)

    if (y == 0) {
        return log_1m_theta;
    } else {
        // P(Y=y|Y>0) = Binom(y|n,p) / (1 - (1-p)^n)
        T log_binom = log_lik_binomial(y, n, p);
        T log_1m_zero = log1m_exp_fn(T(0.0) - n * safe_log(T(1.0) - p));
        return log_theta + log_binom - log_1m_zero;
    }
}

// ============================================================================
// ZI/Hurdle count log-likelihoods (templated for autodiff)
// ============================================================================

// Zero-inflated Poisson log-PMF (logit scale for zi)
template<typename T>
inline T log_lik_zi_poisson(int y, const T& mu, const T& logit_zi) {
    T log_zi = -safe_log(T(1.0) + safe_exp(-logit_zi));    // log(sigmoid(logit_zi))
    T log_1m_zi = -safe_log(T(1.0) + safe_exp(logit_zi));  // log(1 - sigmoid(logit_zi))

    if (y == 0) {
        // log(zi + (1-zi) * exp(-mu))
        T a = log_zi;
        T b = log_1m_zi + (-mu);
        return log_sum_exp_fn(a, b);
    } else {
        return log_1m_zi + log_lik_poisson(y, mu);
    }
}

// Zero-inflated NegBin log-PMF (logit scale for zi)
template<typename T>
inline T log_lik_zi_negbin(int y, const T& mu, const T& phi, const T& logit_zi) {
    T log_zi = -safe_log(T(1.0) + safe_exp(-logit_zi));
    T log_1m_zi = -safe_log(T(1.0) + safe_exp(logit_zi));

    if (y == 0) {
        // log(zi + (1-zi) * (phi/(phi+mu))^phi)
        T log_p0_count = phi * safe_log(phi / (mu + phi));
        T a = log_zi;
        T b = log_1m_zi + log_p0_count;
        return log_sum_exp_fn(a, b);
    } else {
        return log_1m_zi + log_lik_negbin(y, mu, phi);
    }
}

// Hurdle Poisson log-PMF (logit scale for hurdle theta)
template<typename T>
inline T log_lik_hurdle_poisson(int y, const T& mu, const T& logit_theta) {
    T log_theta = -safe_log(T(1.0) + safe_exp(-logit_theta));    // P(Y > 0)
    T log_1m_theta = -safe_log(T(1.0) + safe_exp(logit_theta));  // P(Y = 0)

    if (y == 0) {
        return log_1m_theta;
    } else {
        // P(Y=y|Y>0) * theta = theta * Poisson(y|mu) / (1 - exp(-mu))
        T log_trunc = log_lik_poisson(y, mu) - log1m_exp_fn(mu);
        return log_theta + log_trunc;
    }
}

// Hurdle NegBin log-PMF (logit scale for hurdle theta)
template<typename T>
inline T log_lik_hurdle_negbin(int y, const T& mu, const T& phi, const T& logit_theta) {
    T log_theta = -safe_log(T(1.0) + safe_exp(-logit_theta));
    T log_1m_theta = -safe_log(T(1.0) + safe_exp(logit_theta));

    if (y == 0) {
        return log_1m_theta;
    } else {
        // P(Y=y|Y>0) * theta = theta * NB(y|mu,phi) / (1 - (phi/(phi+mu))^phi)
        T log_p0 = phi * safe_log(phi / (phi + mu));
        T log_trunc = log_lik_negbin(y, mu, phi) - log1m_exp_fn(T(0.0) - log_p0);
        return log_theta + log_trunc;
    }
}

// ============================================================================
// Beta-Binomial
// ============================================================================

// Beta-Binomial: y ~ BetaBinomial(n, alpha, beta)
// Using mean-precision parameterization: mu = alpha/(alpha+beta), phi = alpha+beta
// So alpha = mu*phi, beta = (1-mu)*phi
template<typename T>
inline T log_lik_beta_binomial(int y, int n, const T& mu, const T& phi) {
    T alpha = mu * phi;
    T beta = (T(1.0) - mu) * phi;

    // log P(y|n,alpha,beta) = log(B(y+alpha, n-y+beta)) - log(B(alpha, beta)) + log(C(n,y))
    // = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+alpha+beta)
    //   - lgamma(alpha) - lgamma(beta) + lgamma(alpha+beta)
    //   + lgamma(n+1) - lgamma(y+1) - lgamma(n-y+1)

    T ll = lgamma_fn(T(y) + alpha) + lgamma_fn(T(n - y) + beta) - lgamma_fn(T(n) + alpha + beta);
    ll = ll - lgamma_fn(alpha) - lgamma_fn(beta) + lgamma_fn(alpha + beta);
    ll = ll + portable_lchoose(n, y);  // constant w.r.t. parameters

    return ll;
}

// ============================================================================
// Prior distributions (templated)
// ============================================================================

// Normal prior: -0.5 * tau * x^2, the N(0, 1 / tau) log-density up to the
// constant log_prior_normal_log_normalizer(tau).
template<typename T>
inline T log_prior_normal(const T& x, double tau) {
    return T(-0.5) * tau * x * x;
}

// 0.5 log(tau / (2 pi)): what log_prior_normal leaves out of the N(0, 1 / tau)
// log-density.
inline double log_prior_normal_log_normalizer(double tau) {
    return 0.5 * std::log(tau / (2.0 * 3.14159265358979323846));
}

// Capped half-Cauchy on sigma: log_prior_half_cauchy on the supported region,
// -INFINITY above sigma_max. Pass sigma_max <= 0 to disable the cap.
//
// The cap is a hard box on a sampled quantity, so a step above sigma_max has
// no gradient for the sampler to recover along and is reported as a
// divergence. Where a bound on sigma is wanted under HMC, prefer the smooth
// route this file already carries: sample an unconstrained u and evaluate at
// bounded_from_logit(u, 0, sigma_max) with log_jacobian_bounded().
template<typename T>
inline T log_prior_capped_half_cauchy(const T& log_sigma,
                                      double scale,
                                      double sigma_max) {
    if (sigma_max > 0.0 && get_value(safe_exp(log_sigma)) > sigma_max) {
        return T(-INFINITY);
    }
    return log_prior_half_cauchy(log_sigma, scale);
}

// Gamma prior on phi (log scale): (shape-1)*log(phi) - rate*phi + log(phi)
template<typename T>
inline T log_prior_gamma(const T& log_phi, double shape, double rate) {
    T phi = safe_exp(log_phi);
    return (shape - 1.0) * log_phi - rate * phi + log_phi;
}

// shape log(rate) - lgamma(shape): what log_prior_gamma leaves out of the
// Gamma(shape, rate) density rate^shape phi^(shape - 1) e^(-rate phi) /
// Gamma(shape), carried to log(phi).
inline double log_prior_gamma_log_normalizer(double shape, double rate) {
    return shape * std::log(rate) - std::lgamma(shape);
}

// -log B(a, b): what a log(u) + b log(1 - u) leaves out of a Beta(a, b)
// density on u carried to logit(u), whose Jacobian u (1 - u) raises each
// exponent by one.
inline double log_prior_beta_logit_log_normalizer(double a, double b) {
    return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
}

// Capped Normal log-prior — single source of truth lives in
// inst/include/tulpa/priors_capped.h so downstream packages
// (tulpaGlmm) and tulpa share one implementation. Include via
// `#include <tulpa/priors_capped.h>` and call
// `tulpa::priors::log_prior_capped_normal(...)` instead.

}  // namespace math
}  // namespace tulpa

#endif  // TULPA_AUTODIFF_UTILS_H
