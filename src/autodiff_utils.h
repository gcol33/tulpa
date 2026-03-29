// autodiff_utils.h
// Templated math functions that work with double, ad::Var (tape), fwd::Dual (forward),
// and arena::Var (arena reverse-mode)
// Enables single implementation of compute_log_post for both evaluation and gradient

#ifndef TULPA_AUTODIFF_UTILS_H
#define TULPA_AUTODIFF_UTILS_H

#include <cmath>
#include <vector>
#include <Rcpp.h>
#include "autodiff.h"
#include "autodiff_fwd.h"
#include "autodiff_arena.h"

namespace tulpa {
namespace math {

// Portable math functions are in portable_math.h (included by autodiff headers)

// ============================================================================
// Type traits to detect ad::Var (tape), fwd::Dual (forward), arena::Var (arena)
// ============================================================================

template<typename T>
struct is_ad_var : std::false_type {};

template<>
struct is_ad_var<ad::Var> : std::true_type {};

template<typename T>
struct is_fwd_dual : std::false_type {};

template<>
struct is_fwd_dual<fwd::Dual> : std::true_type {};

template<typename T>
struct is_arena_var : std::false_type {};

template<>
struct is_arena_var<arena::Var> : std::true_type {};

// Helper: is any autodiff type
template<typename T>
struct is_autodiff : std::integral_constant<bool,
    is_ad_var<T>::value || is_fwd_dual<T>::value || is_arena_var<T>::value> {};

// ============================================================================
// Basic math functions - dispatch to std:: or ad::
// ============================================================================

// exp - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
safe_exp(T x) {
    constexpr double EXP_MAX = 700.0;
    constexpr double EXP_MIN = -700.0;
    if (x > EXP_MAX) x = EXP_MAX;
    if (x < EXP_MIN) x = EXP_MIN;
    return std::exp(x);
}

// exp - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_exp(const T& x) {
    return ad::exp(x);
}

// exp - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_exp(const T& x) {
    constexpr double EXP_MAX = 700.0;
    constexpr double EXP_MIN = -700.0;
    double v = x.val;
    if (v > EXP_MAX) v = EXP_MAX;
    if (v < EXP_MIN) v = EXP_MIN;
    double e = std::exp(v);
    return fwd::Dual(e, e * x.grad);
}

// exp - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
safe_exp(const T& x) {
    return arena::exp(x);
}

// log - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
safe_log(T x) {
    if (x <= 0.0) return -1e10;
    return std::log(x);
}

// log - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_log(const T& x) {
    return ad::log(x);
}

// log - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_log(const T& x) {
    if (x.val <= 0.0) return fwd::Dual(-1e10, 0.0);
    return fwd::Dual(std::log(x.val), x.grad / x.val);
}

// log - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
safe_log(const T& x) {
    return arena::log(x);
}

// sqrt - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
safe_sqrt(T x) {
    if (x < 0.0) return 0.0;
    return std::sqrt(x);
}

// sqrt - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_sqrt(const T& x) {
    return ad::sqrt(x);
}

// sqrt - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_sqrt(const T& x) {
    if (x.val < 0.0) return fwd::Dual(0.0, 0.0);
    double s = std::sqrt(x.val);
    return fwd::Dual(s, 0.5 * x.grad / s);
}

// sqrt - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
safe_sqrt(const T& x) {
    return arena::sqrt(x);
}

// max - double version (returns the larger of a and b)
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
safe_max(T a, T b) {
    return (a > b) ? a : b;
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

// max - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_max(const T& a, const T& b) {
    if (a.val >= b.val) {
        return a;
    } else {
        return b;
    }
}

// max - arena::Var version (subgradient: return the active branch)
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
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

// inv_logit (logistic function) - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
inv_logit(T x) {
    if (x > 0) {
        double exp_neg_x = std::exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

// inv_logit - ad::Var (tape) version
template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
inv_logit(const T& x) {
    return ad::inv_logit(x);
}

// inv_logit - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
inv_logit(const T& x) {
    return fwd::inv_logit(x);
}

// inv_logit - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
inv_logit(const T& x) {
    return arena::inv_logit(x);
}

// safe_tanh — works for all autodiff types via inv_logit
// tanh(x) = 2*sigmoid(2x) - 1
template<typename T>
inline T safe_tanh(const T& x) {
    return T(2.0) * inv_logit(T(2.0) * x) - T(1.0);
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

template<typename T>
inline T log_lik_gamma(double y, const T& shape, const T& mu) {
    // Gamma(y | shape, rate) where rate = shape/mu (mean parameterization)
    // = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape)
    T rate = shape / mu;
    return shape * safe_log(rate) + (shape - 1.0) * std::log(y) - rate * y - lgamma_fn(shape);
}

template<typename T>
inline T log_lik_negbin(int y, const T& mu, const T& phi) {
    // Negative binomial with mean mu and overdispersion phi
    // Using the Gamma-Poisson mixture parameterization
    T log_lik = lgamma_fn(T(y) + phi) - lgamma_fn(T(y + 1.0)) - lgamma_fn(phi);
    log_lik = log_lik + phi * safe_log(phi / (mu + phi));
    log_lik = log_lik + y * safe_log(mu / (mu + phi));
    return log_lik;
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
        T max_ab = (get_value(a) > get_value(b)) ? a : b;
        return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
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
        T max_ab = (get_value(a) > get_value(b)) ? a : b;
        return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
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
        T max_ab = (get_value(a) > get_value(b)) ? a : b;
        return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
    } else if (y == n) {
        // P(Y=n) = (1-zi) * (oi + (1-oi) * Binom(n|n,p))
        T log_binom_n = n * safe_log(p);
        T a = log_1m_zi + log_oi;
        T b = log_1m_zi + log_1m_oi + log_binom_n;
        T max_ab = (get_value(a) > get_value(b)) ? a : b;
        return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
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
        T log_1m_zero = safe_log(T(1.0) - safe_exp(n * safe_log(T(1.0) - p)));
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
        T max_ab = (get_value(a) > get_value(b)) ? a : b;
        return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
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
        T max_ab = (get_value(a) > get_value(b)) ? a : b;
        return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
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
        T log_trunc = log_lik_poisson(y, mu) - safe_log(T(1.0) - safe_exp(-mu));
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
        T log_trunc = log_lik_negbin(y, mu, phi) - safe_log(T(1.0) - safe_exp(log_p0));
        return log_theta + log_trunc;
    }
}

// ============================================================================
// New family likelihoods: Gamma-Gamma, Lognormal, Beta-Binomial
// ============================================================================

// Gamma-Gamma: Both numerator and denominator are Gamma distributed
// Using mean parameterization: mu = shape/rate, so rate = shape/mu
template<typename T>
inline T log_lik_gamma_gamma(double y_num, double y_denom,
                             const T& mu_num, const T& mu_denom,
                             const T& shape_num, const T& shape_denom) {
    // Numerator: Gamma(y_num | shape_num, shape_num/mu_num)
    T rate_num = shape_num / mu_num;
    T ll_num = shape_num * safe_log(rate_num) + (shape_num - T(1.0)) * std::log(y_num)
               - rate_num * y_num - lgamma_fn(shape_num);

    // Denominator: Gamma(y_denom | shape_denom, shape_denom/mu_denom)
    T rate_denom = shape_denom / mu_denom;
    T ll_denom = shape_denom * safe_log(rate_denom) + (shape_denom - T(1.0)) * std::log(y_denom)
                 - rate_denom * y_denom - lgamma_fn(shape_denom);

    return ll_num + ll_denom;
}

// Lognormal: log(y) ~ Normal(mu, sigma^2)
// Log-likelihood: -log(y) - log(sigma) - 0.5*log(2*pi) - 0.5*((log(y)-mu)/sigma)^2
template<typename T>
inline T log_lik_lognormal(double y, const T& mu, const T& sigma) {
    double log_y = std::log(y);
    T z = (log_y - mu) / sigma;
    // Omitting constant -0.5*log(2*pi)
    return -std::log(y) - safe_log(sigma) - T(0.5) * z * z;
}

// Lognormal-Lognormal: Both numerator and denominator are Lognormal
template<typename T>
inline T log_lik_lognormal_lognormal(double y_num, double y_denom,
                                     const T& mu_num, const T& mu_denom,
                                     const T& sigma_num, const T& sigma_denom) {
    return log_lik_lognormal(y_num, mu_num, sigma_num) +
           log_lik_lognormal(y_denom, mu_denom, sigma_denom);
}

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

// Normal prior: -0.5 * tau * x^2 (ignoring constants)
template<typename T>
inline T log_prior_normal(const T& x, double tau) {
    return T(-0.5) * tau * x * x;
}

// Half-Cauchy prior on sigma (log scale): -log(1 + (sigma/scale)^2) + log(sigma)
template<typename T>
inline T log_prior_half_cauchy(const T& log_sigma, double scale) {
    T sigma = safe_exp(log_sigma);
    T ratio = sigma / scale;
    return -safe_log(T(1.0) + ratio * ratio) + log_sigma;
}

// Gamma prior on phi (log scale): (shape-1)*log(phi) - rate*phi + log(phi)
template<typename T>
inline T log_prior_gamma(const T& log_phi, double shape, double rate) {
    T phi = safe_exp(log_phi);
    return (shape - 1.0) * log_phi - rate * phi + log_phi;
}

}  // namespace math
}  // namespace tulpa

#endif  // TULPA_AUTODIFF_UTILS_H
