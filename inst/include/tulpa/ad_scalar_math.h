// ad_scalar_math.h
// The scalar math the exported prior helpers are written in, for the three
// scalar types a downstream package evaluates a log-density with: double,
// fwd::Dual and arena::Var. tulpa's src/autodiff_utils.h includes this header
// and adds the tape (ad::Var) overloads, so the engine and a consumer
// package evaluate one implementation.

#ifndef TULPA_AD_SCALAR_MATH_H
#define TULPA_AD_SCALAR_MATH_H

#include <algorithm>
#include <cmath>
#include <type_traits>
#include "portable_math.h"
#include "autodiff_fwd.h"
#include "autodiff_arena.h"

namespace tulpa {
namespace math {

// ============================================================================
// Type traits to detect ad::Var (tape), fwd::Dual (forward), arena::Var (arena)
// ============================================================================

template<typename T>
struct is_ad_var : std::false_type {};

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

// exp - double version
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
safe_exp(T x) {
    return std::exp(tulpa::math::clamp_exp_arg(x));
}

// exp - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_exp(const T& x) {
    double e = std::exp(tulpa::math::clamp_exp_arg(x.val));
    return fwd::Dual(e, e * x.grad);
}

// exp - arena::Var version
template<typename T>
inline typename std::enable_if<is_arena_var<T>::value, T>::type
safe_exp(const T& x) {
    return arena::exp(x);
}

// log - double version.
//
// At or below zero the value is the finite sentinel -1e10, not -infinity. The
// log posterior is a sum: one -infinity term takes the whole objective and
// every gradient computed through it to -infinity or NaN, which leaves a
// sampler no direction to step back along. A large finite penalty keeps the
// objective ordered and the step recoverable. All four instantiations below
// use the same sentinel, so the double instantiation the runtime gradient
// check differences is the same function of its argument as the autodiff ones.
template<typename T>
inline typename std::enable_if<!is_autodiff<T>::value, T>::type
safe_log(T x) {
    if (x <= 0.0) return -1e10;
    return std::log(x);
}

// log - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_log(const T& x) {
    // One rule for all four instantiations (see the double overload above):
    // value log(x) above zero and the constant -1e10 at or below it, partial
    // 1 / max(x, 1e-15) above zero and 0 at or below it. A clamped VALUE is
    // locally constant, so its derivative is 0; carrying 1/1e-15 there instead
    // hands an HMC trajectory a 1e15 adjoint on a flat region, and the double
    // finite-difference reference the AD paths are checked against sees the
    // flat one.
    if (x.val <= 0.0) return fwd::Dual(-1e10, 0.0);
    return fwd::Dual(std::log(x.val), x.grad / std::max(x.val, 1e-15));
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

// sqrt - fwd::Dual version
template<typename T>
inline typename std::enable_if<is_fwd_dual<T>::value, T>::type
safe_sqrt(const T& x) {
    // Zero is included: 0.5 / sqrt(0) is +Inf, where the double overload and
    // both reverse-mode overloads give a finite 0.
    if (x.val <= 0.0) return fwd::Dual(0.0, 0.0);
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

// Half-Cauchy prior on sigma (log scale): -log(1 + (sigma/scale)^2) + log(sigma)
template<typename T>
inline T log_prior_half_cauchy(const T& log_sigma, double scale) {
    T sigma = safe_exp(log_sigma);
    T ratio = sigma / scale;
    return -safe_log(T(1.0) + ratio * ratio) + log_sigma;
}

}  // namespace math
}  // namespace tulpa

#endif  // TULPA_AD_SCALAR_MATH_H
