#ifndef TULPA_PRECISION_MATH_H
#define TULPA_PRECISION_MATH_H

#include <cmath>
#include <limits>
#include "precision_accumulator.h"

namespace tulpa { namespace math {

constexpr double log_two_pi = 1.837877066409345483560659472811;

// B_(2n) / (2n (2n-1)), in increasing powers x^-(2n-1).
inline detail::PrecisionAccumulator stirling_series(detail::PrecisionAccumulator x, int derivative) {
    using Wide = detail::PrecisionAccumulator;
    static const Wide c[] = {
        Wide(1)/12, Wide(-1)/360, Wide(1)/1260, Wide(-1)/1680, Wide(1)/1188,
        Wide(-691)/360360, Wide(1)/156, Wide(-3617)/122400, Wide(43867)/244188,
        Wide(-174611)/125400, Wide(854513)/63756, Wide(-236364091)/1506960,
        Wide(8553103)/3900, Wide(-23749461029.L)/657720};
    const Wide z = Wide(1) / x, z2 = z * z;
    Wide acc = 0;
    for (int j = 13; j >= 0; --j) {
        Wide coefficient = c[j];
        for (int k = 0; k < derivative; ++k) coefficient *= -(2*j + 1 + k);
        acc = acc * z2 + coefficient;
    }
    for (int k = 0; k <= derivative; ++k) acc *= z;
    return acc;
}

// Derivatives of h(x) = (x + 1/2) log1p(1/x) - 1. With r = 1/(2x+1),
// h(x) = sum_{n>=1} r^(2n)/(2n+1); its differentiated terms have one sign.
inline detail::PrecisionAccumulator stirling_step(detail::PrecisionAccumulator x, int order) {
    using Wide = detail::PrecisionAccumulator;
    if (order >= 2) {
        const Wide a = Wide(1)/x, b = Wide(1)/(x+Wide(1));
        Wide sum = 0, coefficient = 1;
        for (int j = 0; j <= order-2; ++j) {
            if (j == 0) {
                for (int k = 2; k < order; ++k) coefficient *= k;
            } else coefficient *= Wide((order-1-j)*(j+1)) / Wide(j*(order-j));
            Wide term = Wide(.5)*coefficient*detail::accumulator_power(b, order-j);
            for (int k=0; k<j+2; ++k) term *= a;
            sum += term;
        }
        return Wide(order % 2 ? -1 : 1)*sum;
    }
    if (x.hi < .25L) {
        const Wide l = detail::accumulator_log(x+Wide(1))-detail::accumulator_log(x);
        return order == 0 ? (x+Wide(.5L))*l-Wide(1)
                          : l - (Wide(.5L)/x + Wide(.5L)/(x+Wide(1)));
    }
    const Wide r = Wide(1)/(Wide(2)*x+Wide(1)), r2 = r*r;
    const Wide sum = detail::accumulator_log_tail(r2);
    return order == 0 ? r2*sum : Wide(2)*r*r2*(sum-Wide(1)/(Wide(1)-r2));
}

// s^(k)(x) = s^(k)(x+n) + sum_{j<n} h^(k)(x+j). Every term has
// sign (-1)^k, so shifting into the asymptotic range preserves relative error.
inline double stirling_remainder_derivative(double value, int order = 0) {
    using Wide = detail::PrecisionAccumulator;
    const Wide x = value;
    if (!(value > 0)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(value)) return 0;
    if (value >= 10) return static_cast<double>(stirling_series(x, order));
    const int n = static_cast<int>(std::ceil(10-value));
    Wide result = stirling_series(x+Wide(n), order);
    for (int j = n-1; j >= 0; --j) result += stirling_step(x+Wide(j), order);
    return static_cast<double>(result);
}

inline double stirling_remainder(double x) {
    return stirling_remainder_derivative(x);
}

// d - log(1+d), d >= -1. The atanh series uses r=d/(2+d), as in
// R's nmath pgamma.c. In this interval |r| <= 3/5.
inline double log1pmx(double value) {
    const long double d = value;
    if (value == std::numeric_limits<double>::infinity()) return value;
    if (d >= -.75L && d <= 2.L) {
        using Wide = detail::PrecisionAccumulator;
        const Wide r = Wide(d)/(Wide(2)+Wide(d)), r2 = r*r;
        const Wide s = detail::accumulator_log_tail(r2);
        return static_cast<double>(r*(Wide(d) - Wide(2)*r2*s));
    }
    return static_cast<double>(d - std::log1p(d));
}

inline double precision_value(double x) { return x; }
template<class T> inline auto precision_value(const T& x) -> decltype(x.val()) { return x.val(); }
template<class T> inline auto precision_value(const T& x) -> decltype(+x.val) { return x.val; }

// Scalar adapters record one unary operation with its value and chain rule.
template<class T> struct PrecisionUnary;
template<class T> inline T stirling_remainder(const T& x) {
    const double v = precision_value(x);
    return PrecisionUnary<T>::apply(x, stirling_remainder(v), stirling_remainder_derivative(v, 1));
}
template<class T> inline T log1pmx(const T& x) {
    const double v = precision_value(x);
    return PrecisionUnary<T>::apply(x, log1pmx(v), v/(1+v));
}

// The direct log ratio retains a positive response whose relative residual
// rounds to -1. Close to the mean, subtraction retains the small residual.
template<class T> inline T relative_deviance_delta(const T& delta, const T& mu, double log_y) {
    using std::log;
    const T d = delta/mu;
    if (std::abs(precision_value(d)) <= .125) return log1pmx(d);
    return d - (T(log_y) - log(mu));
}

template<class T> inline T relative_deviance(double y, const T& mu, double log_y) {
    return relative_deviance_delta(T(y) - mu, mu, log_y);
}

template<class T> inline T gamma_log_normalizer(const T& phi) {
    using std::log;
    return T(.5)*(log(phi) - T(log_two_pi)) - stirling_remainder(phi);
}

inline double gamma_phi_constant(double phi) {
    return .5/phi - stirling_remainder_derivative(phi, 1);
}

struct GammaPrecision {
    double phi=1, normalizer=0, score_constant=0;
    GammaPrecision() = default;
    explicit GammaPrecision(double precision, bool density = true)
        : phi(precision), normalizer(density ? gamma_log_normalizer(precision) : 0),
          score_constant(gamma_phi_constant(precision)) {}
    double log_density(double y, double mu, double log_y) const {
        return normalizer - phi*relative_deviance(y, mu, log_y) - log_y;
    }
    double derivative(double y, double mu, double log_y) const {
        return score_constant - relative_deviance(y, mu, log_y);
    }
};

template<class T> inline T gamma_log_density(double y, const T& mu, const T& phi,
                                            double log_y) {
    using std::log;
    return gamma_log_normalizer(phi)
         - phi*relative_deviance(y, mu, log_y) - T(log_y);
}

template<class T> inline T beta_log_normalizer(const T& mu, const T& q, const T& phi) {
    using std::log;
    return T(.5)*(log(mu) + log(q) + log(phi) - T(log_two_pi))
         + stirling_remainder(phi) - stirling_remainder(mu*phi) - stirling_remainder(q*phi);
}

template<class T> inline T beta_log_response(double y, const T& mu, const T& q,
                                            const T& phi, double log_y, double log_1my) {
    return -mu*phi*relative_deviance(y, mu, log_y)
           - q*phi*relative_deviance_delta(mu-T(y), q, log_1my) - T(log_y + log_1my);
}

template<class T> inline T beta_log_density(double y, const T& mu, const T& q,
                                           const T& phi, double log_y, double log_1my) {
    return beta_log_normalizer(mu, q, phi) + beta_log_response(y, mu, q, phi, log_y, log_1my);
}

inline double beta_response_log_ratio(double y, double mu) {
    const double q = 1-mu, delta = y-mu;
    const double ly = std::abs(delta/mu) <= .125 ? std::log1p(delta/mu)
                                               : std::log(y)-std::log(mu);
    const double lq = std::abs(delta/q) <= .125 ? std::log1p(-delta/q)
                                              : std::log1p(-y)-std::log(q);
    return ly-lq;
}

inline double beta_mu_score(double y, double mu, double phi) {
    const double q = 1-mu, a = mu*phi, b = q*phi;
    return phi*(beta_response_log_ratio(y, mu) - stirling_remainder_derivative(a, 1)
                     + stirling_remainder_derivative(b, 1)) + .5/mu - .5/q;
}

inline double beta_score_phi_derivative(double y, double mu, double phi) {
    const double q = 1-mu, a = mu*phi, b = q*phi;
    return mu*q*(beta_response_log_ratio(y, mu) - stirling_remainder_derivative(a, 1)
                       + stirling_remainder_derivative(b, 1)
                       - a*stirling_remainder_derivative(a, 2)
                       + b*stirling_remainder_derivative(b, 2));
}

inline double gamma_phi_derivative(double y, double mu, double phi, int order) {
    if (order == 1) return gamma_phi_constant(phi)
        - relative_deviance(y, mu, std::log(y));
    return (-.5/phi)/phi - stirling_remainder_derivative(phi, 2);
}

inline double beta_phi_derivative(double y, double mu, double phi, int order) {
    const double q = 1-mu, a = mu*phi, b = q*phi;
    if (order == 1) return .5/phi + stirling_remainder_derivative(phi, 1)
        - mu*stirling_remainder_derivative(a, 1) - q*stirling_remainder_derivative(b, 1)
        - mu*relative_deviance(y, mu, std::log(y))
        - q*relative_deviance_delta(mu-y, q, std::log1p(-y));
    return (-.5/phi)/phi + stirling_remainder_derivative(phi, 2)
        - mu*mu*stirling_remainder_derivative(a, 2)
        - q*q*stirling_remainder_derivative(b, 2);
}

} }
#endif
