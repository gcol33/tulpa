#ifndef TULPA_PRECISION_MATH_H
#define TULPA_PRECISION_MATH_H

#include <cmath>
#include <limits>

namespace tulpa { namespace math {

constexpr double log_two_pi = 1.837877066409345483560659472811;

// B_(2n) / (2n (2n-1)), in increasing powers x^-(2n-1).
inline long double stirling_series(long double x, int derivative) {
    static const long double c[] = {
        1.L/12, -1.L/360, 1.L/1260, -1.L/1680, 1.L/1188,
        -691.L/360360, 1.L/156, -3617.L/122400, 43867.L/244188,
        -174611.L/125400, 854513.L/63756, -236364091.L/1506960,
        8553103.L/3900, -23749461029.L/657720};
    const long double z = 1 / x, z2 = z * z;
    long double acc = 0;
    for (int j = 13; j >= 0; --j) {
        long double coefficient = c[j];
        for (int k = 0; k < derivative; ++k) coefficient *= -(2*j + 1 + k);
        acc = acc * z2 + coefficient;
    }
    for (int k = 0; k <= derivative; ++k) acc *= z;
    return acc;
}

// Derivatives of the Stirling remainder. Recurrence shifts the polygamma
// into the same asymptotic series; extended precision protects the small-x
// subtraction of its elementary part.
inline double stirling_remainder_derivative(double value, int order = 0) {
    const long double x = value;
    if (!(x > 0)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(value)) return 0;
    if (x >= 10) return static_cast<double>(stirling_series(x, order));
    if (order == 0) {
        const long double half_log_two_pi = 0.91893853320467274178032973640561764L;
        return static_cast<double>(std::lgamma(x) - ((x - .5L)*std::log(x) - x + half_log_two_pi));
    }
    long double z = x, sum = 0, fact = 1;
    for (int k = 2; k < order; ++k) fact *= k;
    while (z < 10) {
        sum += std::pow(z, -order);
        z += 1;
    }
    long double result = stirling_series(z, order);
    if (order == 1) {
        result += std::log(z/x) + .5L/x - .5L/z - sum;
    } else {
        const long double sign = order % 2 == 0 ? 1 : -1;
        auto main = [&](long double t) {
            return sign * fact * (std::pow(t, 1-order) / (order-1)
                                  + .5L*std::pow(t, -order));
        };
        result += main(z) - main(x) + sign*fact*sum;
    }
    return static_cast<double>(result);
}

inline double stirling_remainder(double x) {
    return stirling_remainder_derivative(x);
}

// d - log(1+d), d >= -1. The first omitted series term at |d|=1/8
// is smaller than 2^-78 times d^2.
inline double log1pmx(double value) {
    const long double d = value;
    if (value == std::numeric_limits<double>::infinity()) return value;
    if (std::abs(d) <= .125L) {
        long double s = 1.L/26;
        for (int k = 25; k >= 2; --k) s = 1.L/k - d*s;
        return static_cast<double>(d*d*s);
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

template<class T> inline T gamma_log_density(double y, const T& mu, const T& phi,
                                            double log_y) {
    using std::log;
    return T(.5)*(log(phi) - T(log_two_pi)) - stirling_remainder(phi)
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
    if (order == 1) return .5/phi - stirling_remainder_derivative(phi, 1)
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
