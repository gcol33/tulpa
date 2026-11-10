#ifndef TULPA_PRECISION_MATH_H
#define TULPA_PRECISION_MATH_H

#include <cmath>
#include <limits>
#include "precision_coefficients.h"

namespace tulpa { namespace math {

constexpr double log_two_pi = 1.837877066409345483560659472811;

namespace detail {

struct PrecisionPolynomial {
    double hi, lo;
    double value() const { return hi+lo; }
};

// The constant dominates the other terms on each interpolation interval.
// Its low part and the last product residual survive the derivative scaling.
template<int N> inline PrecisionPolynomial precision_polynomial(const double (&c)[N], double x,
                                                              double constant_low) {
    double p = c[0];
    for (int j=1; j<N-1; ++j) p = p*x+c[j];
    const double product = p*x, sum = c[N-1]+product;
    return {sum, ((c[N-1]-sum)+product)+std::fma(p,x,-product)+constant_low};
}

// The division residual carries rounding through x^-k without forming an
// overflowing power or overflowing 1/x before the final result does.
inline double precision_divide_power(double p, double x, int n, double low = 0) {
    for (int j=0; j<n; ++j) {
        const double q = p/x;
        if (!std::isfinite(q)) return q;
        low = (std::fma(-q,x,p)+low)/x;
        p = q;
    }
    return p+low;
}

// Range reduction keeps libm's logarithm on [1,2); the split ln(2)
// retains the exponent contribution when the elementary terms are combined.
inline double stirling_small_value(double x, double log_gamma_over_x) {
    int e;
    const double m = std::frexp(x,&e)*2;
    --e;
    const double lm = std::log(m), le = -.5*e;
    const double a = le*0x1.62e42fefa39efp-1;
    const double ae = std::fma(le,0x1.62e42fefa39efp-1,-a)+le*2.3190468138462996e-17;
    const double h = a-0.91893853320467274178;
    const double he = (a-h)-0.91893853320467274178+ae+3.8782941580672414e-17;
    const double correction = x*(1+log_gamma_over_x-(lm+e*0x1.62e42fefa39efp-1))-.5*lm;
    return h+(correction+he);
}

// Binet's asymptotic expansion has a remainder bounded by its first omitted
// term. Each derivative selects its term count from that relative bound.
template<int Order, int N> inline double stirling_series_terms(double x) {
    const double z = 1/x, z2 = z*z;
    const double* c = stirling_series_coefficients[Order];
    double p = c[N-1];
    for (int j=N-2; j>=0; --j) p = p*z2+c[j];
    return Order == 0 ? p/x : precision_divide_power(p,x,Order+1);
}

template<int Order, int First=1, int Last=(Order==0 ? 10 : Order==1 ? 12 : Order==2 ? 13 : 14)>
inline double stirling_series_select(double x) {
    if constexpr (First == Last) return stirling_series_terms<Order,First>(x);
    else {
        constexpr int mid = (First+Last)/2;
        return x>=stirling_series_cutoffs[Order][mid-1]
            ? stirling_series_select<Order,First,mid>(x)
            : stirling_series_select<Order,mid+1,Last>(x);
    }
}

} // namespace detail

inline double stirling_series(double x, int order) {
    switch (order) {
    case 0: return detail::stirling_series_select<0>(x);
    case 1: return detail::stirling_series_select<1>(x);
    case 2: return detail::stirling_series_select<2>(x);
    case 3: return detail::stirling_series_select<3>(x);
    default: return detail::stirling_series_select<4>(x);
    }
}

inline double stirling_remainder_derivative(double x, int order = 0) {
    if (!(x>0) || order<0 || order>4) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(x)) return 0;
    if (x>=10) return stirling_series(x,order);
    if (x>=.015625) {
        int e;
        const double t = std::frexp(x,&e)*2-1.5;
        const auto p = detail::precision_polynomial(detail::stirling_coefficients[order][e+5],t,
                                                   detail::stirling_constant_low[order][e+5]);
        return order == 0 ? p.value() : detail::precision_divide_power(p.hi,x,order,p.lo);
    }
    if (order>=2) {
        const auto p = detail::precision_polynomial(detail::stirling_small_coefficients[order-2],x,
                                                   detail::stirling_small_constant_low[0][order-2]);
        return detail::precision_divide_power(p.hi,x,order,p.lo);
    }
    const double p = detail::precision_polynomial(detail::stirling_elementary_coefficients[order],x,
                         detail::stirling_elementary_constant_low[0][order]).value();
    return order == 0 ? detail::stirling_small_value(x,p)
                      : std::fma(x,p-std::log(x),-.5)/x;
}

inline double stirling_remainder(double x) {
    return stirling_remainder_derivative(x);
}

// d-log(1+d). R nmath's pgamma.c uses r=d/(2+d) and the series
// S(y)=sum_(j>=0) y^j/(2j+3), y=r^2. The division residual and the final
// product residual retain the bits lost at the negative end of this range.
inline double log1pmx(double x) {
    if (x>=-.75 && x<=2) {
        const double d = 2+x, de = (2-d)+x;
        const double r = x/d, re = (std::fma(-r,d,x)-r*de)/d;
        const double y = r*r;
        double p;
        if (y<=.005) {
            const int n = y<=.00003 ? 4 : 8;
            p = detail::log1pmx_coefficients[n-1];
            for (int j=n-2; j>=0; --j) p = p*y+detail::log1pmx_coefficients[j];
        } else {
            const int branch = y<=.125 ? 0 : 1;
            const double t = y-(branch == 0 ? .0625 : .25);
            const double* c = detail::log1pmx_tail_coefficients[branch];
            p = c[0];
            for (int j=1; j<16; ++j) p = p*t+c[j];
        }
        const double inner = x-2*y*p, out = r*inner;
        const double err = r*std::fma(-2*y,p,x-inner)+std::fma(r,inner,-out)+(x-2*y/(1-y))*re;
        return out+err;
    }
    return x==std::numeric_limits<double>::infinity() ? x : x-std::log1p(x);
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
