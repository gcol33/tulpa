#ifndef TULPA_COUNT_MATH_H
#define TULPA_COUNT_MATH_H

#include "device_math.h"
#include "precision_math.h"

namespace tulpa {
namespace math {

struct CountScalarMath {
  TULPA_HD static double value(double x) { return x; }
  TULPA_HD static double log(double x) { return std::log(x); }
  TULPA_HD static double log1p(double x) { return std::log1p(x); }
  TULPA_HD static double log1pmx(double x) { return tulpa::math::log1pmx(x); }
  TULPA_HD static double lgamma(double x) { return std::lgamma(x); }
  TULPA_HD static double exp(double x) { return std::exp(clamp_exp_arg(x)); }
  TULPA_HD static double softplus(double x) {
    return (x > 0 ? x : 0)+std::log1p(std::exp(-std::abs(x)));
  }
};

template <typename T>
TULPA_HD inline T count_stirling_remainder(const T& x) {
  const T inv=T(1)/x, inv2=inv*inv;
  return inv*(T(1.0/12) + inv2*(T(-1.0/360) + inv2*T(1.0/1260)));
}

// log Gamma(phi + y) - log Gamma(phi), using the finite recurrence for small
// counts and the Stirling difference at large phi. The omitted Stirling term
// is bounded by 1 / (1680 phi^7) on the asymptotic branch.
template <typename T, class Math = CountScalarMath>
TULPA_HD inline T lgamma_ratio_count(int y, const T& phi) {
  if (y <= 32) {
    T sum(0.0);
    for (int j = 0; j < y; ++j) sum = sum + Math::log(phi + T(j));
    return sum;
  }
  if (Math::value(phi) < 1e3) return Math::lgamma(phi + T(y)) - Math::lgamma(phi);
  const T yy(y), l1 = Math::log1p(yy / phi);
  return phi * l1 - yy + yy * (Math::log(phi) + l1) - T(0.5) * l1
       + count_stirling_remainder(phi + yy) - count_stirling_remainder(phi);
}

// The rising factorial with its y log(phi) term removed analytically.
template <typename T, class Math = CountScalarMath>
TULPA_HD inline T lgamma_ratio_count_correction(int y, const T& phi) {
  if (y <= 32) {
    T sum(0.0);
    for (int j=1; j<y; ++j) sum = sum + Math::log1p(T(j)/phi);
    return sum;
  }
  if (Math::value(phi) < 1e3)
    return lgamma_ratio_count<T,Math>(y,phi) - T(y)*Math::log(phi);
  const T yy(y), t=yy/phi, l1=Math::log1p(t);
  return -phi*Math::log1pmx(t) + (yy-T(.5))*l1 + count_stirling_remainder(phi+yy)-count_stirling_remainder(phi);
}

template <typename T, class Math = CountScalarMath>
TULPA_HD inline T beta_binomial_log_pmf(int y, int n, const T& p, const T& q, const T& phi,
                               double log_choose) {
  return T(log_choose) + T(y)*Math::log(p) + T(n-y)*Math::log(q)
      + lgamma_ratio_count_correction<T,Math>(y,p*phi)
      + lgamma_ratio_count_correction<T,Math>(n-y,q*phi)
      - lgamma_ratio_count_correction<T,Math>(n,phi);
}

// psi^(k)(phi + y) - psi^(k)(phi). Each asymptotic power difference is
// phi^-q expm1(-q log1p(y / phi)), preserving its small relative increment.
TULPA_HD inline double polygamma_count_asymptotic(int k, int y, double phi, bool centered = false) {
  double factorial=1;
  for (int i=2; i<=k; ++i) factorial*=i;
  const double sign=k%2==0 ? 1 : -1;
  double sum=0;
  const double L = std::log1p(y / phi);
  auto power_difference = [&](int q) {
    return std::pow(phi, -q) * std::expm1(-q * L);
  };
  if (k == 0) sum += centered ? -log1pmx(y/phi) : L;
  else sum -= sign * (factorial / k) * power_difference(k);
  static const double coef[] = {-0.5, -1.0 / 12.0, 1.0 / 120.0, -1.0 / 252.0,
      1.0 / 240.0, -1.0 / 132.0, 691.0 / 32760.0, -1.0 / 12.0};
  static const int power[] = {1, 2, 4, 6, 8, 10, 12, 14};
  for (int t = 0; t < 8; ++t) {
    double rising = 1.0;
    for (int i = 0; i < k; ++i) rising *= power[t] + i;
    sum += coef[t] * sign * rising * power_difference(power[t] + k);
  }
  return sum;
}

TULPA_HD inline double polygamma_rising_difference(int k, int y, double phi) {
  double factorial = 1.0;
  for (int i = 2; i <= k; ++i) factorial *= i;
  const double sign = k % 2 == 0 ? 1.0 : -1.0;
  double sum = 0.0;
  if (y <= 32) {
    for (int j = 0; j < y; ++j) sum += sign * factorial / std::pow(phi + j, k + 1);
    return sum;
  }
  while (phi < 10.0 && y > 0) {
    sum += sign * factorial / std::pow(phi, k + 1);
    phi += 1.0;
    --y;
  }
  return sum + polygamma_count_asymptotic(k,y,phi);
}

// d/dphi of the rising correction; its y/phi term is cancelled before rounding.
TULPA_HD inline double rising_correction_derivative(int y, double phi) {
  if (y <= 32) {
    double out=0;
    for(int j=1; j<y; ++j) out -= (j/phi)/(phi+j);
    return out;
  }
  if(phi < 1e3) return polygamma_rising_difference(0,y,phi)-y/phi;
  return polygamma_count_asymptotic(0,y,phi,true);
}

TULPA_HD inline double beta_binomial_phi_derivative(int y, int n, double p, double q, double phi) {
  return p*rising_correction_derivative(y,p*phi)
       + q*rising_correction_derivative(n-y,q*phi)-rising_correction_derivative(n,phi);
}

template <typename T, class Math = CountScalarMath>
TULPA_HD inline T negbin_log_p0(const T& mu, const T& phi) {
  return -phi * Math::log1p(mu / phi);
}

template <typename T, class Math = CountScalarMath>
TULPA_HD inline T negbin_log_pmf_log(int y, const T& eta, const T& log_phi,
                                    const T& log_factorial) {
  const T phi = Math::exp(log_phi), s = Math::softplus(eta-log_phi);
  return lgamma_ratio_count<T,Math>(y,phi)-log_factorial
       + T(y)*(eta-log_phi-s)-phi*s;
}

// The count normalizer can arrive from a per-count lookup table.
template <typename T, class Math = CountScalarMath>
TULPA_HD inline T negbin_log_pmf(int y, const T& mu, const T& phi, const T& rising, double log_factorial) {
  const T s = Math::log1p(mu / phi);
  return rising - T(log_factorial) + T(y) * (Math::log(mu) - Math::log(phi) - s) - phi * s;
}

template <typename T, class Math = CountScalarMath>
TULPA_HD inline T negbin_log_pmf(int y, const T& mu, const T& phi) {
  return negbin_log_pmf<T, Math>(y, mu, phi, lgamma_ratio_count<T, Math>(y, phi), std::lgamma(y + 1.0));
}

TULPA_HD inline double negbin_size_score(int y, double mu, double phi, double digamma_difference) {
  return digamma_difference - std::log1p(mu / phi) + (mu - y) / (phi + mu);
}

TULPA_HD inline double negbin_size_score(int y, double mu, double phi) {
  return negbin_size_score(y, mu, phi, polygamma_rising_difference(0, y, phi));
}

}  // namespace math
}  // namespace tulpa

#endif
