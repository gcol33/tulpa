#ifndef TULPA_BILINEAR_FACTOR_H
#define TULPA_BILINEAR_FACTOR_H
#include <vector>
#include <cmath>
#include <tulpa/ad_scalar_math.h>
namespace tulpa { namespace bilinear {
enum class LatentConstraint { SUM_TO_ZERO = 0, FIRST_ZERO = 1 };

inline int n_loadings(int P, int K) {
  int n = 0;
  for (int k = 0; k < K; k++) n += P - k;
  return n;
}


template <typename T>
inline void loadings(const T* theta, int P, int K, std::vector<T>& lam) {
  lam.assign(static_cast<size_t>(P) * K, T(0.0));
  int j = 0;
  for (int k = 0; k < K; k++) {
    for (int p = k; p < P; p++, j++) {
      lam[p * K + k] = (p == k) ? tulpa::math::safe_exp(theta[j]) : theta[j];
    }
  }
}




template <typename T>
inline T loadings_log_prior(const T* theta, int P, int K, double rate,
                            double sd_off) {
  T lp = T(0.0);
  int j = 0;
  for (int k = 0; k < K; k++) {
    for (int p = k; p < P; p++, j++) {
      if (p == k) {
        lp = lp + T(std::log(rate)) + theta[j]
                - T(rate) * tulpa::math::safe_exp(theta[j]);
      } else {
        lp = lp - T(0.5 / (sd_off * sd_off)) * theta[j] * theta[j];
      }
    }
  }
  return lp;
}


inline void loadings_log_prior_grad(const double* theta, int P, int K,
                                    double rate, double sd_off, double* g) {
  int j = 0;
  for (int k = 0; k < K; k++) {
    for (int p = k; p < P; p++, j++) {
      g[j] += (p == k) ? 1.0 - rate * std::exp(theta[j])
                       : -theta[j] / (sd_off * sd_off);
    }
  }
}


template <typename T>
inline void constrain_scores(const T* raw, int N, int K,
                             LatentConstraint constraint, std::vector<T>& f) {
  f.assign(raw, raw + static_cast<size_t>(N) * K);
  for (int k = 0; k < K; k++) {
    T shift = T(0.0);
    if (constraint == LatentConstraint::SUM_TO_ZERO) {
      for (int i = 0; i < N; i++) shift = shift + f[i * K + k];
      shift = shift / T(static_cast<double>(N));
    } else {
      shift = raw[k];
    }
    for (int i = 0; i < N; i++) f[i * K + k] = f[i * K + k] - shift;
  }
}


template <typename T>
inline T scores_log_prior(const T* raw, int n) {
  T q = T(0.0);
  for (int j = 0; j < n; j++) q = q + raw[j] * raw[j];
  return T(-0.5) * q;
}

} }
#endif
