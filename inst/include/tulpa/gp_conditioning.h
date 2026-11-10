#ifndef TULPA_GP_CONDITIONING_H
#define TULPA_GP_CONDITIONING_H
#include <cmath>
#include <vector>
#include <tulpa/ad_scalar_math.h>
namespace tulpa_gp_conditioning {
using namespace tulpa::math;
constexpr double NNGP_RIDGE = 1e-8;

template <typename T>
inline bool nngp_chol(const T* A, int n, T* L, double ridge = NNGP_RIDGE) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < j; i++) L[i * n + j] = T(0.0);

    T sum = A[j * n + j] + T(ridge);
    for (int k = 0; k < j; k++) {
      sum = sum - L[j * n + k] * L[j * n + k];
    }
    if (get_value(sum) <= 0.0) return false;
    L[j * n + j] = safe_sqrt(sum);

    for (int i = j + 1; i < n; i++) {
      T sum_ij = A[i * n + j];
      for (int k = 0; k < j; k++) {
        sum_ij = sum_ij - L[i * n + k] * L[j * n + k];
      }
      L[i * n + j] = sum_ij / L[j * n + j];
    }
  }
  return true;
}

template <typename T>
inline void nngp_forward_solve(const T* L, int n, const T* b, T* y) {
  for (int i = 0; i < n; i++) {
    T sum = b[i];
    for (int j = 0; j < i; j++) {
      sum = sum - L[i * n + j] * y[j];
    }
    y[i] = sum / L[i * n + i];
  }
}

template <typename T>
inline void nngp_back_solve(const T* L, int n, const T* y, T* x) {
  for (int i = n - 1; i >= 0; i--) {
    T sum = y[i];
    for (int j = i + 1; j < n; j++) {
      sum = sum - L[j * n + i] * x[j];
    }
    x[i] = sum / L[i * n + i];
  }
}

template <typename T>
inline bool nngp_chol(const std::vector<T>& A, int n, std::vector<T>& L,
                      double ridge = NNGP_RIDGE) {
  L.assign(static_cast<size_t>(n) * n, T(0.0));
  return nngp_chol(A.data(), n, L.data(), ridge);
}

template <typename T>
inline void nngp_forward_solve(const std::vector<T>& L, int n,
                               const std::vector<T>& b, std::vector<T>& y) {
  y.resize(n);
  nngp_forward_solve(L.data(), n, b.data(), y.data());
}

template <typename T>
inline void nngp_back_solve(const std::vector<T>& L, int n,
                            const std::vector<T>& y, std::vector<T>& x) {
  x.resize(n);
  nngp_back_solve(L.data(), n, y.data(), x.data());
}

constexpr double NNGP_MIN_COND_VAR = 1e-10;

template <typename T>
inline T nngp_floor_cond_var(const T& v) {
  if (get_value(v) < NNGP_MIN_COND_VAR) return T(NNGP_MIN_COND_VAR);
  return v;
}

}
#endif
