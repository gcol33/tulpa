#ifndef TULPA_TEMPORAL_NC_H
#define TULPA_TEMPORAL_NC_H

#include <cmath>
#include <vector>

#include <tulpa/sum_to_zero.h>

namespace tulpa_temporal_nc {

inline double lin_basis(int t, int n) {
  return static_cast<double>(t) - 0.5 * (static_cast<double>(n) - 1.0);
}

inline bool nc_applies(int n, int order, bool cyclic) {
  if (cyclic && (order == 1 || order == 2)) return n >= 3;
  if (order == 1) return n >= 2;
  return !cyclic && n >= 3;
}

inline int nc_normal_dim(int n, int order, bool cyclic) {
  if (!nc_applies(n, order, cyclic)) return 0;
  if (order == 1 || cyclic) return n;
  return n - 1;
}

template <typename T>
inline void rw_nc_forward(const T* z, int n, int order, bool cyclic,
                          const T& sigma, T* phi) {
  if (!nc_applies(n, order, cyclic)) {
    for (int t = 0; t < n; t++) phi[t] = z[t];
    return;
  }
  const double dn = static_cast<double>(n);
  const double sqrt_n = std::sqrt(dn);

  if (order == 2 && cyclic) {
    std::vector<T> first(n);
    rw_nc_forward(z, n, 1, true, T(1.0), first.data());
    rw_nc_forward(first.data(), n, 1, true, sigma, phi);
    return;
  }

  if (order == 1 && !cyclic) {

    T run = T(0.0), sum_u = T(0.0);
    for (int t = 0; t < n; t++) {
      if (t > 0) run = run + z[t - 1];
      phi[t] = run;
      sum_u = sum_u + run;
    }
    const T a = (T(sqrt_n) * z[n - 1] - sum_u) / T(dn);
    for (int t = 0; t < n; t++) phi[t] = sigma * (phi[t] + a);
    return;
  }

  if (order == 1 && cyclic) {

    T zbar = T(0.0);
    for (int t = 0; t < n; t++) zbar = zbar + z[t];
    zbar = zbar / T(dn);

    T run = T(0.0), sum_w = T(0.0);
    for (int t = 0; t < n; t++) {
      if (t > 0) run = run + (z[t] - zbar);
      phi[t] = run;
      sum_w = sum_w + run;
    }
    const T wbar = sum_w / T(dn);
    for (int t = 0; t < n; t++) phi[t] = sigma * (phi[t] - wbar + zbar);
    return;
  }

  T sum_v = T(0.0);
  for (int t = 0; t < n; t++) {
    if (t < 2) {
      phi[t] = T(0.0);
    } else {
      phi[t] = T(2.0) * phi[t - 1] - phi[t - 2] + z[t - 2];
    }
    sum_v = sum_v + phi[t];
  }
  T vL = T(0.0);
  double LL = 0.0;
  for (int t = 0; t < n; t++) {
    const double L = lin_basis(t, n);
    vL = vL + phi[t] * T(L);
    LL += L * L;
  }
  const T cL = vL / T(LL);
  for (int t = 0; t < n; t++) phi[t] = phi[t] - cL * T(lin_basis(t, n));

  const T a = (T(sqrt_n) * z[n - 2] - sum_v) / T(dn);
  for (int t = 0; t < n; t++) {
    phi[t] = sigma * (phi[t] + a) + z[n - 1] * T(lin_basis(t, n));
  }
}

inline void rw_nc_backward(const double* g, const double* phi, const double* z,
                           int n, int order, bool cyclic, double sigma,
                           double* gz, double* g_log_sigma2) {
  if (!nc_applies(n, order, cyclic)) {
    for (int t = 0; t < n; t++) gz[t] += g[t];
    return;
  }
  const double dn = static_cast<double>(n);
  const double sqrt_n = std::sqrt(dn);

  if (order == 2 && cyclic) {
    std::vector<double> first(n), first_gradient(n, 0.0);
    rw_nc_forward(z, n, 1, true, 1.0, first.data());
    rw_nc_backward(g, phi, first.data(), n, 1, true, sigma,
      first_gradient.data(), nullptr);
    rw_nc_backward(first_gradient.data(), first.data(), z, n, 1, true, 1.0,
      gz, nullptr);
    if (g_log_sigma2 != nullptr) {
      double dot = 0.0;
      for (int t = 0; t < n; ++t) dot += g[t] * phi[t];
      *g_log_sigma2 += 0.5 * dot;
    }
    return;
  }

  double gsum = 0.0;
  for (int t = 0; t < n; t++) gsum += g[t];

  if (order == 1 && !cyclic) {

    double tail = 0.0;
    std::vector<double> S1(n + 1, 0.0);
    for (int t = n - 1; t >= 0; t--) { tail += g[t]; S1[t] = tail; }
    for (int k = 0; k <= n - 2; k++) {
      gz[k] += sigma * (S1[k + 1] - (dn - k - 1.0) / dn * gsum);
    }
    gz[n - 1] += sigma * gsum / sqrt_n;
  } else if (order == 1 && cyclic) {

    const double gbar = gsum / dn;
    std::vector<double> y(n, 0.0);
    double run = 0.0;
    for (int k = n - 2; k >= 0; k--) { run += (g[k] - gbar); y[k] = run; }
    double ybar = 0.0;
    for (int k = 0; k < n; k++) ybar += y[k];
    ybar /= dn;
    for (int k = 0; k < n; k++) gz[k] += sigma * (y[k] - ybar + gbar);
  } else {

    std::vector<double> S1(n + 2, 0.0), S2(n + 2, 0.0);
    for (int t = n - 1; t >= 0; t--) {
      S1[t] = S1[t + 1] + g[t];
      S2[t] = S2[t + 1] + static_cast<double>(t) * g[t];
    }

    std::vector<double> L1(n + 2, 0.0), L2(n + 2, 0.0);
    double LL = 0.0, gL = 0.0;
    for (int t = n - 1; t >= 0; t--) {
      const double L = lin_basis(t, n);
      L1[t] = L1[t + 1] + L;
      L2[t] = L2[t + 1] + static_cast<double>(t) * L;
      LL += L * L;
      gL += g[t] * L;
    }
    for (int k = 0; k <= n - 3; k++) {
      const int j = k + 2;
      const double ramp_g = S2[j] - (k + 1.0) * S1[j];

      const double ramp_L = L2[j] - (k + 1.0) * L1[j];

      const double m = dn - k - 2.0;
      const double ramp_sum = 0.5 * m * (m + 1.0);
      gz[k] += sigma * (ramp_g - ramp_L / LL * gL - ramp_sum / dn * gsum);
    }
    gz[n - 2] += sigma * gsum / sqrt_n;
    double glin = 0.0;
    for (int t = 0; t < n; t++) glin += g[t] * lin_basis(t, n);
    gz[n - 1] += glin;
  }

  if (g_log_sigma2 != nullptr) {

    double dot = 0.0;
    const bool has_lin = (order == 2 && !cyclic);
    for (int t = 0; t < n; t++) {
      const double scaled =
          has_lin ? (phi[t] - z[n - 1] * lin_basis(t, n)) : phi[t];
      dot += g[t] * scaled;
    }
    *g_log_sigma2 += 0.5 * dot;
  }
}

template <typename T>
inline T rw_nc_log_prior(const T* z, int n, int order, bool cyclic) {
  const int d = nc_normal_dim(n, order, cyclic);
  T quad = T(0.0);
  for (int k = 0; k < d; k++) quad = quad + z[k] * z[k];
  return T(-0.5) * quad - T(0.5 * d) * T(std::log(2.0 * std::acos(-1.0)));
}

template <typename T>
inline void rw_nc_grouped_forward(const T* z, int n, int n_groups, int order,
                                  bool cyclic, const T& sigma, T* phi) {
  for (int g = 0; g < n_groups; g++) {
    rw_nc_forward(z + g * n, n, order, cyclic, sigma, phi + g * n);
  }
}

template <typename T>
inline T rw_nc_grouped_log_prior(const T* z, int n, int n_groups, int order,
                                 bool cyclic) {
  T lp = T(0.0);
  for (int g = 0; g < n_groups; g++) {
    lp = lp + rw_nc_log_prior(z + g * n, n, order, cyclic);
  }
  return lp;
}

inline void rw_nc_grouped_log_prior_grad(const double* z, int n, int n_groups,
                                         int order, bool cyclic, double* gz) {
  const int d = nc_normal_dim(n, order, cyclic);
  for (int g = 0; g < n_groups; g++) {
    for (int k = 0; k < d; k++) gz[g * n + k] -= z[g * n + k];
  }
}

inline void rw_nc_grouped_backward(const double* g, const double* z, int n,
                                   int n_groups, int order, bool cyclic,
                                   double sigma, double* gz,
                                   double* g_log_sigma2) {
  std::vector<double> phig(n, 0.0);
  for (int gg = 0; gg < n_groups; gg++) {
    rw_nc_forward(z + gg * n, n, order, cyclic, sigma, phig.data());
    rw_nc_backward(g + gg * n, phig.data(), z + gg * n, n, order, cyclic,
                   sigma, gz + gg * n, g_log_sigma2);
  }
}

}

#endif
