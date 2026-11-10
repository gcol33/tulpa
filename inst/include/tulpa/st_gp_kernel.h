#ifndef TULPA_ST_GP_KERNEL_H
#define TULPA_ST_GP_KERNEL_H
#include <cmath>
#include <limits>
#include <vector>
#include <tulpa/ad_scalar_math.h>
#include <tulpa/gp_conditioning.h>
#include <tulpa/soft_sum_to_zero.h>
namespace tulpa_st_gp {
using namespace tulpa::math;
template <typename Scalar>
inline Scalar gneiting_corr(double h, double u, const Scalar& phi_space,
                            const Scalar& phi_time) {
  const Scalar ut = Scalar(std::abs(u)) / phi_time;
  const Scalar base = ut * ut + Scalar(1.0);
  const Scalar hs = Scalar(h) / phi_space;
  return safe_exp(-hs / safe_sqrt(base)) / base;
}

template <typename Scalar, class Data>
inline Scalar st_corr(double h, double u, const Scalar& phi_space,
                      const Scalar& phi_time, const Data& st) {
  if (st.nonsep_type == decltype(st.nonsep_type)::GNEITING) {
    return gneiting_corr(h, u, phi_space, phi_time);
  }
  const Scalar ks = st_cov_value(h, Scalar(1.0), phi_space, st.cov_space);
  const Scalar kt = st_cov_value(u, Scalar(1.0), phi_time, st.cov_time);
  return ks * kt;
}

template <typename Scalar, class Data>
inline Scalar st_cov(double h, double u, const Scalar& sigma2,
                     const Scalar& phi_space, const Scalar& phi_time,
                     const Data& st) {
  return sigma2 * st_corr(h, u, phi_space, phi_time, st);
}

template <class Data>
inline void st_corr_dphi(double h, double u, double phi_space, double phi_time,
                         const Data& st,
                         double* dr_dphi_space, double* dr_dphi_time) {
  if (st.nonsep_type == decltype(st.nonsep_type)::GNEITING) {
    const double ut = std::abs(u) / phi_time;
    const double base = ut * ut + 1.0;
    const double sqrt_base = std::sqrt(base);
    const double hs = h / phi_space;
    const double r = std::exp(-hs / sqrt_base) / base;


    *dr_dphi_space = r * hs / (phi_space * sqrt_base);

    const double dr_dbase = r * (-1.0 / base + 0.5 * hs / (base * sqrt_base));
    *dr_dphi_time = dr_dbase * (-2.0 * ut * ut / phi_time);
    return;
  }
  const double ks = st_cov_value(h, 1.0, phi_space, st.cov_space);
  const double kt = st_cov_value(u, 1.0, phi_time, st.cov_time);
  const double dks = st_cov_dphi(h, 1.0, phi_space, ks, st.cov_space);
  const double dkt = st_cov_dphi(u, 1.0, phi_time, kt, st.cov_time);
  *dr_dphi_space = dks * kt;
  *dr_dphi_time = ks * dkt;
}

template <class Data>
inline void st_pair_separation(const Data& st, int a, int b,
                               double* h, double* u) {
  if (a == b) {
    *h = 0.0;
    *u = 0.0;
    return;
  }
  const double dx = st.coords[a * 2] - st.coords[b * 2];
  const double dy = st.coords[a * 2 + 1] - st.coords[b * 2 + 1];
  *h = std::sqrt(dx * dx + dy * dy);
  *u = std::abs(st.time_values[a] - st.time_values[b]);
}

template <class Data>
inline bool st_gp_structure_filled(const Data& st) {
  const int N = st.n_params;
  if (N <= 0 || st.nn <= 0) return false;
  if (static_cast<int>(st.nn_order.size()) != N) return false;
  if (static_cast<int>(st.coords.size()) != 2 * N) return false;
  if (static_cast<int>(st.time_values.size()) != N) return false;
  const size_t need = static_cast<size_t>(N) * st.nn;
  return st.nn_idx.size() == need && st.nn_dist_space.size() == need &&
         st.nn_dist_time.size() == need;
}

template <typename Scalar, class Data, typename Visit>
inline bool st_gp_nngp_scan(const Scalar* w, const Scalar& sigma2,
                            const Scalar& phi_space, const Scalar& phi_time,
                            const Data& st, Visit&& visit) {
  if (!st_gp_structure_filled(st)) return false;

  const int N = st.n_params;
  const int nn = st.nn;
  const Scalar zero(0.0);

  visit(0, st.nn_order[0], static_cast<const int*>(nullptr), 0,
        static_cast<const Scalar*>(nullptr), static_cast<const Scalar*>(nullptr),
        static_cast<const Scalar*>(nullptr), zero, sigma2, false);

  std::vector<int> nb;
  std::vector<Scalar> c_vec, C_mat, L, y, alpha;

  for (int i = 1; i < N; i++) {
    const int obs_idx = st.nn_order[i];

    int n_nb = 0;
    for (int j = 0; j < nn; j++) {
      if (st.nn_idx[static_cast<size_t>(i) * nn + j] > 0) n_nb++;
    }
    if (n_nb == 0) {
      visit(i, obs_idx, static_cast<const int*>(nullptr), 0,
            static_cast<const Scalar*>(nullptr), static_cast<const Scalar*>(nullptr),
            static_cast<const Scalar*>(nullptr), zero, sigma2, false);
      continue;
    }

    nb.resize(n_nb);
    c_vec.resize(n_nb);
    for (int j = 0; j < n_nb; j++) {
      const size_t f = static_cast<size_t>(i) * nn + j;
      nb[j] = st.nn_order[st.nn_idx[f] - 1];
      c_vec[j] = st_cov(st.nn_dist_space[f], st.nn_dist_time[f], sigma2,
                        phi_space, phi_time, st);
    }

    C_mat.assign(static_cast<size_t>(n_nb) * n_nb, zero);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = j1; j2 < n_nb; j2++) {
        double h12, u12;
        st_pair_separation(st, nb[j1], nb[j2], &h12, &u12);
        const Scalar cov_val = st_cov(h12, u12, sigma2, phi_space, phi_time, st);
        C_mat[static_cast<size_t>(j1) * n_nb + j2] = cov_val;
        C_mat[static_cast<size_t>(j2) * n_nb + j1] = cov_val;
      }
    }

    if (!tulpa_gp_conditioning::nngp_chol(C_mat, n_nb, L)) return false;
    tulpa_gp_conditioning::nngp_forward_solve(L, n_nb, c_vec, y);
    tulpa_gp_conditioning::nngp_back_solve(L, n_nb, y, alpha);

    Scalar cond_mean = zero;
    Scalar c_Cinv_c = zero;
    for (int j = 0; j < n_nb; j++) {
      cond_mean = cond_mean + alpha[j] * w[nb[j]];
      c_Cinv_c = c_Cinv_c + c_vec[j] * alpha[j];
    }
    const Scalar cond_var_raw = sigma2 - c_Cinv_c;
    const bool floored =
        (get_value(cond_var_raw) < tulpa_gp_conditioning::NNGP_MIN_COND_VAR);
    const Scalar cond_var = tulpa_gp_conditioning::nngp_floor_cond_var(cond_var_raw);

    visit(i, obs_idx, static_cast<const int*>(nb.data()), n_nb,
          static_cast<const Scalar*>(L.data()),
          static_cast<const Scalar*>(c_vec.data()),
          static_cast<const Scalar*>(alpha.data()), cond_mean, cond_var, floored);
  }

  return true;
}

template <typename Scalar, class Data>
inline Scalar st_gp_nngp_log_lik(
    const Scalar* w,
    const Scalar& sigma2,
    const Scalar& phi_space,
    const Scalar& phi_time,
    const Data& st_data
) {
  Scalar log_lik = Scalar(0.0);
  const bool ok = st_gp_nngp_scan<Scalar>(
      w, sigma2, phi_space, phi_time, st_data,
      [&](int /*i*/, int obs_idx, const int* /*nb*/, int /*n_nb*/,
          const Scalar* /*L*/, const Scalar* /*c_vec*/, const Scalar* /*alpha*/,
          const Scalar& cond_mean, const Scalar& cond_var, bool /*floored*/) {
        const Scalar resid = w[obs_idx] - cond_mean;
        log_lik = log_lik - Scalar(0.5) * safe_log(Scalar(2.0 * M_PI) * cond_var)
                          - Scalar(0.5) * resid * resid / cond_var;
      });
  if (!ok) return Scalar(-std::numeric_limits<double>::infinity());
  return log_lik;
}

template <class Data, typename Emit>
inline void emit_margin_factor(const Data& st, Emit&& emit) {
  const int S = st.n_spatial, T = st.n_times;
  const double rs = std::sqrt(tulpa::s2z_precision(S));
  const double rt = std::sqrt(tulpa::s2z_precision(T));
  for (int s = 0; s < S; s++) {
    for (int t = 0; t < T; t++) {
      emit(s * T + t, t, rs);
      emit(s * T + t, T + s, rt);
    }
  }
}

template <typename Scalar>
struct StGpAutoregression {
  std::vector<int> order, start, nb;
  std::vector<Scalar> alpha, cond_var;
  std::vector<char> floored;
  void reset(int n) {
    order.resize(n); cond_var.resize(n); floored.resize(n);
    start.assign(n + 1, 0); nb.clear(); alpha.clear();
  }
  void append(int i, int location, const int* neighbors, int count,
              const Scalar* weights, const Scalar& variance, bool at_floor) {
    order[i] = location; cond_var[i] = variance; floored[i] = at_floor;
    for (int j = 0; j < count; ++j) {
      nb.push_back(neighbors[j]); alpha.push_back(weights[j]);
    }
    start[i + 1] = static_cast<int>(nb.size());
  }
};

template <typename Scalar>
inline void st_gp_autoregression_back(const StGpAutoregression<Scalar>& ar,
                                      int m, std::vector<Scalar>& Y) {
  const int N = static_cast<int>(ar.order.size());
  for (int i = N - 1; i >= 0; i--) {
    const Scalar* yi = &Y[static_cast<size_t>(ar.order[i]) * m];
    for (int q = ar.start[i]; q < ar.start[i + 1]; q++) {
      Scalar* yn = &Y[static_cast<size_t>(ar.nb[q]) * m];
      for (int c = 0; c < m; c++) yn[c] = yn[c] + ar.alpha[q] * yi[c];
    }
  }
}

template<class Scalar, class Data>
bool st_gp_build_autoregression(const Scalar& sigma2, const Scalar& space,
    const Scalar& time, const Data& st, StGpAutoregression<Scalar>& ar) {
  ar.reset(st.n_params);
  std::vector<Scalar> zero(st.n_params, Scalar(0.0));
  return st_gp_nngp_scan(zero.data(), sigma2, space, time, st,
    [&](int i, int location, const int* neighbors, int count, const Scalar*,
        const Scalar*, const Scalar* weights, const Scalar&, const Scalar& variance, bool at_floor) {
      ar.append(i, location, neighbors, count, weights, variance, at_floor);
    });
}

// Each rank-one contraction conditions the GP on one Gaussian margin.
// Their product factors (Sigma^-1 + B B')^-1 without an N-by-N factorization.
template<class Scalar>
struct StGpCoordinates {
  StGpAutoregression<Scalar> ar;
  std::vector<Scalar> roots, directions, shrink, expand;
  Scalar log_jacobian = Scalar(0.0);

  void contract(std::vector<Scalar>& value, int column, bool inverse) const {
    const int n = static_cast<int>(roots.size());
    const Scalar* direction = directions.data() + static_cast<size_t>(column) * n;
    Scalar dot(0.0);
    for (int i = 0; i < n; ++i) dot = dot + direction[i] * value[i];
    const Scalar coefficient = (inverse ? expand[column] : -shrink[column]) * dot;
    for (int i = 0; i < n; ++i) value[i] = value[i] + coefficient * direction[i];
  }

  template<class Data>
  bool build(const Scalar& sigma2, const Scalar& space, const Scalar& time, const Data& st) {
    const int n = st.n_params, m = st.n_spatial + st.n_times;
    if (n != st.n_spatial * st.n_times ||
        !st_gp_build_autoregression(sigma2, space, time, st, ar)) return false;
    roots.assign(n, Scalar(0.0));
    log_jacobian = Scalar(0.0);
    std::vector<Scalar> factor(static_cast<size_t>(n) * m, Scalar(0.0));
    emit_margin_factor(st, [&](int cell, int column, double value) {
      factor[static_cast<size_t>(cell) * m + column] = Scalar(value);
    });
    st_gp_autoregression_back(ar, m, factor);
    for (int i = 0; i < n; ++i) {
      const int location = ar.order[i];
      roots[location] = safe_sqrt(ar.cond_var[i]);
      log_jacobian = log_jacobian + safe_log(roots[location]);
      for (int c = 0; c < m; ++c)
        factor[static_cast<size_t>(location) * m + c] =
          factor[static_cast<size_t>(location) * m + c] * roots[location];
    }
    directions.assign(static_cast<size_t>(n) * m, Scalar(0.0));
    shrink.resize(m); expand.resize(m);
    std::vector<Scalar> value(n, Scalar(0.0));
    for (int c = 0; c < m; ++c) {
      for (int i = 0; i < n; ++i) value[i] = factor[static_cast<size_t>(i) * m + c];
      for (int j = 0; j < c; ++j) contract(value, j, false);
      Scalar squared_norm(0.0);
      for (int i = 0; i < n; ++i) {
        directions[static_cast<size_t>(c) * n + i] = value[i];
        squared_norm = squared_norm + value[i] * value[i];
      }
      const Scalar root = safe_sqrt(Scalar(1.0) + squared_norm);
      expand[c] = Scalar(1.0) / (root + Scalar(1.0));
      shrink[c] = expand[c] / root;
      log_jacobian = log_jacobian - safe_log(root);
    }
    return true;
  }

  void forward(const Scalar* z, Scalar* field) const {
    const int n = static_cast<int>(roots.size());
    std::vector<Scalar> value(z, z + n);
    for (int c = static_cast<int>(shrink.size()) - 1; c >= 0; --c) contract(value, c, false);
    for (int i = 0; i < n; ++i) {
      const int location = ar.order[i];
      Scalar current = roots[location] * value[location];
      for (int j = ar.start[i]; j < ar.start[i + 1]; ++j)
        current = current + ar.alpha[j] * field[ar.nb[j]];
      field[location] = current;
    }
  }

  void inverse(const Scalar* field, Scalar* z) const {
    const int n = static_cast<int>(roots.size());
    std::vector<Scalar> value(n, Scalar(0.0));
    for (int i = 0; i < n; ++i) {
      const int location = ar.order[i];
      Scalar current = field[location];
      for (int j = ar.start[i]; j < ar.start[i + 1]; ++j)
        current = current - ar.alpha[j] * field[ar.nb[j]];
      value[location] = current / roots[location];
    }
    for (int c = 0; c < static_cast<int>(expand.size()); ++c) contract(value, c, true);
    for (int i = 0; i < n; ++i) z[i] = value[i];
  }
};

template<class Scalar, class Data>
bool st_gp_nc_forward(const Scalar* z, const Scalar& sigma2, const Scalar& space,
    const Scalar& time, const Data& st, Scalar* field, Scalar* log_jacobian = nullptr) {
  StGpCoordinates<Scalar> coordinates;
  if (!coordinates.build(sigma2, space, time, st)) return false;
  coordinates.forward(z, field);
  if (log_jacobian) *log_jacobian = coordinates.log_jacobian;
  return true;
}

template<class Scalar, class Data>
bool st_gp_nc_inverse(const Scalar* field, const Scalar& sigma2, const Scalar& space,
    const Scalar& time, const Data& st, Scalar* z) {
  StGpCoordinates<Scalar> coordinates;
  if (!coordinates.build(sigma2, space, time, st)) return false;
  coordinates.inverse(field, z);
  return true;
}

template <typename Scalar, class Data>
inline bool st_gp_margin_factor(const StGpAutoregression<Scalar>& ar,
                                const Data& st,
                                std::vector<Scalar>& X, std::vector<Scalar>& LM) {
  const int N = st.n_params;
  const int m = st.n_spatial + st.n_times;
  X.assign(static_cast<size_t>(N) * m, Scalar(0.0));
  emit_margin_factor(st, [&](int cell, int col, double v) {
    X[static_cast<size_t>(cell) * m + col] = Scalar(v);
  });
  st_gp_autoregression_back(ar, m, X);

  std::vector<Scalar> M(static_cast<size_t>(m) * m, Scalar(0.0));
  for (int c = 0; c < m; c++) M[static_cast<size_t>(c) * m + c] = Scalar(1.0);
  for (int i = 0; i < N; i++) {
    const Scalar* xi = &X[static_cast<size_t>(ar.order[i]) * m];
    const Scalar d = ar.cond_var[i];
    for (int c1 = 0; c1 < m; c1++) {
      const Scalar dx = d * xi[c1];
      for (int c2 = c1; c2 < m; c2++) {
        M[static_cast<size_t>(c1) * m + c2] = M[static_cast<size_t>(c1) * m + c2] + dx * xi[c2];
      }
    }
  }
  for (int c1 = 0; c1 < m; c1++) {
    for (int c2 = 0; c2 < c1; c2++) {
      M[static_cast<size_t>(c1) * m + c2] = M[static_cast<size_t>(c2) * m + c1];
    }
  }
  return tulpa_gp_conditioning::nngp_chol(M, m, LM, 0.0);
}

template <typename Scalar>
inline Scalar st_gp_margin_half_logdet(const std::vector<Scalar>& LM, int m) {
  Scalar h(0.0);
  for (int c = 0; c < m; c++) h = h + safe_log(LM[static_cast<size_t>(c) * m + c]);
  return h;
}

template <typename Scalar, class Data>
inline bool st_gp_margin_log_normalizer(const Scalar& sigma2,
                                        const Scalar& phi_space,
                                        const Scalar& phi_time,
                                        const Data& st,
                                        Scalar* out) {
  const int N = st.n_params;
  if (N != st.n_spatial * st.n_times) return false;

  StGpAutoregression<Scalar> ar;
  if (!st_gp_build_autoregression(sigma2, phi_space, phi_time, st, ar)) return false;

  std::vector<Scalar> X, LM;
  if (!st_gp_margin_factor(ar, st, X, LM)) return false;
  *out = st_gp_margin_half_logdet(LM, st.n_spatial + st.n_times);
  return true;
}

struct StGpMarginNormalizer {
  double value = 0.0;
  double log_tau = 0.0;
  double log_phi_space = 0.0;
  double log_phi_time = 0.0;
  bool ok = true;
};

// A fixed neighbour ridge makes the regression weights depend on precision.
inline void st_gp_log_precision_direction(const double* L, const double* covariance,
    const double* alpha, int count, double sigma2, bool floored,
    std::vector<double>& scratch, std::vector<double>& dalpha, double& dvariance) {
  scratch.resize(count); dalpha.resize(count);
  double raw = sigma2, squared_norm = 0.0;
  for (int j = 0; j < count; ++j) {
    raw -= covariance[j] * alpha[j];
    squared_norm += alpha[j] * alpha[j];
  }
  dvariance = floored ? 0.0 : -raw + tulpa_gp_conditioning::NNGP_RIDGE * squared_norm;
  tulpa_gp_conditioning::nngp_forward_solve(L, count, alpha, scratch.data());
  tulpa_gp_conditioning::nngp_back_solve(L, count, scratch.data(), dalpha.data());
  for (double& value : dalpha) value *= -tulpa_gp_conditioning::NNGP_RIDGE;
}

template <class Data>
inline StGpMarginNormalizer st_gp_margin_normalizer_grad(
    double tau, double phi_space, double phi_time, const Data& st) {
  StGpMarginNormalizer out;
  const int N = st.n_params;
  const int m = st.n_spatial + st.n_times;
  const int nn = st.nn;
  if (N != st.n_spatial * st.n_times) {
    out.ok = false;
    return out;
  }
  const double sigma2 = 1.0 / tau;

  StGpAutoregression<double> ar;
  ar.reset(N);
  std::vector<double> dalpha_tau, dalpha_s, dalpha_t, dd_tau(N, 0.0), dd_s(N, 0.0), dd_t(N, 0.0);
  std::vector<double> dc_s, dc_t, dC_s, dC_t, v, tmp, da;
  std::vector<double> zeros(N, 0.0);

  out.ok = st_gp_nngp_scan<double>(
      zeros.data(), sigma2, phi_space, phi_time, st,
      [&](int i, int obs_idx, const int* nb, int n_nb, const double* L,
          const double* c_vec, const double* alpha,
          const double& /*cond_mean*/, const double& cv, bool fl) {
        ar.append(i, obs_idx, nb, n_nb, alpha, cv, fl);
        st_gp_log_precision_direction(L, c_vec, alpha, n_nb, sigma2, fl, tmp, da, dd_tau[i]);
        dalpha_tau.insert(dalpha_tau.end(), da.begin(), da.end());
        if (n_nb == 0) return;

        dc_s.resize(n_nb);
        dc_t.resize(n_nb);
        dC_s.assign(static_cast<size_t>(n_nb) * n_nb, 0.0);
        dC_t.assign(static_cast<size_t>(n_nb) * n_nb, 0.0);
        for (int j = 0; j < n_nb; j++) {
          const size_t f = static_cast<size_t>(i) * nn + j;
          double ds, dt;
          st_corr_dphi(st.nn_dist_space[f], st.nn_dist_time[f], phi_space,
                       phi_time, st, &ds, &dt);
          dc_s[j] = sigma2 * ds;
          dc_t[j] = sigma2 * dt;
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
          for (int j2 = j1; j2 < n_nb; j2++) {
            double h12, u12, ds, dt;
            st_pair_separation(st, nb[j1], nb[j2], &h12, &u12);
            st_corr_dphi(h12, u12, phi_space, phi_time, st, &ds, &dt);
            dC_s[static_cast<size_t>(j1) * n_nb + j2] = sigma2 * ds;
            dC_s[static_cast<size_t>(j2) * n_nb + j1] = sigma2 * ds;
            dC_t[static_cast<size_t>(j1) * n_nb + j2] = sigma2 * dt;
            dC_t[static_cast<size_t>(j2) * n_nb + j1] = sigma2 * dt;
          }
        }

        v.resize(n_nb);
        tmp.resize(n_nb);
        da.resize(n_nb);
        auto direction = [&](const std::vector<double>& dc,
                             const std::vector<double>& dC,
                             std::vector<double>& dalpha_all, double& dd) {
          double dc_a = 0.0, a_dC_a = 0.0;
          for (int j1 = 0; j1 < n_nb; j1++) {
            double row = 0.0;
            for (int j2 = 0; j2 < n_nb; j2++) {
              row += dC[static_cast<size_t>(j1) * n_nb + j2] * alpha[j2];
            }
            v[j1] = dc[j1] - row;
            dc_a += dc[j1] * alpha[j1];
            a_dC_a += alpha[j1] * row;
          }
          tulpa_gp_conditioning::nngp_forward_solve(L, n_nb, v.data(), tmp.data());
          tulpa_gp_conditioning::nngp_back_solve(L, n_nb, tmp.data(), da.data());
          for (int j = 0; j < n_nb; j++) dalpha_all.push_back(da[j]);
          dd = fl ? 0.0 : (-2.0 * dc_a + a_dC_a);
        };
        direction(dc_s, dC_s, dalpha_s, dd_s[i]);
        direction(dc_t, dC_t, dalpha_t, dd_t[i]);
      });
  if (!out.ok) return out;

  std::vector<double> X, LM;
  if (!st_gp_margin_factor(ar, st, X, LM)) {
    out.ok = false;
    return out;
  }
  out.value = st_gp_margin_half_logdet(LM, m);


  std::vector<double> Y(static_cast<size_t>(N) * m), y_tmp(m);
  for (int k = 0; k < N; k++) {
    tulpa_gp_conditioning::nngp_forward_solve(LM.data(), m, &X[static_cast<size_t>(k) * m], y_tmp.data());
    tulpa_gp_conditioning::nngp_back_solve(LM.data(), m, y_tmp.data(), &Y[static_cast<size_t>(k) * m]);
  }
  std::vector<double> q(N, 0.0);
  for (int i = 0; i < N; i++) {
    const size_t r = static_cast<size_t>(ar.order[i]) * m;
    for (int c = 0; c < m; c++) q[i] += X[r + c] * Y[r + c];
  }

  auto direction_slope = [&](const std::vector<double>& dalpha,
                         const std::vector<double>& dd) {
    std::vector<double> dX(static_cast<size_t>(N) * m, 0.0);
    for (int i = 0; i < N; i++) {
      const double* xi = &X[static_cast<size_t>(ar.order[i]) * m];
      for (int qn = ar.start[i]; qn < ar.start[i + 1]; qn++) {
        double* rn = &dX[static_cast<size_t>(ar.nb[qn]) * m];
        for (int c = 0; c < m; c++) rn[c] += dalpha[qn] * xi[c];
      }
    }
    st_gp_autoregression_back(ar, m, dX);
    double g = 0.0;
    for (int i = 0; i < N; i++) {
      const size_t r = static_cast<size_t>(ar.order[i]) * m;
      double cross = 0.0;
      for (int c = 0; c < m; c++) cross += dX[r + c] * Y[r + c];
      g += 0.5 * dd[i] * q[i] + ar.cond_var[i] * cross;
    }
    return g;
  };
  out.log_tau = direction_slope(dalpha_tau, dd_tau);
  out.log_phi_space = direction_slope(dalpha_s, dd_s) * phi_space;
  out.log_phi_time = direction_slope(dalpha_t, dd_t) * phi_time;
  return out;
}

}
#endif
