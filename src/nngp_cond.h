// nngp_cond.h
// Shared Vecchia/NNGP conditional kernel (gcol33/tulpa#142 A3).
//
// The per-location NNGP conditional -- factor the neighbour covariance, krige
// the conditional mean, floor the conditional variance, accumulate a Gaussian
// log-density -- was transcribed four times: tulpa_svc::nngp_log_lik (double),
// tulpa_svc_ad::nngp_log_lik (AD), tulpa_gp::gp_nngp_log_lik_t (AD) and
// tulpa_gp::gp_nngp_log_lik (double, via Eigen). Each carried its own Cholesky
// and its own conditioning constants, and they had drifted apart.
//
// tulpa_linalg::nngp_conditional_moments already existed for this, but it is
// double*-typed, so the AD copies could not use it and were skipped by the #109
// consolidation -- which is why that pass landed on the (uncalled) double SVC
// function while the live AD one kept its own literals.
//
// The conditioning policy is a deliberate per-kernel ARGUMENT, not a default:
// the SVC kernel intentionally runs a looser jitter/floor than the GP one (see
// hmc_svc.h and NEWS #109). What must NOT differ is a kernel from its own twin:
// an AD copy and its double copy are the same function and must agree, or the
// value and the finite-differenced gradient describe different models.
//
// Jitter is applied as an unconditional diagonal nugget: C_jj + jitter before
// factorizing. That is what every double path does. (The GP AD copy used to add
// it only to an already-degenerate pivot, so on well-conditioned input it
// evaluated a different log-density than the double copy its gradients are
// finite-differenced from.)

#ifndef TULPA_NNGP_COND_H
#define TULPA_NNGP_COND_H

#include <vector>
#include <cmath>
#include <algorithm>

#include "autodiff_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_nngp {

// safe_log / safe_sqrt / get_value are overloaded for double and for every AD
// type (tape, forward, arena), which is what lets one kernel serve the value and
// gradient paths -- the thing the double-only tulpa_linalg core could not do.
using tulpa::math::safe_log;
using tulpa::math::safe_sqrt;
using tulpa::math::get_value;

// How a conditional variance at or below the floor is handled.
enum class VarFloor {
  // cond_var <- floor. Discards the gradient through the floor.
  Clamp,
  // cond_var <- 0.99 * floor + 0.01 * cond_var. Keeps a little gradient so an
  // HMC trajectory that grazes the floor is not handed a flat direction.
  Blend
};

// Lower Cholesky of the n x n row-major A, with `jitter` added to every
// diagonal first. Returns false (leaving L unusable) on a non-positive pivot.
template <typename T>
inline bool chol_decomp(const std::vector<T>& A, int n, std::vector<T>& L,
                        double jitter) {
  L.assign(static_cast<std::size_t>(n) * n, T(0.0));
  for (int j = 0; j < n; j++) {
    for (int k = 0; k <= j; k++) {
      T sum = A[j * n + k];
      if (j == k) sum = sum + T(jitter);
      for (int m = 0; m < k; m++) {
        sum = sum - L[j * n + m] * L[k * n + m];
      }
      if (j == k) {
        if (get_value(sum) <= 1e-12) return false;  // not positive definite
        L[j * n + j] = safe_sqrt(sum);
      } else {
        L[j * n + k] = sum / L[k * n + k];
      }
    }
  }
  return true;
}

// Solve L y = b for lower-triangular L.
template <typename T>
inline void solve_lower(const std::vector<T>& L, int n, const std::vector<T>& b,
                        std::vector<T>& y) {
  y.resize(n);
  for (int j = 0; j < n; j++) {
    T sum = b[j];
    for (int k = 0; k < j; k++) sum = sum - L[j * n + k] * y[k];
    y[j] = sum / L[j * n + j];
  }
}

// Solve L' x = y for lower-triangular L.
template <typename T>
inline void solve_upper(const std::vector<T>& L, int n, const std::vector<T>& y,
                        std::vector<T>& x) {
  x.resize(n);
  for (int j = n - 1; j >= 0; j--) {
    T sum = y[j];
    for (int k = j + 1; k < n; k++) sum = sum - L[k * n + j] * x[k];
    x[j] = sum / L[j * n + j];
  }
}

// Kriging moments for one location given its neighbour covariance C, the
// cross-covariance c_vec, and the neighbour values w_nb (in c_vec order):
//   cond_mean = c' C^-1 w_nb,  cond_var = floor(sigma2 - c' C^-1 c).
// Returns false if C could not be factorized; the caller decides the penalty.
template <typename T>
inline bool cond_moments(const std::vector<T>& C, const std::vector<T>& c_vec,
                         const std::vector<T>& w_nb, int n, const T& sigma2,
                         double jitter, double var_floor, VarFloor floor_mode,
                         T& cond_mean, T& cond_var) {
  std::vector<T> L;
  if (!chol_decomp(C, n, L, jitter)) return false;

  std::vector<T> y, alpha;
  solve_lower(L, n, c_vec, y);
  solve_upper(L, n, y, alpha);

  T cm = T(0.0);
  T c_Cinv_c = T(0.0);
  for (int j = 0; j < n; j++) {
    cm = cm + alpha[j] * w_nb[j];
    c_Cinv_c = c_Cinv_c + c_vec[j] * alpha[j];
  }
  cond_mean = cm;
  cond_var = sigma2 - c_Cinv_c;

  if (get_value(cond_var) < var_floor) {
    cond_var = (floor_mode == VarFloor::Blend)
                   ? T(var_floor * 0.99) + cond_var * T(0.01)
                   : T(var_floor);
  }
  return true;
}

// log N(w_i; cond_mean, cond_var).
template <typename T>
inline T cond_log_density(const T& w_i, const T& cond_mean, const T& cond_var) {
  T resid = w_i - cond_mean;
  return T(-0.5) * safe_log(T(2.0 * M_PI) * cond_var)
       - T(0.5) * resid * resid / cond_var;
}

// log N(w_i; 0, sigma2) -- the marginal arm, used for the first location in the
// Vecchia order and for any location left with no neighbours.
template <typename T>
inline T marginal_log_density(const T& w_i, const T& sigma2) {
  return T(-0.5) * safe_log(T(2.0 * M_PI) * sigma2)
       - T(0.5) * w_i * w_i / sigma2;
}

// Per-location Vecchia conditional-gradient assembly (double path), shared by
// the GP and SVC NNGP analytic gradients (gcol33/tulpa#142). The two differ only
// in how they factorize C (Eigen LLT / CG vs a hand-rolled Cholesky) and how
// they source the pairwise `dC/dphi` (a cached distance table vs recomputed
// coordinates); the arithmetic below is identical, so each caller does its own
// solve, builds `dC` (row-major n*n, zero diagonal), and calls this.
//
// Given the already-solved alpha = C^-1 c and beta = C^-1 w_nb, the
// cross-covariance `c` and its phi-derivative `dc`, `dC`, the neighbour values
// `w_nb` and the location's own `w_obs`, it returns the gradient of the Vecchia
// conditional log-density on the log scale: add `grad_w_obs` to grad_w[obs], add
// `alpha[j] * r_over_v` to each neighbour's grad_w, and add `dlog_sigma2` /
// `dlog_phi` to the corresponding hyperparameter gradients.
struct VecchiaGrad {
  double grad_w_obs;
  double r_over_v;
  double dlog_sigma2;
  double dlog_phi;
};

inline VecchiaGrad vecchia_cond_grad(
    int n_nb, const double* alpha, const double* beta,
    const double* c, const double* dc, const double* dC,
    const double* w_nb, double w_obs, double sigma2, double phi,
    double var_floor) {
  double mu = 0.0, c_alpha = 0.0;
  for (int j = 0; j < n_nb; j++) {
    mu += alpha[j] * w_nb[j];
    c_alpha += c[j] * alpha[j];
  }
  double v = std::max(sigma2 - c_alpha, var_floor);
  double r = w_obs - mu;
  double r_over_v = r / v;
  double dll_dv = 0.5 * (r * r / v - 1.0) / v;
  double dlog_sigma2 = dll_dv * (1.0 - c_alpha / sigma2) * sigma2;

  double alpha_dc = 0.0, dc_beta = 0.0;
  for (int j = 0; j < n_nb; j++) {
    alpha_dc += alpha[j] * dc[j];
    dc_beta += dc[j] * beta[j];
  }
  double alpha_dC_alpha = 0.0, alpha_dC_beta = 0.0;
  for (int j1 = 0; j1 < n_nb; j1++) {
    for (int j2 = 0; j2 < n_nb; j2++) {
      double d = dC[j1 * n_nb + j2];
      alpha_dC_alpha += alpha[j1] * d * alpha[j2];
      alpha_dC_beta += alpha[j1] * d * beta[j2];
    }
  }
  double dv_dphi = -2.0 * alpha_dc + alpha_dC_alpha;
  double dr_dphi = -dc_beta + alpha_dC_beta;
  double dlog_phi = (dll_dv * dv_dphi + (-r / v) * dr_dphi) * phi;

  return { -r_over_v, r_over_v, dlog_sigma2, dlog_phi };
}

}  // namespace tulpa_nngp

#endif  // TULPA_NNGP_COND_H
