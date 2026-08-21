// nngp_cond.h
// Shared Vecchia/NNGP conditional kernel (A3).
//
// The per-location NNGP conditional -- factor the neighbour covariance, krige
// the conditional mean, floor the conditional variance, accumulate a Gaussian
// log-density -- was transcribed four times: tulpa_svc::nngp_log_lik (double),
// tulpa_svc_ad::nngp_log_lik (AD), tulpa_gp::gp_nngp_log_lik_t (AD) and
// tulpa_gp::gp_nngp_log_lik (double, via Eigen). Each carried its own Cholesky
// and its own conditioning constants, and they had drifted apart.
//
// tulpa_linalg::nngp_conditional_moments already existed for this, but it is
// double*-typed, so the AD copies could not use it and were skipped by the
// earlier consolidation -- which is why that pass landed on the (uncalled) double SVC
// function while the live AD one kept its own literals.
//
// The conditioning policy is a deliberate per-kernel ARGUMENT, not a default:
// the SVC kernel intentionally runs a looser jitter/floor than the GP one (see
// hmc_svc.h and the NEWS). What must NOT differ is a kernel from its own twin:
// an AD copy and its double copy are the same function and must agree, or the
// value and the finite-differenced gradient describe different models.
//
// Jitter is a diagonal NUGGET: C_jj + jitter before factorizing. It changes the
// matrix being factorized, so it is part of the density the kernel evaluates,
// and a path that instead FLOORS the pivot at the same number evaluates a
// different density on exactly the ill-conditioned inputs where it matters.
// chol_decomp at T = double IS tulpa_linalg::chol_factor_lower, so the double
// paths and the templated ones cannot drift apart on this again.

#ifndef TULPA_NNGP_COND_H
#define TULPA_NNGP_COND_H

#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "autodiff_utils.h"
#include "linalg_fast.h"  // the double Cholesky core chol_decomp delegates to

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
  // cond_var <- 0.99 * floor + 0.01 * max(cond_var, 0). Keeps a little gradient
  // so an HMC trajectory that grazes the floor is not handed a flat direction.
  Blend
};

// Fraction of d(raw variance) that survives the floor, i.e. d(floored)/d(raw).
// The gradient kernels multiply their unfloored dv/dtheta by this, so a floored
// variance reports the derivative of the value that was actually used rather
// than of the value it replaced.
inline double var_floor_slope(bool floor_bound, bool raw_positive,
                              VarFloor floor_mode) {
  if (!floor_bound) return 1.0;
  if (floor_mode != VarFloor::Blend) return 0.0;
  return raw_positive ? 0.01 : 0.0;
}

// Apply `floor_mode` to a raw conditional variance. The result is strictly
// positive under both policies: Blend mixes against max(raw, 0) rather than
// against raw, because 0.99 * floor + 0.01 * raw is NEGATIVE once raw drops
// below -99 * floor, and a negative variance turns cond_log_density's
// -0.5 * resid^2 / var term POSITIVE and grows it with the residual, so a
// sampler chasing it walks INTO the degenerate region instead of away from it.
// `slope_out`, when non-null, receives d(result)/d(raw).
template <typename T>
inline T apply_var_floor(const T& raw, double var_floor, VarFloor floor_mode,
                         double* slope_out = nullptr) {
  const double raw_val = get_value(raw);
  const bool bound = raw_val < var_floor;
  if (slope_out) {
    *slope_out = var_floor_slope(bound, raw_val > 0.0, floor_mode);
  }
  if (!bound) return raw;
  if (floor_mode != VarFloor::Blend) return T(var_floor);
  if (raw_val > 0.0) return T(var_floor * 0.99) + raw * T(0.01);
  return T(var_floor * 0.99);
}

// Number of neighbours in row `i` of an nn_idx table.
//
// nn_idx rows are LEFT-PACKED: a row's neighbours occupy its leading columns as
// 1-based indices into the Vecchia ordering, and the tail is padded with 0.
// (Every builder in the package emits that shape -- compute_nngp_neighbors()
// and its spatiotemporal twin write nn_idx[i, seq_len(n_found)].) The count is
// therefore the length of the leading RUN of entries in [1, n_order]; the first
// entry outside that range ends the row.
//
// Counting every positive entry in the row instead, which four density kernels
// did, gives the same answer on a left-packed row and an out-of-bounds read on
// any other: for a row [5, 0, 7, ...] it returns 2, and slot 1 then resolves
// nn_order[0 - 1]. Stopping at the first invalid entry means every column below
// the returned count resolves inside nn_order by construction.
//
// `row` points at column 0 of the row and `stride` is the gap between its
// columns -- 1 for the flat [n x nn] row-major buffers GPData / SVCData hold,
// nrow() for an Rcpp::IntegerMatrix.
inline int nngp_row_neighbours(const int* row, int stride, int nn,
                               int n_order) {
  int m = 0;
  while (m < nn) {
    const int v = row[static_cast<std::ptrdiff_t>(m) * stride];
    if (v < 1 || v > n_order) break;
    ++m;
  }
  return m;
}

// Lower Cholesky of the n x n row-major A, with `nugget` added to every
// diagonal first. Returns false (leaving L unusable) on a pivot at or below
// tulpa_linalg::kCholMinPivot.
//
// At T = double this IS tulpa_linalg::chol_factor_lower, so there is one
// implementation of the double factorization in the package rather than a copy
// the AD twin can drift away from. The templated body below serves the autodiff
// types, which the double* core cannot take.
template <typename T>
inline bool chol_decomp(const std::vector<T>& A, int n, std::vector<T>& L,
                        double nugget) {
  L.assign(static_cast<std::size_t>(n) * n, T(0.0));
  if constexpr (std::is_same<T, double>::value) {
    return tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
        A.data(), L.data(), n, n, nugget);
  } else {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k <= j; k++) {
        T sum = A[j * n + k];
        if (j == k) sum = sum + T(nugget);
        for (int m = 0; m < k; m++) {
          sum = sum - L[j * n + m] * L[k * n + m];
        }
        if (j == k) {
          if (get_value(sum) <= tulpa_linalg::kCholMinPivot) return false;
          L[j * n + j] = safe_sqrt(sum);
        } else {
          L[j * n + k] = sum / L[k * n + k];
        }
      }
    }
    return true;
  }
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
                         double nugget, double var_floor, VarFloor floor_mode,
                         T& cond_mean, T& cond_var) {
  std::vector<T> L;
  if (!chol_decomp(C, n, L, nugget)) return false;

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
  cond_var = apply_var_floor(sigma2 - c_Cinv_c, var_floor, floor_mode);
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

// Per-location Vecchia conditional-gradient assembly (double path). A caller
// does its own factorization and solve, builds `dC` (row-major n*n, zero
// diagonal) from whichever distance source it has, and calls this.
//
// Given the already-solved alpha = C^-1 c and beta = C^-1 w_nb, the
// cross-covariance `c` and its phi-derivative `dc`, `dC`, the neighbour values
// `w_nb` and the location's own `w_obs`, it returns the gradient of the Vecchia
// conditional log-density on the log scale: add `grad_w_obs` to grad_w[obs], add
// `alpha[j] * r_over_v` to each neighbour's grad_w, and add `dlog_sigma2` /
// `dlog_phi` to the corresponding hyperparameter gradients.
//
// `var_floor` / `floor_mode` must be the ones the VALUE kernel ran, so this
// differentiates the variance the density actually reported.
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
    double var_floor, VarFloor floor_mode) {
  double mu = 0.0, c_alpha = 0.0;
  for (int j = 0; j < n_nb; j++) {
    mu += alpha[j] * w_nb[j];
    c_alpha += c[j] * alpha[j];
  }
  // The value kernel reports the FLOORED variance, so the hyperparameter
  // derivatives are of the floored quantity: where the floor binds, v no longer
  // moves with sigma2 or phi at all under Clamp, and moves at 1/100 the rate
  // under Blend. `floor_slope` carries that factor, and the value path's own
  // policy is an argument so the two cannot disagree about which one ran.
  double floor_slope = 1.0;
  double v = apply_var_floor(sigma2 - c_alpha, var_floor, floor_mode,
                             &floor_slope);
  double r = w_obs - mu;
  double r_over_v = r / v;
  double dll_dv = 0.5 * (r * r / v - 1.0) / v;
  double dlog_sigma2 = dll_dv * floor_slope * (1.0 - c_alpha / sigma2) * sigma2;

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
  double dv_dphi = floor_slope * (-2.0 * alpha_dc + alpha_dC_alpha);
  double dr_dphi = -dc_beta + alpha_dC_beta;
  double dlog_phi = (dll_dv * dv_dphi + (-r / v) * dr_dphi) * phi;

  return { -r_over_v, r_over_v, dlog_sigma2, dlog_phi };
}

}  // namespace tulpa_nngp

#endif  // TULPA_NNGP_COND_H
