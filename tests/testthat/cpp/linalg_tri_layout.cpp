// linalg_tri_layout_export.cpp
// Probes for the layout contract of the shared small-dense Cholesky core.
//
// A lower-triangular factor stored column-major is the same bytes as an
// upper-triangular one stored row-major, so a factor handed to a solve written
// for the other convention solves against the transpose and returns a plausible
// vector -- no crash, no NaN, no dimension check. A cuSOLVER factor consumed
// the wrong way round is the case that happens on.
//
// The solves take the layout as a required template argument. These exports
// let test-tri-solve-layout.R pin what each convention reads, and pin that the
// two disagree in exactly the transposed way, so a buffer built for one and
// consumed by the other stays detectable rather than plausible.

#include <Rcpp.h>

#include <vector>

#include "linalg_fast.h"
#include "nngp_cond.h"

namespace {

// 0 = TriLayout::RowMajor, 1 = TriLayout::ColMajor.
constexpr int kRowMajorCode = 0;

}  // namespace

// Solve L y = b (transpose = false) or L' y = b (transpose = true) for the
// lower-triangular factor held in `Lbuf` under `layout`.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_tri_solve(Rcpp::NumericVector Lbuf, int n,
                                       Rcpp::NumericVector b, int layout,
                                       bool transpose) {
  Rcpp::NumericVector out(n);
  const double* L = Lbuf.begin();
  const double* rhs = b.begin();
  double* y = out.begin();
  if (layout == kRowMajorCode) {
    if (transpose) {
      tulpa_linalg::tri_solve_lower_transpose<tulpa_linalg::TriLayout::RowMajor>(
          L, n, n, rhs, y);
    } else {
      tulpa_linalg::tri_solve_lower<tulpa_linalg::TriLayout::RowMajor>(
          L, n, n, rhs, y);
    }
  } else {
    if (transpose) {
      tulpa_linalg::tri_solve_lower_transpose<tulpa_linalg::TriLayout::ColMajor>(
          L, n, n, rhs, y);
    } else {
      tulpa_linalg::tri_solve_lower<tulpa_linalg::TriLayout::ColMajor>(
          L, n, n, rhs, y);
    }
  }
  return out;
}

// Factorize the n x n SPD matrix held in `Abuf` under `layout` and return the
// factor in the same layout. The opposite triangle is zeroed so the buffer can
// be compared against a reference factor entry for entry.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_chol_factor(Rcpp::NumericVector Abuf, int n,
                                         int layout) {
  Rcpp::NumericVector out(static_cast<R_xlen_t>(n) * n);
  const double* A = Abuf.begin();
  double* L = out.begin();
  if (layout == kRowMajorCode) {
    tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
        A, L, n, n, /*nugget=*/0.0);
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) L[i * n + j] = 0.0;
    }
  } else {
    tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::ColMajor>(
        A, L, n, n, /*nugget=*/0.0);
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) L[j * n + i] = 0.0;
    }
  }
  return out;
}

// NNGP kriging moments from an already-factored neighbour covariance, under
// `layout`. This is the call the cuSOLVER factor's layout was got wrong on.
// [[Rcpp::export]]
Rcpp::List cpp_test_nngp_moments(Rcpp::NumericVector Lbuf, int n,
                                 Rcpp::NumericVector c_vec,
                                 Rcpp::NumericVector w_nb, double sigma2,
                                 double var_floor, int layout) {
  double cond_mean = 0.0, cond_var = 0.0;
  std::vector<double> alpha(n, 0.0);
  if (layout == kRowMajorCode) {
    tulpa_linalg::nngp_moments_from_chol<tulpa_linalg::TriLayout::RowMajor>(
        Lbuf.begin(), n, n, c_vec.begin(), w_nb.begin(), sigma2, var_floor,
        cond_mean, cond_var, alpha.data());
  } else {
    tulpa_linalg::nngp_moments_from_chol<tulpa_linalg::TriLayout::ColMajor>(
        Lbuf.begin(), n, n, c_vec.begin(), w_nb.begin(), sigma2, var_floor,
        cond_mean, cond_var, alpha.data());
  }
  return Rcpp::List::create(Rcpp::Named("cond_mean") = cond_mean,
                            Rcpp::Named("cond_var") = cond_var,
                            Rcpp::Named("alpha") = Rcpp::wrap(alpha));
}

// The two NNGP conditional-moment cores on one input. tulpa_nngp::cond_moments
// is the templated core the autodiff and value kernels run;
// tulpa_linalg::nngp_conditional_moments is the double core the Laplace and
// batched paths run. `jitter` is a diagonal NUGGET in both: C_jj + jitter
// before the factorization, so it is part of the density being evaluated. A
// pivot FLOOR at the same number leaves a well-conditioned input untouched and
// replaces an ill-conditioned one, which is a divergence confined to exactly
// the inputs where it matters and invisible in the result.
//
// `ok` is each core's own report of a non-PD neighbour covariance, which the
// two must also agree on: a core that cannot decline hands its caller kriged
// moments off an unusable factor.
// [[Rcpp::export]]
Rcpp::List cpp_test_nngp_cond_cores(Rcpp::NumericVector Cbuf, int n,
                                    Rcpp::NumericVector c_vec,
                                    Rcpp::NumericVector w_nb, double sigma2,
                                    double jitter, double var_floor) {
  std::vector<double> C(Cbuf.begin(), Cbuf.end());
  std::vector<double> c_t(c_vec.begin(), c_vec.end());
  std::vector<double> w_t(w_nb.begin(), w_nb.end());

  double t_mean = 0.0, t_var = 0.0;
  const bool t_ok = tulpa_nngp::cond_moments<double>(
      C, c_t, w_t, n, sigma2, jitter, var_floor,
      tulpa_nngp::VarFloor::Clamp, t_mean, t_var);

  double d_mean = 0.0, d_var = 0.0;
  const bool d_ok = tulpa_linalg::nngp_conditional_moments(
      C.data(), c_t.data(), w_t.data(), n, sigma2, jitter, var_floor,
      d_mean, d_var);

  return Rcpp::List::create(
      Rcpp::Named("templated_ok")   = t_ok,
      Rcpp::Named("templated_mean") = t_mean,
      Rcpp::Named("templated_var")  = t_var,
      Rcpp::Named("plain_ok")       = d_ok,
      Rcpp::Named("plain_mean")     = d_mean,
      Rcpp::Named("plain_var")      = d_var);
}
