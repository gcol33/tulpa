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
