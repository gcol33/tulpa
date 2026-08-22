// test_lkj_chol.cpp
// Rcpp wrappers exposing the lkj_chol_helpers.h functions to R for unit
// testing. The helpers themselves are header-only and namespace-private.

#include <Rcpp.h>
#include <string>
#include <vector>
#include "lkj_chol_helpers.h"

using namespace Rcpp;

namespace {

// Number of strict-lower entries the raw parameterization carries for an
// n x n correlation Cholesky factor. Every helper below reads or writes
// exactly this many raw values.
inline int lkj_raw_len(int n) { return n * (n - 1) / 2; }

inline void check_dim(int n, const char* nm) {
  if (n <= 0) stop("%s must be positive: got %d.", nm, n);
}

inline void check_square(const NumericMatrix& M, const char* nm) {
  if (M.nrow() != M.ncol()) {
    stop("%s must be square: got %d x %d.", nm,
         (int)M.nrow(), (int)M.ncol());
  }
  check_dim((int)M.nrow(), (std::string("nrow(") + nm + ")").c_str());
}

inline void check_raw(const NumericVector& raw, int n, const char* nm) {
  if ((int)raw.size() != lkj_raw_len(n)) {
    stop("%s must have length n * (n - 1) / 2 = %d for n = %d: got %d.",
         nm, lkj_raw_len(n), n, (int)raw.size());
  }
}

inline void check_group_matrix(const NumericMatrix& M, int n_groups, int n,
                               const char* nm) {
  if (M.nrow() != n_groups || M.ncol() != n) {
    stop("%s must be %d x %d: got %d x %d.", nm, n_groups, n,
         (int)M.nrow(), (int)M.ncol());
  }
}

}  // namespace

// [[Rcpp::export]]
List cpp_test_lkj_build_L(NumericVector raw, int n) {
  check_dim(n, "n");
  check_raw(raw, n, "raw");
  std::vector<double> L_flat((size_t)n * n, 0.0);
  double log_jac = 0.0;
  tulpa::build_L_from_raw(raw.begin(), n, L_flat.data(), &log_jac);

  NumericMatrix L(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      L(i, j) = L_flat[(size_t)i * n + j];
    }
  }
  return List::create(
    _["L"] = L,
    _["log_jac"] = log_jac
  );
}

// [[Rcpp::export]]
NumericVector cpp_test_lkj_raw_from_L(NumericMatrix L) {
  check_square(L, "L");
  int n = L.nrow();
  std::vector<double> L_flat((size_t)n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[(size_t)i * n + j] = L(i, j);
  }
  std::vector<double> raw((size_t)lkj_raw_len(n), 0.0);
  tulpa::raw_from_L(L_flat.data(), n, raw.data());
  return NumericVector(raw.begin(), raw.end());
}

// [[Rcpp::export]]
double cpp_test_lkj_density(NumericMatrix L, double eta) {
  check_square(L, "L");
  int n = L.nrow();
  std::vector<double> L_flat((size_t)n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      L_flat[(size_t)i * n + j] = L(i, j);
    }
  }
  return tulpa::lkj_log_prior_density(L_flat.data(), n, eta);
}

// [[Rcpp::export]]
NumericVector cpp_test_lkj_grad(NumericVector raw, int n, double eta) {
  check_dim(n, "n");
  check_raw(raw, n, "raw");
  std::vector<double> grad((size_t)lkj_raw_len(n), 0.0);
  tulpa::lkj_log_prior_grad_add(raw.begin(), n, eta, grad.data());
  return NumericVector(grad.begin(), grad.end());
}

// [[Rcpp::export]]
NumericMatrix cpp_test_compute_u_eff(NumericMatrix L,
                                     NumericVector sigma,
                                     NumericMatrix z) {
  check_square(L, "L");
  int n = L.nrow();
  int n_groups = z.nrow();
  if (z.ncol() != n) stop("z must have ncol = nrow(L)");
  if ((int)sigma.size() != n) stop("sigma must have length nrow(L)");

  std::vector<double> L_flat((size_t)n * n), z_flat((size_t)n_groups * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[(size_t)i * n + j] = L(i, j);
  }
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) z_flat[(size_t)g * n + c] = z(g, c);
  }
  std::vector<double> u((size_t)n_groups * n, 0.0);
  tulpa::compute_u_eff(L_flat.data(), n, sigma.begin(), z_flat.data(), n_groups, u.data());

  NumericMatrix out(n_groups, n);
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) out(g, c) = u[(size_t)g * n + c];
  }
  return out;
}

// [[Rcpp::export]]
List cpp_test_chol_nc_chain_rule(NumericMatrix L,
                                 NumericVector sigma,
                                 NumericMatrix z,
                                 NumericVector raw,
                                 NumericMatrix u_eff,
                                 NumericMatrix glik) {
  check_square(L, "L");
  int n = L.nrow();
  int n_groups = z.nrow();
  if (z.ncol() != n) stop("z must have ncol = nrow(L)");
  if ((int)sigma.size() != n) stop("sigma must have length nrow(L)");
  check_raw(raw, n, "raw");
  check_group_matrix(u_eff, n_groups, n, "u_eff");
  check_group_matrix(glik, n_groups, n, "glik");

  std::vector<double> L_flat((size_t)n * n), z_flat((size_t)n_groups * n),
                      u_flat((size_t)n_groups * n), glik_flat((size_t)n_groups * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[(size_t)i * n + j] = L(i, j);
  }
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) {
      z_flat[(size_t)g * n + c] = z(g, c);
      u_flat[(size_t)g * n + c] = u_eff(g, c);
      glik_flat[(size_t)g * n + c] = glik(g, c);
    }
  }
  std::vector<double> grad_z((size_t)n_groups * n, 0.0),
                      grad_log_sigma((size_t)n, 0.0),
                      grad_raw((size_t)lkj_raw_len(n), 0.0);
  tulpa::chol_nc_chain_rule_add(L_flat.data(), n, sigma.begin(), z_flat.data(),
                                raw.begin(), u_flat.data(), n_groups, glik_flat.data(),
                                grad_z.data(), grad_log_sigma.data(), grad_raw.data());
  NumericMatrix gz(n_groups, n);
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) gz(g, c) = grad_z[(size_t)g * n + c];
  }
  return List::create(
    _["grad_z"] = gz,
    _["grad_log_sigma"] = NumericVector(grad_log_sigma.begin(), grad_log_sigma.end()),
    _["grad_raw"] = NumericVector(grad_raw.begin(), grad_raw.end())
  );
}

// [[Rcpp::export]]
NumericMatrix cpp_test_correlation_from_L(NumericMatrix L) {
  check_square(L, "L");
  int n = L.nrow();
  std::vector<double> L_flat((size_t)n * n), R_flat((size_t)n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[(size_t)i * n + j] = L(i, j);
  }
  tulpa::correlation_from_L(L_flat.data(), n, R_flat.data());
  NumericMatrix R(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) R(i, j) = R_flat[(size_t)i * n + j];
  }
  return R;
}
