// test_lkj_chol.cpp
// Rcpp wrappers exposing the lkj_chol_helpers.h functions to R for unit
// testing. The helpers themselves are header-only and namespace-private.

#include <Rcpp.h>
#include <vector>
#include "lkj_chol_helpers.h"

using namespace Rcpp;

// [[Rcpp::export]]
List cpp_test_lkj_build_L(NumericVector raw, int n) {
  std::vector<double> L_flat(n * n, 0.0);
  double log_jac_tanh = 0.0;
  bool ok = tulpa::build_L_from_raw(raw.begin(), n, L_flat.data(), &log_jac_tanh);

  NumericMatrix L(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      L(i, j) = L_flat[i * n + j];
    }
  }
  return List::create(
    _["L"] = L,
    _["log_jac_tanh"] = log_jac_tanh,
    _["ok"] = ok
  );
}

// [[Rcpp::export]]
double cpp_test_lkj_density(NumericMatrix L, double eta) {
  int n = L.nrow();
  std::vector<double> L_flat(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      L_flat[i * n + j] = L(i, j);
    }
  }
  return tulpa::lkj_log_prior_density(L_flat.data(), n, eta);
}

// [[Rcpp::export]]
NumericVector cpp_test_lkj_grad(NumericVector raw, int n, double eta) {
  std::vector<double> L_flat(n * n, 0.0);
  if (!tulpa::build_L_from_raw(raw.begin(), n, L_flat.data())) {
    stop("build_L_from_raw failed: row norm constraint violated");
  }
  std::vector<double> grad(raw.size(), 0.0);
  tulpa::lkj_log_prior_grad_add(raw.begin(), L_flat.data(), n, eta, grad.data());
  return NumericVector(grad.begin(), grad.end());
}

// [[Rcpp::export]]
NumericMatrix cpp_test_compute_u_eff(NumericMatrix L,
                                     NumericVector sigma,
                                     NumericMatrix z) {
  int n = L.nrow();
  int n_groups = z.nrow();
  if (z.ncol() != n) stop("z must have ncol = nrow(L)");
  if ((int)sigma.size() != n) stop("sigma must have length nrow(L)");

  std::vector<double> L_flat(n * n), z_flat(n_groups * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[i * n + j] = L(i, j);
  }
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) z_flat[g * n + c] = z(g, c);
  }
  std::vector<double> u(n_groups * n, 0.0);
  tulpa::compute_u_eff(L_flat.data(), n, sigma.begin(), z_flat.data(), n_groups, u.data());

  NumericMatrix out(n_groups, n);
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) out(g, c) = u[g * n + c];
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
  int n = L.nrow();
  int n_groups = z.nrow();
  std::vector<double> L_flat(n * n), z_flat(n_groups * n), u_flat(n_groups * n), glik_flat(n_groups * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[i * n + j] = L(i, j);
  }
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) {
      z_flat[g * n + c] = z(g, c);
      u_flat[g * n + c] = u_eff(g, c);
      glik_flat[g * n + c] = glik(g, c);
    }
  }
  std::vector<double> grad_z(n_groups * n, 0.0),
                      grad_log_sigma(n, 0.0),
                      grad_raw(raw.size(), 0.0);
  tulpa::chol_nc_chain_rule_add(L_flat.data(), n, sigma.begin(), z_flat.data(),
                                raw.begin(), u_flat.data(), n_groups, glik_flat.data(),
                                grad_z.data(), grad_log_sigma.data(), grad_raw.data());
  NumericMatrix gz(n_groups, n);
  for (int g = 0; g < n_groups; g++) {
    for (int c = 0; c < n; c++) gz(g, c) = grad_z[g * n + c];
  }
  return List::create(
    _["grad_z"] = gz,
    _["grad_log_sigma"] = NumericVector(grad_log_sigma.begin(), grad_log_sigma.end()),
    _["grad_raw"] = NumericVector(grad_raw.begin(), grad_raw.end())
  );
}

// [[Rcpp::export]]
NumericMatrix cpp_test_correlation_from_L(NumericMatrix L) {
  int n = L.nrow();
  std::vector<double> L_flat(n * n), R_flat(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) L_flat[i * n + j] = L(i, j);
  }
  tulpa::correlation_from_L(L_flat.data(), n, R_flat.data());
  NumericMatrix R(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) R(i, j) = R_flat[i * n + j];
  }
  return R;
}
