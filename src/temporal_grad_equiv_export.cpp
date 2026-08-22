// temporal_grad_equiv_export.cpp
// Equivalence probe for the analytic temporal gradient kernels.
//
// The RW1 / RW2 / AR1 gradients are reachable in two parameterizations:
// tulpa_tvc::*_grad_w in the precision (tau) one, which is the canonical
// kernel, and tulpa_temporal_grad::*_grad_phi in the variance (sigma2) one,
// which wraps it at tau = 1/sigma2.
//
// Neither header is reached from a compiled translation unit, so without this
// file a wrong wrapper would not even be compiled. Including them here builds
// them, and the export lets test-temporal-grad-equiv.R assert the sigma2
// wrappers agree with the canonical tau kernels at tau = 1/sigma2.
//
// The export also returns the multiscale VALUE functions and the analytic
// d/d log(sigma2) beside them, so the test can central-difference the density
// the kernels claim to differentiate. The cross-parameterization check compares
// two wrappers over one kernel and cannot see a value that disagrees with its
// own gradient; the finite difference is the arbiter that can.
//
// `augment` selects the sum-to-zero augmented intrinsic arms, which is what
// multiscale_temporal_log_lik evaluates on the production path, and it drives
// BOTH the value and the gradient here so the finite difference sees the same
// density the kernel claims (gcol33/tulpa#588). The tau-parameterized
// tulpa_tvc kernels have no augmented form, so the cross-parameterization
// columns are meaningful at augment = false only.

#include <Rcpp.h>
#include <vector>

#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"

// [[Rcpp::export]]
Rcpp::List cpp_test_temporal_grad_equiv(Rcpp::NumericVector w, double sigma2,
                                        double rho, bool augment = false) {
  const int n = w.size();
  // The exact conversion between the two parameterizations, taken here as the
  // caller would take it. A conversion that reproduced a wrapper's own internal
  // arithmetic would make the comparison below true by construction.
  const double tau = 1.0 / sigma2;
  const double* p = w.begin();
  const std::vector<double> wv(w.begin(), w.end());

  std::vector<double> rw1_tau(n, 0.0), rw1_sig(n, 0.0);
  std::vector<double> rw1c_sig(n, 0.0);
  std::vector<double> rw2_tau(n, 0.0), rw2_sig(n, 0.0);
  std::vector<double> ar1_tau(n, 0.0), ar1_sig(n, 0.0);
  std::vector<double> iid_sig(n, 0.0);

  tulpa_tvc::rw1_grad_w(p, n, tau, rw1_tau.data());
  tulpa_temporal_grad::rw1_grad_phi(p, n, sigma2, rw1_sig.data(), augment);
  tulpa_temporal_grad::rw1_cyclic_grad_phi(p, n, sigma2, rw1c_sig.data(),
                                           augment);

  tulpa_tvc::rw2_grad_w(p, n, tau, rw2_tau.data());
  tulpa_temporal_grad::rw2_grad_phi(p, n, sigma2, rw2_sig.data(), augment);

  tulpa_tvc::ar1_grad_w(p, n, tau, rho, ar1_tau.data());
  tulpa_temporal_grad::ar1_grad_phi(p, n, sigma2, rho, ar1_sig.data());

  tulpa_temporal_grad::iid_grad_phi(p, n, sigma2, iid_sig.data());

  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["rw1_tau"]  = Rcpp::NumericVector(rw1_tau.begin(), rw1_tau.end()),
    Rcpp::_["rw1_sig"]  = Rcpp::NumericVector(rw1_sig.begin(), rw1_sig.end()),
    Rcpp::_["rw1c_sig"] = Rcpp::NumericVector(rw1c_sig.begin(), rw1c_sig.end()),
    Rcpp::_["rw2_tau"]  = Rcpp::NumericVector(rw2_tau.begin(), rw2_tau.end()),
    Rcpp::_["rw2_sig"]  = Rcpp::NumericVector(rw2_sig.begin(), rw2_sig.end()),
    Rcpp::_["ar1_tau"]  = Rcpp::NumericVector(ar1_tau.begin(), ar1_tau.end()),
    Rcpp::_["ar1_sig"]  = Rcpp::NumericVector(ar1_sig.begin(), ar1_sig.end()),
    Rcpp::_["iid_sig"]  = Rcpp::NumericVector(iid_sig.begin(), iid_sig.end()),
    Rcpp::_["rho_tau"]  = tulpa_tvc::ar1_grad_logit_rho(p, n, tau, rho),
    Rcpp::_["rho_sig"]  = tulpa_temporal_grad::ar1_grad_logit_rho(p, n, sigma2,
                                                                  rho));

  // The densities the gradients above differentiate, at the same arguments.
  out["rw1_val"]  = tulpa_temporal::rw1_log_lik(wv, sigma2, false, augment);
  out["rw1c_val"] = tulpa_temporal::rw1_log_lik(wv, sigma2, true, augment);
  out["rw2_val"]  = tulpa_temporal::rw2_log_lik(wv, sigma2, false, augment);
  out["ar1_val"]  = tulpa_temporal::ar1_log_lik(wv, sigma2, rho);
  out["iid_val"]  = tulpa_temporal::iid_log_lik(wv, sigma2);

  out["rw1_dls2"]  = tulpa_temporal_grad::rw1_grad_log_sigma2(p, n, sigma2,
                                                              augment);
  out["rw1c_dls2"] = tulpa_temporal_grad::rw1_cyclic_grad_log_sigma2(p, n,
                                                                     sigma2,
                                                                     augment);
  out["rw2_dls2"]  = tulpa_temporal_grad::rw2_grad_log_sigma2(p, n, sigma2,
                                                              augment);
  out["ar1_dls2"]  = tulpa_temporal_grad::ar1_grad_log_sigma2(p, n, sigma2, rho);
  out["iid_dls2"]  = tulpa_temporal_grad::iid_grad_log_sigma2(p, n, sigma2);

  return out;
}

// The stationary-factor slope on its own, so the test can check the piece that
// makes the value and the rho gradient one function (gcol33/tulpa#589) against
// a central difference of the factor itself, rather than only through the
// assembled density.
// [[Rcpp::export]]
double cpp_test_ar1_omr2_slope(double rho) {
  return tulpa_temporal::ar1_one_minus_rho2_slope(rho);
}
