// temporal_grad_equiv_export.cpp
// Equivalence probe for the analytic temporal gradient kernels (gcol33/tulpa#142 A7).
//
// The RW1 / RW2 / AR1 gradients were written twice: tulpa_tvc::*_grad_w in the
// precision (tau) parameterization, and tulpa_temporal_grad::*_grad_phi in the
// variance (sigma2) one. Only RW1 had been unified into a wrapper; the RW2 and
// AR1 copies were independent transcriptions that had already drifted on their
// guards (the sigma2 copies guarded n < 3 / n == 1 and the tau copies did not,
// which left two out-of-bounds accesses in the tau side).
//
// Neither header is reached from a compiled translation unit today, so nothing
// would catch a wrong wrapper -- the code would not even be compiled. Including
// them here builds them, and the export lets test-temporal-grad-equiv.R assert
// the sigma2 wrappers agree with the canonical tau kernels at tau = 1/sigma2.

#include <Rcpp.h>
#include <vector>

#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"

// [[Rcpp::export]]
Rcpp::List cpp_test_temporal_grad_equiv(Rcpp::NumericVector w, double sigma2,
                                        double rho) {
  const int n = w.size();
  const double tau = 1.0 / (sigma2 + 1e-10);
  const double* p = w.begin();

  std::vector<double> rw1_tau(n, 0.0), rw1_sig(n, 0.0);
  std::vector<double> rw2_tau(n, 0.0), rw2_sig(n, 0.0);
  std::vector<double> ar1_tau(n, 0.0), ar1_sig(n, 0.0);

  tulpa_tvc::rw1_grad_w(p, n, tau, rw1_tau.data());
  tulpa_temporal_grad::rw1_grad_phi(p, n, sigma2, rw1_sig.data());

  tulpa_tvc::rw2_grad_w(p, n, tau, rw2_tau.data());
  tulpa_temporal_grad::rw2_grad_phi(p, n, sigma2, rw2_sig.data());

  tulpa_tvc::ar1_grad_w(p, n, tau, rho, ar1_tau.data());
  tulpa_temporal_grad::ar1_grad_phi(p, n, sigma2, rho, ar1_sig.data());

  return Rcpp::List::create(
    Rcpp::_["rw1_tau"]  = Rcpp::NumericVector(rw1_tau.begin(), rw1_tau.end()),
    Rcpp::_["rw1_sig"]  = Rcpp::NumericVector(rw1_sig.begin(), rw1_sig.end()),
    Rcpp::_["rw2_tau"]  = Rcpp::NumericVector(rw2_tau.begin(), rw2_tau.end()),
    Rcpp::_["rw2_sig"]  = Rcpp::NumericVector(rw2_sig.begin(), rw2_sig.end()),
    Rcpp::_["ar1_tau"]  = Rcpp::NumericVector(ar1_tau.begin(), ar1_tau.end()),
    Rcpp::_["ar1_sig"]  = Rcpp::NumericVector(ar1_sig.begin(), ar1_sig.end()),
    Rcpp::_["rho_tau"]  = tulpa_tvc::ar1_grad_logit_rho(p, n, tau, rho),
    Rcpp::_["rho_sig"]  = tulpa_temporal_grad::ar1_grad_logit_rho(p, n, sigma2,
                                                                  rho));
}
