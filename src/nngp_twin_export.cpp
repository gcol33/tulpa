// nngp_twin_export.cpp
// Twin-equivalence probes for the NNGP conditional kernels (A3).
//
// Each NNGP log-likelihood exists twice: a hand-written `double` version and a
// templated autodiff one. They are the same function -- the AD copy instantiated
// at T = double MUST return the hand-written copy's value. Where it does not,
// the value path and the gradient path describe different models. That is not
// hypothetical: the GP AD copy added its jitter only to an already-degenerate
// pivot while the GP double copy (whose value the analytic gradients are
// finite-differenced from) added it to every diagonal, so on well-conditioned
// input the two evaluated different log-densities. hmc_gp_autodiff.h carried a
// "known heisenbug with autodiff - use numerical gradients for GP" note.
//
// These exports evaluate both twins at identical inputs so test-nngp-twin.R can
// assert they agree.

#include <Rcpp.h>
#include <vector>

// hmc_gp.h is the umbrella: hmc_gp_log_lik.h / hmc_gp_gradients.h are
// order-dependent fragments it stitches in, and are not self-contained.
#include "hmc_gp.h"
#include "hmc_gp_autodiff.h"
#include "hmc_svc.h"
#include "hmc_svc_autodiff.h"

namespace {

tulpa::GPData make_gp(const Rcpp::NumericMatrix& coords,
                      const Rcpp::IntegerMatrix& nn_idx,
                      const Rcpp::NumericMatrix& nn_dist,
                      const Rcpp::NumericVector& nn_neighbor_dist,
                      const Rcpp::IntegerVector& nn_order,
                      const Rcpp::IntegerVector& nn_order_inv,
                      int cov_type) {
  const int N = coords.nrow();
  const int nn = nn_idx.ncol();
  tulpa::GPData gp;
  gp.n_obs = N;
  gp.nn = nn;
  gp.coords.resize(N * 2);
  for (int i = 0; i < N; i++) {
    gp.coords[i * 2 + 0] = coords(i, 0);
    gp.coords[i * 2 + 1] = coords(i, 1);
  }
  gp.nn_idx.resize(N * nn);
  gp.nn_dist.resize(N * nn);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < nn; j++) {
      gp.nn_idx[i * nn + j] = nn_idx(i, j);
      gp.nn_dist[i * nn + j] = nn_dist(i, j);
    }
  }
  gp.nn_neighbor_dist.assign(nn_neighbor_dist.begin(), nn_neighbor_dist.end());
  gp.nn_order.assign(nn_order.begin(), nn_order.end());
  gp.nn_order_inv.assign(nn_order_inv.begin(), nn_order_inv.end());
  gp.cov_type = static_cast<tulpa::CovType>(cov_type);
  gp.solver_config.solver = tulpa_gp::parse_gp_solver("chol");
  gp.solver_config.n_obs = N;
  return gp;
}

}  // namespace

// GP NNGP log-likelihood from both twins at the same inputs.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_gp_nngp_twins(Rcpp::NumericVector w, double sigma2,
                                           double phi,
                                           Rcpp::NumericMatrix coords,
                                           Rcpp::IntegerMatrix nn_idx,
                                           Rcpp::NumericMatrix nn_dist,
                                           Rcpp::NumericVector nn_neighbor_dist,
                                           Rcpp::IntegerVector nn_order,
                                           Rcpp::IntegerVector nn_order_inv,
                                           int cov_type) {
  tulpa::GPData gp = make_gp(coords, nn_idx, nn_dist, nn_neighbor_dist,
                             nn_order, nn_order_inv, cov_type);
  std::vector<double> w_vec(w.begin(), w.end());
  const double ll_double = tulpa_gp::gp_nngp_log_lik(w_vec, sigma2, phi, gp);
  const double ll_ad = tulpa_gp::gp_nngp_log_lik_t<double>(w_vec, sigma2, phi, gp);
  return Rcpp::NumericVector::create(Rcpp::_["dbl"] = ll_double,
                                     Rcpp::_["ad"]  = ll_ad);
}

// SVC NNGP log-likelihood from both twins at the same inputs.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_svc_nngp_twins(Rcpp::NumericVector w, double sigma2,
                                            double phi,
                                            Rcpp::NumericMatrix coords,
                                            Rcpp::IntegerMatrix nn_idx,
                                            Rcpp::NumericMatrix nn_dist,
                                            Rcpp::IntegerVector nn_order,
                                            int cov_type) {
  const int N = coords.nrow();
  const int nn = nn_idx.ncol();
  tulpa::SVCData sd;
  sd.n_obs = N;
  sd.nn = nn;
  sd.n_svc = 1;
  sd.coords.resize(N * 2);
  for (int i = 0; i < N; i++) {
    sd.coords[i * 2 + 0] = coords(i, 0);
    sd.coords[i * 2 + 1] = coords(i, 1);
  }
  sd.nn_idx.resize(N * nn);
  sd.nn_dist.resize(N * nn);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < nn; j++) {
      sd.nn_idx[i * nn + j] = nn_idx(i, j);
      sd.nn_dist[i * nn + j] = nn_dist(i, j);
    }
  }
  sd.nn_order.assign(nn_order.begin(), nn_order.end());
  sd.cov_type = static_cast<tulpa::CovType>(cov_type);

  std::vector<double> w_vec(w.begin(), w.end());
  const double ll_double = tulpa_svc::nngp_log_lik(w_vec, sigma2, phi, sd);
  const double ll_ad = tulpa_svc_ad::nngp_log_lik<double>(w_vec, sigma2, phi, sd);
  return Rcpp::NumericVector::create(Rcpp::_["dbl"] = ll_double,
                                     Rcpp::_["ad"]  = ll_ad);
}
