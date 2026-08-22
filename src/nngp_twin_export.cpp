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

#include "linalg_fast.h"   // require_coords_2col
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
  tulpa_linalg::require_coords_2col(coords, "the NNGP twin GP probe");
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

tulpa::SVCData make_svc(const Rcpp::NumericMatrix& coords,
                        const Rcpp::IntegerMatrix& nn_idx,
                        const Rcpp::NumericMatrix& nn_dist,
                        const Rcpp::IntegerVector& nn_order,
                        const Rcpp::IntegerVector& nn_order_inv,
                        int cov_type) {
  const int N = coords.nrow();
  const int nn = nn_idx.ncol();
  tulpa::SVCData sd;
  sd.n_obs = N;
  sd.nn = nn;
  sd.n_svc = 1;
  tulpa_linalg::require_coords_2col(coords, "the NNGP twin SVC probe");
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
  sd.nn_order_inv.assign(nn_order_inv.begin(), nn_order_inv.end());
  sd.cov_type = static_cast<tulpa::CovType>(cov_type);
  return sd;
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

// Non-centered NNGP transform: hand-derived backward vs central differences.
// The transform is w = f(z, sigma2, phi); with a scalar loss L(z, log_sigma2,
// log_phi) = sum(a_i w_i) the analytic reverse pass nngp_nc_backward (called
// with dL/dw = a) returns dL/dz, dL/d(log_sigma2) and dL/d(log_phi) -- the
// exact quantities the arena custom_backward injects on the sampling path.
// grad_z here is the likelihood/transform gradient only (the -z prior is the
// caller's, so it is absent). We compare each against a central difference of
// the forward. The z->w log-Jacobian derivative (grad_log_phi_jac) is not part
// of this loss and is not checked.
// [[Rcpp::export]]
Rcpp::List cpp_test_nngp_nc_grad(Rcpp::NumericVector z,
                                 double log_sigma2, double log_phi,
                                 Rcpp::NumericVector a,
                                 Rcpp::NumericMatrix coords,
                                 Rcpp::IntegerMatrix nn_idx,
                                 Rcpp::NumericMatrix nn_dist,
                                 Rcpp::NumericVector nn_neighbor_dist,
                                 Rcpp::IntegerVector nn_order,
                                 Rcpp::IntegerVector nn_order_inv,
                                 int cov_type, double fd_eps = 1e-6) {
  tulpa::GPData gp = make_gp(coords, nn_idx, nn_dist, nn_neighbor_dist,
                             nn_order, nn_order_inv, cov_type);
  const int N = gp.n_obs;
  std::vector<double> z_vec(z.begin(), z.end());
  std::vector<double> a_vec(a.begin(), a.end());

  const tulpa_gp::NNGPNCView gp_view = tulpa_gp::make_gp_nc_view(gp);

  auto forward_loss = [&](const std::vector<double>& zz,
                          double lsig2, double lphi) -> double {
    tulpa_gp::NNGPNCWorkspace ws;
    tulpa_gp::nngp_nc_forward(zz.data(), std::exp(lsig2), std::exp(lphi), gp_view, ws);
    double L = 0.0;
    for (int i = 0; i < N; i++) L += a_vec[i] * ws.w[i];
    return L;
  };

  // Analytic reverse pass at (z, log_sigma2, log_phi).
  tulpa_gp::NNGPNCWorkspace ws;
  tulpa_gp::nngp_nc_forward(z_vec.data(), std::exp(log_sigma2), std::exp(log_phi),
                            gp_view, ws);
  std::vector<double> grad_z(N, 0.0);
  double g_log_sigma2 = 0.0, g_log_phi = 0.0, g_log_phi_jac = 0.0;
  tulpa_gp::nngp_nc_backward(z_vec.data(), std::exp(log_sigma2), std::exp(log_phi),
                             gp_view, ws, a_vec.data(), grad_z.data(),
                             g_log_sigma2, g_log_phi, g_log_phi_jac);

  // Central differences.
  std::vector<double> grad_z_fd(N, 0.0);
  std::vector<double> z_pt = z_vec;
  for (int i = 0; i < N; i++) {
    z_pt[i] += fd_eps; double fp = forward_loss(z_pt, log_sigma2, log_phi);
    z_pt[i] -= 2 * fd_eps; double fm = forward_loss(z_pt, log_sigma2, log_phi);
    z_pt[i] += fd_eps;
    grad_z_fd[i] = (fp - fm) / (2.0 * fd_eps);
  }
  double sp = forward_loss(z_vec, log_sigma2 + fd_eps, log_phi);
  double sm = forward_loss(z_vec, log_sigma2 - fd_eps, log_phi);
  double g_log_sigma2_fd = (sp - sm) / (2.0 * fd_eps);
  double pp = forward_loss(z_vec, log_sigma2, log_phi + fd_eps);
  double pm = forward_loss(z_vec, log_sigma2, log_phi - fd_eps);
  double g_log_phi_fd = (pp - pm) / (2.0 * fd_eps);

  return Rcpp::List::create(
    Rcpp::_["grad_z"]            = Rcpp::NumericVector(grad_z.begin(), grad_z.end()),
    Rcpp::_["grad_z_fd"]         = Rcpp::NumericVector(grad_z_fd.begin(), grad_z_fd.end()),
    Rcpp::_["grad_log_sigma2"]   = g_log_sigma2,
    Rcpp::_["grad_log_sigma2_fd"] = g_log_sigma2_fd,
    Rcpp::_["grad_log_phi"]      = g_log_phi,
    Rcpp::_["grad_log_phi_fd"]   = g_log_phi_fd);
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
  tulpa_linalg::require_coords_2col(coords, "the NNGP twin SVC probe");
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

// The analytic-vs-finite-difference probe the non-centered NNGP exports below
// share. Once a caller has built its NNGPNCView -- from SVCData on one side, a
// MultiscaleGPData scale on the other -- the comparison is identical: run the
// forward transform, take the hand-derived backward against the loss weights
// `a`, then central-difference the same forward loss in z, log sigma2 and
// log phi. `view` borrows from the caller's data, which must outlive this call.
static Rcpp::List nngp_nc_grad_probe(const tulpa_gp::NNGPNCView& view, int N,
                                     const Rcpp::NumericVector& z,
                                     const Rcpp::NumericVector& a,
                                     double log_sigma2, double log_phi,
                                     double fd_eps) {
  std::vector<double> z_vec(z.begin(), z.end());
  std::vector<double> a_vec(a.begin(), a.end());

  auto forward_loss = [&](const std::vector<double>& zz,
                          double lsig2, double lphi) -> double {
    tulpa_gp::NNGPNCWorkspace ws;
    tulpa_gp::nngp_nc_forward(zz.data(), std::exp(lsig2), std::exp(lphi), view, ws);
    double L = 0.0;
    for (int i = 0; i < N; i++) L += a_vec[i] * ws.w[i];
    return L;
  };

  tulpa_gp::NNGPNCWorkspace ws;
  tulpa_gp::nngp_nc_forward(z_vec.data(), std::exp(log_sigma2), std::exp(log_phi),
                            view, ws);
  std::vector<double> grad_z(N, 0.0);
  double g_log_sigma2 = 0.0, g_log_phi = 0.0, g_log_phi_jac = 0.0;
  tulpa_gp::nngp_nc_backward(z_vec.data(), std::exp(log_sigma2), std::exp(log_phi),
                             view, ws, a_vec.data(), grad_z.data(),
                             g_log_sigma2, g_log_phi, g_log_phi_jac);

  std::vector<double> grad_z_fd(N, 0.0);
  std::vector<double> z_pt = z_vec;
  for (int i = 0; i < N; i++) {
    z_pt[i] += fd_eps; double fp = forward_loss(z_pt, log_sigma2, log_phi);
    z_pt[i] -= 2 * fd_eps; double fm = forward_loss(z_pt, log_sigma2, log_phi);
    z_pt[i] += fd_eps;
    grad_z_fd[i] = (fp - fm) / (2.0 * fd_eps);
  }
  double sp = forward_loss(z_vec, log_sigma2 + fd_eps, log_phi);
  double sm = forward_loss(z_vec, log_sigma2 - fd_eps, log_phi);
  double g_log_sigma2_fd = (sp - sm) / (2.0 * fd_eps);
  double pp = forward_loss(z_vec, log_sigma2, log_phi + fd_eps);
  double pm = forward_loss(z_vec, log_sigma2, log_phi - fd_eps);
  double g_log_phi_fd = (pp - pm) / (2.0 * fd_eps);

  return Rcpp::List::create(
    Rcpp::_["grad_z"]            = Rcpp::NumericVector(grad_z.begin(), grad_z.end()),
    Rcpp::_["grad_z_fd"]         = Rcpp::NumericVector(grad_z_fd.begin(), grad_z_fd.end()),
    Rcpp::_["grad_log_sigma2"]   = g_log_sigma2,
    Rcpp::_["grad_log_sigma2_fd"] = g_log_sigma2_fd,
    Rcpp::_["grad_log_phi"]      = g_log_phi,
    Rcpp::_["grad_log_phi_fd"]   = g_log_phi_fd);
}

// Non-centered NNGP transform on the SVC neighbour topology (coords-fallback
// pair_dist branch, no cached nn_neighbor_dist): hand-derived backward vs
// central differences, same scalar-loss construction as cpp_test_nngp_nc_grad.
// Exercises the OTHER branch of NNGPNCView::pair_dist -- cpp_test_nngp_nc_grad
// only exercises the cached-table branch.
// [[Rcpp::export]]
Rcpp::List cpp_test_svc_nngp_nc_grad(Rcpp::NumericVector z,
                                     double log_sigma2, double log_phi,
                                     Rcpp::NumericVector a,
                                     Rcpp::NumericMatrix coords,
                                     Rcpp::IntegerMatrix nn_idx,
                                     Rcpp::NumericMatrix nn_dist,
                                     Rcpp::IntegerVector nn_order,
                                     Rcpp::IntegerVector nn_order_inv,
                                     int cov_type, double fd_eps = 1e-6) {
  tulpa::SVCData sd = make_svc(coords, nn_idx, nn_dist, nn_order, nn_order_inv,
                               cov_type);
  const tulpa_gp::NNGPNCView view = tulpa_gp::make_svc_nc_view(sd);
  return nngp_nc_grad_probe(view, sd.n_obs, z, a, log_sigma2, log_phi, fd_eps);
}

// Non-centered NNGP transform on one scale of a MultiscaleGPData (cached
// nn_neighbor_dist, the same fast path GPData uses): hand-derived backward vs
// central differences, same construction as cpp_test_nngp_nc_grad. `scale`
// selects "local" or "regional" -- both share the same transform code
// (make_msgp_nc_view_{local,regional} only pick different MultiscaleGPData
// fields), so one export covers both by argument rather than duplicating the
// function.
// [[Rcpp::export]]
Rcpp::List cpp_test_msgp_nngp_nc_grad(Rcpp::NumericVector z,
                                      double log_sigma2, double log_phi,
                                      Rcpp::NumericVector a,
                                      Rcpp::NumericMatrix coords,
                                      Rcpp::IntegerMatrix nn_idx,
                                      Rcpp::NumericMatrix nn_dist,
                                      Rcpp::NumericVector nn_neighbor_dist,
                                      Rcpp::IntegerVector nn_order,
                                      Rcpp::IntegerVector nn_order_inv,
                                      int cov_type, std::string scale,
                                      double fd_eps = 1e-6) {
  const int N = coords.nrow();
  const int nn = nn_idx.ncol();
  tulpa::MultiscaleGPData ms;
  ms.n_obs = N;
  tulpa_linalg::require_coords_2col(coords, "the NNGP twin multiscale probe");
  ms.coords.resize(N * 2);
  for (int i = 0; i < N; i++) {
    ms.coords[i * 2 + 0] = coords(i, 0);
    ms.coords[i * 2 + 1] = coords(i, 1);
  }
  ms.cov_type = static_cast<tulpa::CovType>(cov_type);

  auto fill_idx_dist = [&](std::vector<int>& idx_out, std::vector<double>& dist_out) {
    idx_out.resize(N * nn);
    dist_out.resize(N * nn);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < nn; j++) {
        idx_out[i * nn + j] = nn_idx(i, j);
        dist_out[i * nn + j] = nn_dist(i, j);
      }
    }
  };

  tulpa_gp::NNGPNCView view;
  if (scale == "local") {
    ms.nn_local = nn;
    fill_idx_dist(ms.nn_idx_local, ms.nn_dist_local);
    ms.nn_neighbor_dist_local.assign(nn_neighbor_dist.begin(), nn_neighbor_dist.end());
    ms.nn_order_local.assign(nn_order.begin(), nn_order.end());
    ms.nn_order_inv_local.assign(nn_order_inv.begin(), nn_order_inv.end());
    view = tulpa_gp::make_msgp_nc_view_local(ms);
  } else if (scale == "regional") {
    ms.nn_regional = nn;
    fill_idx_dist(ms.nn_idx_regional, ms.nn_dist_regional);
    ms.nn_neighbor_dist_regional.assign(nn_neighbor_dist.begin(), nn_neighbor_dist.end());
    ms.nn_order_regional.assign(nn_order.begin(), nn_order.end());
    ms.nn_order_inv_regional.assign(nn_order_inv.begin(), nn_order_inv.end());
    view = tulpa_gp::make_msgp_nc_view_regional(ms);
  } else {
    Rcpp::stop("cpp_test_msgp_nngp_nc_grad: scale must be \"local\" or \"regional\".");
  }

  return nngp_nc_grad_probe(view, N, z, a, log_sigma2, log_phi, fd_eps);
}
