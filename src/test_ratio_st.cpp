#include <RcppEigen.h>
#include "st_gp_kernel.h"
#include <tulpa/st_data.h>
using namespace Rcpp;
static tulpa::CovType probe_covariance(const std::string& name) {
  if(name=="exponential") return tulpa::CovType::EXPONENTIAL;
  if(name=="matern") return tulpa::CovType::MATERN32;
  if(name=="gaussian") return tulpa::CovType::GAUSSIAN;
  if(name=="spherical") return tulpa::CovType::SPHERICAL;
  Rcpp::stop("Unknown covariance");
}
static tulpa::SpatiotemporalData test_st_gp_structure(
    int n, NumericMatrix coords, NumericVector time_values, int nn,
    IntegerVector nn_idx, NumericVector nn_dist_space, NumericVector nn_dist_time,
    IntegerVector nn_order, const std::string& nonsep_type,
    const std::string& cov_space, const std::string& cov_time) {
  tulpa::SpatiotemporalData st;
  st.type = tulpa::STType::NONSEP_GP;
  st.n_params = n;
  st.nn = nn;
  st.coords.resize(2 * (size_t)n);
  for (int i = 0; i < n; i++) {
    st.coords[(size_t)i * 2 + 0] = coords(i, 0);
    st.coords[(size_t)i * 2 + 1] = coords(i, 1);
  }
  st.time_values = Rcpp::as<std::vector<double>>(time_values);
  st.nn_idx = Rcpp::as<std::vector<int>>(nn_idx);
  st.nn_dist_space = Rcpp::as<std::vector<double>>(nn_dist_space);
  st.nn_dist_time = Rcpp::as<std::vector<double>>(nn_dist_time);
  std::vector<int> ord = Rcpp::as<std::vector<int>>(nn_order);
  st.nn_order.resize(ord.size());
  for (size_t i = 0; i < ord.size(); i++) st.nn_order[i] = ord[i] - 1;
  st.nn_order_inv.assign(ord.size(), 0);
  for (size_t i = 0; i < st.nn_order.size(); i++)
    st.nn_order_inv[st.nn_order[i]] = (int)i;

  if (nonsep_type == "product") {
    st.nonsep_type = tulpa::NonsepType::PRODUCT;
  } else if (nonsep_type == "gneiting") {
    st.nonsep_type = tulpa::NonsepType::GNEITING;
  } else {
    Rcpp::stop("unknown nonsep_type '%s'", nonsep_type.c_str());
  }
  st.cov_space = probe_covariance(cov_space);
  st.cov_time = probe_covariance(cov_time);
  return st;
}

// [[Rcpp::export]]
double cpp_test_st_gp_log_lik(
    NumericVector w,
    double sigma2,
    double phi_space,
    double phi_time,
    NumericMatrix coords,        // n x 2, by field index
    NumericVector time_values,   // n, by field index
    int nn,
    IntegerVector nn_idx,        // n x nn, row-major, 1-based within ordering
    NumericVector nn_dist_space,
    NumericVector nn_dist_time,
    IntegerVector nn_order,      // 1-based from R, stored 0-based
    std::string nonsep_type = "product",
    std::string cov_space = "exponential",
    std::string cov_time = "exponential"
) {
  const tulpa::SpatiotemporalData st = test_st_gp_structure(
      w.size(), coords, time_values, nn, nn_idx, nn_dist_space, nn_dist_time,
      nn_order, nonsep_type, cov_space, cov_time);
  return tulpa_st_gp::st_gp_nngp_log_lik(
      w.begin(), sigma2, phi_space, phi_time, st);
}

// The margin penalty's normalizer on an S x T grid (field index s * T + t),
// with its derivatives in log tau and the two log ranges.
// [[Rcpp::export]]
Rcpp::List cpp_test_st_gp_margin_normalizer(
    double tau,
    double phi_space,
    double phi_time,
    int n_spatial,
    int n_times,
    NumericMatrix coords,
    NumericVector time_values,
    int nn,
    IntegerVector nn_idx,
    NumericVector nn_dist_space,
    NumericVector nn_dist_time,
    IntegerVector nn_order,
    std::string nonsep_type = "gneiting"
) {
  tulpa::SpatiotemporalData st = test_st_gp_structure(
      n_spatial * n_times, coords, time_values, nn, nn_idx, nn_dist_space,
      nn_dist_time, nn_order, nonsep_type, "exponential", "exponential");
  st.n_spatial = n_spatial;
  st.n_times = n_times;
  const tulpa_st_gp::StGpMarginNormalizer nz =
      tulpa_st_gp::st_gp_margin_normalizer_grad(tau, phi_space, phi_time, st);
  if (!nz.ok) Rcpp::stop("the normalizer could not be formed");
  return Rcpp::List::create(
      Rcpp::Named("value") = nz.value,
      Rcpp::Named("grad") = Rcpp::NumericVector::create(
          nz.log_tau, nz.log_phi_space, nz.log_phi_time));
}

