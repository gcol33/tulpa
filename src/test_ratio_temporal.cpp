#include <RcppEigen.h>
#include "hmc_temporal_multiscale.h"
#include "hmc_tvc.h"
using namespace Rcpp;
static tulpa::TemporalType probe_temporal_type(const std::string& name) {
  if(name=="none") return tulpa::TemporalType::NONE;
  if(name=="rw1") return tulpa::TemporalType::RW1;
  if(name=="rw2") return tulpa::TemporalType::RW2;
  if(name=="ar1") return tulpa::TemporalType::AR1;
  if(name=="iid") return tulpa::TemporalType::IID;
  Rcpp::stop("Unknown temporal structure '%s'",name.c_str());
}
// [[Rcpp::export]]
double cpp_test_rw1_quadratic_form(NumericVector phi, bool cyclic) {
  return tulpa_temporal::rw1_quadratic_form(phi.begin(), phi.size(), cyclic);
}

// [[Rcpp::export]]
double cpp_test_rw2_quadratic_form(NumericVector phi, bool cyclic) {
  return tulpa_temporal::rw2_quadratic_form(phi.begin(), phi.size(), cyclic);
}

// [[Rcpp::export]]
double cpp_test_ar1_log_density(NumericVector phi, double rho, double tau) {
  return tulpa_temporal::ar1_log_density(phi.begin(), phi.size(), rho, tau);
}

// [[Rcpp::export]]
double cpp_test_temporal_log_prior(
    NumericVector phi,
    std::string type_str,
    double tau,
    double rho,
    bool cyclic,
    int n_groups = 1,
    bool augment = false,
    bool tvc = false
) {
  if (tvc) {
    tulpa_tvc::TVCData data;
    data.n_times = phi.size() / n_groups;
    data.n_tvc = 1; data.n_groups = n_groups;
    data.structure = probe_temporal_type(type_str);
    data.cyclic = cyclic;
    return tulpa_tvc::tvc_log_prior(std::vector<double>(phi.begin(),phi.end()),
                                     data,std::vector<double>{tau},std::vector<double>{rho});
  }
  const auto type=probe_temporal_type(type_str);
  const int n=phi.size()/n_groups;
  if(augment) {
    tulpa::TVCData data;data.n_times=n;data.n_tvc=1;data.n_groups=n_groups;
    data.structure=type;data.cyclic=cyclic;
    return tulpa_tvc::tvc_log_prior(std::vector<double>(phi.begin(),phi.end()),
      data,std::vector<double>{tau},std::vector<double>{rho});
  }
  double value=0;
  for(int g=0;g<n_groups;++g) value+=tulpa_temporal::log_prior_temporal(
    phi.begin()+g*n,n,type,tau,rho,cyclic);
  return value;
}

// [[Rcpp::export]]
double cpp_test_ms_rw_log_lik(NumericVector phi, double sigma2, int order,
                              bool cyclic, bool augment) {
  std::vector<double> v(phi.begin(), phi.end());
  return order == 2
      ? tulpa_temporal::rw2_log_lik(v, sigma2, cyclic, augment)
      : tulpa_temporal::rw1_log_lik(v, sigma2, cyclic, augment);
}

