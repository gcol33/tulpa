#include <RcppEigen.h>
#include "test_prior_model.h"
#include "sampler_field_store.h"
#include "tulpa_priors_st.h"

namespace {
void temporal_gp_probe_data(tulpa::ModelData& data,const Rcpp::List& p) {
  data.temporal_type=tulpa::TemporalType::GP;data.has_temporal_gp=true;
  data.n_times=Rcpp::as<int>(p["n_times"]);
  data.n_temporal_groups=Rcpp::as<int>(p["n_groups"]);
  data.n_temporal_params=data.n_times*data.n_temporal_groups;
  auto& gp=data.temporal_gp_data;
  gp.n_obs=data.n_times;gp.n_groups=data.n_temporal_groups;
  gp.time_values=Rcpp::as<std::vector<double>>(p["time_values"]);
  const auto cov=Rcpp::as<std::string>(p["cov_type"]);
  if(cov=="exponential") gp.cov_type=tulpa::TemporalCovType::EXPONENTIAL;
  else if(cov=="matern") gp.cov_type=tulpa::TemporalCovType::MATERN;
  else if(cov=="gaussian") gp.cov_type=tulpa::TemporalCovType::GAUSSIAN;
  else if(cov=="periodic") gp.cov_type=tulpa::TemporalCovType::PERIODIC;
  else Rcpp::stop("Unknown temporal covariance");
  gp.nu=Rcpp::as<double>(p["nu"]);gp.period=Rcpp::as<double>(p["period"]);
  data.temporal_gp_relative_nugget=1e-6;
  data.temporal_gp_parameterization=Rcpp::as<std::string>(p["gp_parameterization"])=="centered" ? 0:1;
  data.temporal_gp_phi_prior_lower=Rcpp::as<double>(p["gp_phi_prior_lower"]);
  data.temporal_gp_phi_prior_upper=Rcpp::as<double>(p["gp_phi_prior_upper"]);
  data.temporal_gp_sigma2_prior_U=Rcpp::as<double>(p["gp_sigma2_prior_U"]);
  data.temporal_gp_sigma2_prior_alpha=Rcpp::as<double>(p["gp_sigma2_prior_alpha"]);
  if(cov=="gaussian") {
    data.temporal_gp_phi_prior_lower*=std::sqrt(2.0);
    data.temporal_gp_phi_prior_upper*=std::sqrt(2.0);
  }
}
}

// [[Rcpp::export]]
Rcpp::List cpp_temporal_gp_log_prior(Rcpp::NumericVector x,double log_sigma2,
    double logit_phi,const Rcpp::List& temporal_params) {
  tulpa::ModelData data;tulpa::LikelihoodSpec spec;
  tulpa_probe::gaussian_prior(data,spec,0);
  temporal_gp_probe_data(data,temporal_params);
  const auto layout=tulpa_hmc::compute_param_layout(data);
  if(x.size()!=layout.temporal_end-layout.temporal_start)
    Rcpp::stop("Temporal field dimensions do not match its time grid");
  std::vector<double> q(layout.total_params,0),stored(layout.total_params);
  q[layout.log_sigma2_temporal_gp_idx]=log_sigma2;
  q[layout.logit_phi_temporal_gp_idx]=logit_phi;
  std::copy(x.begin(),x.end(),q.begin()+layout.temporal_start);
  const double prior=tulpa_hmc::compute_log_prior(q,data,layout);
  std::copy(q.begin(),q.end(),stored.begin());
  tulpa::store_field_draw(stored.data(),q,data,layout);
  return Rcpp::List::create(Rcpp::Named("log_prior")=prior,
    Rcpp::Named("f")=Rcpp::NumericVector(stored.begin()+layout.temporal_start,
      stored.begin()+layout.temporal_end));
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_temporal_gp_field(const Rcpp::NumericMatrix& x,
    const Rcpp::NumericVector& log_sigma2,const Rcpp::NumericVector& logit_phi,
    const Rcpp::List& temporal_params) {
  if(log_sigma2.size()!=x.nrow() || logit_phi.size()!=x.nrow())
    Rcpp::stop("One hyperparameter pair is required per draw");
  Rcpp::NumericMatrix out(x.nrow(),x.ncol());
  for(int i=0;i<x.nrow();++i) {
    auto result=cpp_temporal_gp_log_prior(x(i,Rcpp::_),log_sigma2[i],logit_phi[i],temporal_params);
    Rcpp::NumericVector field=result["f"];
    out(i,Rcpp::_)=field;
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_st_ar1_constrained_prior(double rho,int n_spatial,
    int n_times,std::string type="II") {
  tulpa::ModelData data;
  auto& st=data.spatiotemporal_data;
  if(type=="II") st.type=tulpa::STType::TYPE_II;
  else if(type=="IV") st.type=tulpa::STType::TYPE_IV;
  else Rcpp::stop("Unknown interaction type");
  st.temporal_type=tulpa::TemporalType::AR1;
  st.n_spatial=n_spatial;st.n_times=n_times;st.n_params=n_spatial*n_times;
  st.adj_row_ptr.push_back(0);
  for(int s=0;s<n_spatial;++s) {
    if(s>0) st.adj_col_idx.push_back(s);
    if(s+1<n_spatial) st.adj_col_idx.push_back(s+2);
    st.adj_row_ptr.push_back(st.adj_col_idx.size());
  }
  tulpa::setup_st_spatial_partition(st);
  std::vector<fwd::Dual> field(st.n_params,fwd::Dual(0));
  const auto value=tulpa::priors::st_constrained_density(field,data,fwd::Dual(2),fwd::Dual(rho,1));
  return Rcpp::NumericVector::create(value.val,value.grad);
}
