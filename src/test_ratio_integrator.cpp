#include <RcppEigen.h>
#include "test_prior_model.h"
#include "ess_sampler.h"

namespace {
double ess_reference_observation(int, const double*, const double&, const double&,
    const std::vector<double>& q, const tulpa::ModelData&, const tulpa::ParamLayout&, const void*) {
  const double x=q[0]-.6,y=q[1]+.35;
  return .5*(q[0]*q[0]+q[1]*q[1])-.5*(1.5*x*x-1.6*x*y+y*y)/.86;
}
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_ess_correlated_reference() {
  tulpa::ModelData data;tulpa::LikelihoodSpec spec;
  tulpa_probe::gaussian_prior(data,spec,2);
  data.N=1;data.processes[0].X_flat.assign(2,0);
  spec.ll_double=ess_reference_observation;
  const auto layout=tulpa_hmc::compute_param_layout(data);
  tulpa_ess::ESSConfig config;
  config.n_iter=12000;config.n_warmup=2000;config.verbose=false;
  const auto result=tulpa_ess::run_ess_sampler(std::vector<double>(2,0),data,layout,config);
  if(!result.success) Rcpp::stop(result.error_msg);
  return Rcpp::wrap(result.samples);
}
// [[Rcpp::export]]
Rcpp::List cpp_test_integrator(Rcpp::NumericVector q,Rcpp::NumericVector p,double epsilon) {
  tulpa::ModelData data;tulpa::LikelihoodSpec spec;
  tulpa_probe::gaussian_prior(data,spec,q.size());
  auto layout=tulpa_hmc::compute_param_layout(data);
  std::vector<double> position(q.begin(),q.end()),momentum(p.begin(),p.end());
  const auto step=tulpa_hmc::leapfrog_step(position,momentum,epsilon,data,layout);
  tulpa_hmc::DenseMassMatrix mass;
  mass.init(q.size(),tulpa::MassMatrixType::DIAG);
  double lp=0;for(double x:position) lp-=0.5*x*x;
  return Rcpp::List::create(Rcpp::Named("q")=step.q,Rcpp::Named("p")=step.p,
    Rcpp::Named("hamiltonian")=tulpa_hmc::nuts_compute_hamiltonian_fast(lp,momentum.data(),mass,q.size()));
}
