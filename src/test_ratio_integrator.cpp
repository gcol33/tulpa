#include <RcppEigen.h>
#include "test_prior_model.h"
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
