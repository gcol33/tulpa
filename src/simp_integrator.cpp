// simp_integrator.cpp
// Active symplectic integrator for the HMC/NUTS trajectory, backed by SIMP.
// The scheme (op sequence + coefficients) is the single source of truth shared
// by both leapfrog steppers; selecting an integrator swaps the scheme here.

#include <Rcpp.h>
#include <string>

#include "hmc_sampler_decls.h"  // simp/simp.h, the extern declarations

namespace tulpa_hmc {

// Default is plain leapfrog, so an unset integrator reproduces the historical
// behaviour byte-for-byte.
simp::Scheme g_integrator_scheme = simp::leapfrog();

void set_integrator_scheme(const std::string& name) {
  g_integrator_scheme = simp::scheme_by_name(name);  // throws on unknown name
}

const simp::Scheme& get_integrator_scheme() {
  return g_integrator_scheme;
}

}  // namespace tulpa_hmc

// Select the trajectory integrator by name. Returns the previous name so the
// caller can restore it. Throws (propagated to R) on an unknown name.
// [[Rcpp::export]]
std::string tulpa_set_integrator_cpp(std::string name) {
  std::string previous = tulpa_hmc::get_integrator_scheme().name;
  tulpa_hmc::set_integrator_scheme(name);
  return previous;
}

// [[Rcpp::export]]
std::string tulpa_get_integrator_cpp() {
  return tulpa_hmc::get_integrator_scheme().name;
}
