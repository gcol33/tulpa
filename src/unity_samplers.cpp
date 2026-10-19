// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "psis_kernel.cpp"
#include "simp_integrator.cpp"
#include "smc_sampler.cpp"
#include "tulpa_beta_sampler.cpp"
#include "tulpa_generic_sampler.cpp"
#include "tulpa_test_funnel.cpp"
#include "vi_convergence_probe.cpp"
#include "vi_sampler.cpp"
