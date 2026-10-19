// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "ess_sampler.cpp"
#include "hmc_gradient_dispatch.cpp"
#include "hmc_gradient_fallback.cpp"
#include "hmc_gradients.cpp"
#include "hmc_mass_matrix.cpp"
#include "hmc_mass_st_gmrf.cpp"
#include "hmc_nuts_chain.cpp"
#include "hmc_nuts_curvature.cpp"
#include "hmc_nuts_dual_avg.cpp"
#include "hmc_nuts_find_epsilon.cpp"
#include "hmc_nuts_helpers.cpp"
#include "hmc_nuts_leapfrog.cpp"
#include "hmc_nuts_mass_init.cpp"
#include "hmc_nuts_optimized.cpp"
#include "hmc_nuts_parallel.cpp"
#include "hmc_param_layout.cpp"
#include "hmc_rcpp_utils.cpp"
#include "hmc_sampler.cpp"
#include "hmc_walnuts.cpp"
#include "mclmc.cpp"
