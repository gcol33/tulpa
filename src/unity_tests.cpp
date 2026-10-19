// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "test_ccallable_registry.cpp"
#include "test_cholesky_log_det_signal.cpp"
#include "test_fastmath_fixture.cpp"
#include "test_helpers.cpp"
#include "test_hsgp_warm_start.cpp"
#include "test_laplace_sample_fixture.cpp"
#include "test_lkj_chol.cpp"
#include "test_mass_lowrank.cpp"
#include "test_nan_gradient_nuts.cpp"
#include "test_pg_nngp_conditional.cpp"
#include "test_smc_cabi.cpp"
#include "test_st_hsgp_prior.cpp"
#include "test_st_iv_fixture.cpp"
#include "test_temporal_gp_fixture.cpp"
