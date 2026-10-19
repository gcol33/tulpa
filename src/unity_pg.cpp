// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "pg_binomial.cpp"
#include "pg_binomial_bym2.cpp"
#include "pg_binomial_gp.cpp"
#include "pg_binomial_gp_rsr.cpp"
#include "pg_binomial_multiscale_gp.cpp"
#include "pg_binomial_rsr.cpp"
#include "pg_binomial_temporal.cpp"
#include "pg_negbin.cpp"
#include "pg_rng.cpp"
#include "pg_spatial.cpp"
