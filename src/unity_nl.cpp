// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "nested_laplace.cpp"
#include "nested_laplace_multi.cpp"
#include "nested_laplace_test_occupancy.cpp"
#include "nl_field_identity_export.cpp"
#include "nl_lq_density.cpp"
