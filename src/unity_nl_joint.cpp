// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "cell_coupling_probe.cpp"
#include "cell_coupling_registry.cpp"
#include "joint_inner_vcov.cpp"
#include "nested_laplace_joint_batch.cpp"
#include "nested_laplace_joint_multi.cpp"
#include "test_cell_coupling_register.cpp"
