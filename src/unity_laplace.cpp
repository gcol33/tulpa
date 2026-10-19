// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "implicit_diff.cpp"
#include "laplace_core.cpp"
#include "laplace_core_gp.cpp"
#include "laplace_core_spatial.cpp"
#include "laplace_likelihoods.cpp"
#include "laplace_profile.cpp"
#include "laplace_s2z_probe.cpp"
#include "laplace_spatial_priors.cpp"
#include "laplace_spec.cpp"
#include "laplace_temporal_priors.cpp"
#include "log_post_generic_impl.cpp"
#include "sparse_cholesky.cpp"
#include "stochastic_logdet.cpp"
