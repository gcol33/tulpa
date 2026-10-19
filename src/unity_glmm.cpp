// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "aghq_re.cpp"
#include "re_cov_gibbs.cpp"
#include "re_cov_grad_export.cpp"
#include "tulpa_glmm_eta_draws.cpp"
#include "tulpa_glmm_layout_probe.cpp"
#include "tulpa_sample_glmm.cpp"
