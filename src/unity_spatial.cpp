// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "brasil_export.cpp"
#include "gp_nc_apply.cpp"
#include "gp_predict.cpp"
#include "hsgp_basis.cpp"
#include "msgp_nc_apply.cpp"
#include "nngp_nc_term_apply.cpp"
#include "nngp_twin_export.cpp"
#include "spde_field_se.cpp"
#include "spde_fractional_marginal.cpp"
#include "spde_laplace.cpp"
#include "spde_nc_apply.cpp"
#include "spde_nc_transform.cpp"
#include "svc_nc_apply.cpp"
#include "tulpa_spde_layout_probe.cpp"
#include "tulpa_spde_sampler.cpp"
