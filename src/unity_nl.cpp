// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "areal_input_check.cpp"
#include "block_projector.cpp"
#include "gpu_nngp_laplace.cpp"
#include "hsgp_block_factory.cpp"
#include "hsgp_sum.cpp"
#include "laplace_spec_fit.cpp"
#include "mcar_block_factory.cpp"
#include "nested_laplace.cpp"
#include "nested_laplace_checkpoint.cpp"
#include "nested_laplace_grid.cpp"
#include "nested_laplace_joint_core.cpp"
#include "nested_laplace_multi.cpp"
#include "nl_entry_inputs.cpp"
#include "nl_lq_density.cpp"
#include "nngp_block_factory.cpp"
#include "re_structure.cpp"
#include "sparse_hessian.cpp"
#include "spde_block_factory.cpp"
#include "spde_qbuilder.cpp"
#include "tgmrf_block_factory.cpp"
