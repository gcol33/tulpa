// One translation unit for the sources below: each Eigen and Rcpp template
// they share is instantiated, and reports its diagnostics, once rather than
// once per file. Add a source to the group it belongs with, then run
// Rscript tools/unity.R to update the Makevars.
//
// RcppEigen.h comes first: its wrap() specializations have to be declared
// before Rcpp.h is, and the first member to include either would decide.
#include <RcppEigen.h>

#include "array_utils.cpp"
#include "autodiff.cpp"
#include "cholmod_stub_resolve.cpp"
#include "cornish_fisher.cpp"
#include "cov_kernel_equiv_export.cpp"
#include "cov_kernel_export.cpp"
#include "crt_rng.cpp"
#include "family_terms_export.cpp"
#include "fastmath_data.cpp"
#include "gpu_backend.cpp"
#include "hyperprior_density_export.cpp"
#include "interval_gaussian_export.cpp"
#include "linalg_tri_layout_export.cpp"
#include "multinomial_logit_export.cpp"
#include "pc_prior_equiv_export.cpp"
#include "portable_math_export.cpp"
#include "sobol_export.cpp"
#include "sysmem.cpp"
#include "temporal_grad_equiv_export.cpp"
#include "tgmrf_nuts.cpp"
#include "tgmrf_registry.cpp"
#include "truncated_gaussian_export.cpp"
#include "tulpa_init.cpp"
