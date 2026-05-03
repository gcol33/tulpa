// hmc_gradient_feature.cpp
// Handcoded H-mode gradient functions for SVC, HSGP-SVC, TVC, and latent
// factor models.  Migrated from hmc_gradient_feature_impl.h.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_gp_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "hmc_likelihood.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

namespace tulpa_hmc {

// Pull in VecGradWorkspace type and vectorized helpers.
// hmc_likelihood.h must be included before this (already done above).
#include "hmc_gradient_vectorized.h"

// Static-inline helpers (gradient priors, residuals, scatter, phi).
#include "hmc_gradient_helpers_impl.h"

// Shared preamble / eta / dispatch / epilogue building blocks.
// Provides extern thread_local vec_grad_ws declaration.
#include "hmc_gradient_shared.h"

// Function definitions.
#include "hmc_gradient_feature_impl.h"

} // namespace tulpa_hmc
