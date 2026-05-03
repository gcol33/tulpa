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

// Self-contained headers (each opens its own namespace). Include before
// the namespace block so they don't end up nested inside tulpa_hmc::tulpa_hmc::.
#include "hmc_gradient_vectorized.h"
#include "hmc_gradient_helpers_impl.h"
#include "hmc_gradient_shared.h"

using namespace Rcpp;

namespace tulpa_hmc {

// Function definitions.
#include "hmc_gradient_feature_impl.h"

} // namespace tulpa_hmc
