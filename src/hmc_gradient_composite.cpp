// hmc_gradient_composite.cpp
// Catch-all handcoded H-mode gradient for exotic multi-feature combinations.
// Migrated from hmc_gradient_composite_impl.h.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_gp_autodiff.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "lkj_chol_helpers.h"
#include "hmc_likelihood.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

namespace tulpa_hmc {

#include "hmc_gradient_vectorized.h"
#include "hmc_gradient_helpers_impl.h"
#include "hmc_gradient_shared.h"
#include "hmc_gradient_composite_impl.h"

} // namespace tulpa_hmc
