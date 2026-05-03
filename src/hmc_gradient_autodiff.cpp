// hmc_gradient_autodiff.cpp
// Autodiff and handcoded H-mode gradient functions for GP, multi-scale GP,
// and GP+temporal models.  Migrated from hmc_gradient_autodiff_impl.h.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "autodiff.h"
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "hmc_temporal_autodiff.h"
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

#include "hmc_gradient_autodiff_impl.h"

} // namespace tulpa_hmc
