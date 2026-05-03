// hmc_gradient_analytical.cpp
// Analytical H-mode gradient function for simple spatial/temporal models.
// Migrated from hmc_gradient_analytical_impl.h.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_car_proper.h"
#include "lkj_chol_helpers.h"
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

#include "hmc_gradient_analytical_impl.h"

} // namespace tulpa_hmc
