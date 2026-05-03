// hmc_gradient_hsgp.cpp
// Handcoded H-mode gradient functions for HSGP and multi-scale HSGP models.
// Migrated from hmc_gradient_hsgp_impl.h.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "hmc_likelihood.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

namespace tulpa_hmc {

#include "hmc_gradient_vectorized.h"
#include "hmc_gradient_helpers_impl.h"
#include "hmc_gradient_shared.h"
#include "hmc_gradient_hsgp_impl.h"

} // namespace tulpa_hmc
