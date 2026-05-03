// hmc_gradient_gp.cpp
// Handcoded H-mode gradient functions for NNGP, collapsed GP/ICAR, and
// temporal GP models.  Migrated from hmc_gradient_gp_impl.h.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_gp_autodiff.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_likelihood.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

namespace tulpa_hmc {

#include "hmc_gradient_vectorized.h"
#include "hmc_gradient_helpers_impl.h"
#include "hmc_gradient_shared.h"

// Collapsed workspaces defined in hmc_sampler.cpp.
extern thread_local CollapsedGPWorkspace collapsed_gp_ws;
extern thread_local CollapsedICARWorkspace collapsed_icar_ws;

#include "hmc_gradient_gp_impl.h"

} // namespace tulpa_hmc
