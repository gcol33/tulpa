// hmc_nuts_sampler.cpp
// HMC/NUTS integration, mass-matrix adaptation, and chain runners.
//
// This translation unit is intentionally a thin umbrella: each
// "// ===" section of the previous monolithic file now lives in its
// own per-section fragment header. The fragments are NOT standalone
// translation units — they are included only from here, inside
// namespace tulpa_hmc, in the order that preserves the original
// definition sequence.

#include "hmc_sampler.h"
#include "hmc_progress.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "log_post_impl.h"
#include <RcppEigen.h>
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa_hmc {

// extern decls + re_value_for_eta + DualAveraging method definitions.
#include "hmc_nuts_dual_avg.h"

// Leapfrog integrator + initial-step-size helper.
#include "hmc_nuts_leapfrog.h"

// Stan-style find_reasonable_epsilon (identity / diagonal / dense mass).
#include "hmc_nuts_find_epsilon.h"

// NUTS helpers (log-sum-exp, leapfrog_step_with_grad, U-turn check, ...).
#include "hmc_nuts_helpers.h"

// Optimized NUTS: zero-allocation in-place leapfrog + build_tree_fast.
#include "hmc_nuts_optimized.h"

// SoftAbs per-trajectory metric (Riemannian-like divergence retry).
#include "hmc_nuts_softabs.h"

// Mass-matrix selection / warm-start helpers.
#include "hmc_nuts_mass_init.h"

// Single-chain NUTS driver (the hot path).
#include "hmc_nuts_chain.h"

// OpenMP across-chain parallel runner.
#include "hmc_nuts_parallel.h"

} // namespace tulpa_hmc
