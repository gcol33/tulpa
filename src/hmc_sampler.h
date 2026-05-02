// hmc_sampler.h
// Full HMC/NUTS backend with spatial, temporal, and zero-inflation support
// Supports ICAR/BYM2 spatial effects, RW/AR1 temporal, and ZI/hurdle models
//
// This umbrella header opens namespace tulpa_hmc { ... } and pulls in the
// per-section fragments in dependency order. Existing includers continue
// to do `#include "hmc_sampler.h"` and see the full set of declarations.

#ifndef TULPA_HMC_SAMPLER_H
#define TULPA_HMC_SAMPLER_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg_fast.h"
#include "hmc_temporal.h"
#include "hmc_temporal_gp.h"
#include "hmc_zi.h"
#include "hmc_svc.h"
#include "hmc_gp.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_latent.h"
#include "hmc_spatiotemporal.h"
#include "hmc_hsgp.h"
#include "hmc_tvc.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa_hmc {

// using-decls, enum parsers, ParamLayout + log-post / gradient declarations.
#include "hmc_sampler_decls.h"

// Leapfrog result variants, U-turn criterion, NUTSWorkspace buffer pool.
#include "hmc_sampler_nuts_infra.h"

// Mass-matrix block hierarchy (MassBlock -> PrecisionBlock -> KroneckerBlock
// -> SparseGMRFBlock -> DenseMassMatrix). Order matters: DenseMassMatrix
// embeds the earlier block types.
#include "hmc_sampler_mass_blocks.h"

// Adaptation state: Welford (mean / cov), DualAveraging.
#include "hmc_sampler_adapt.h"

// Chain state + result structs + NUTS helper declarations
// (DualAveraging is needed first — defined in hmc_sampler_adapt.h above).
#include "hmc_sampler_chain_state.h"

// Sampler function declarations + SoftAbs helpers.
#include "hmc_sampler_funcs.h"

// Mass-matrix configuration helpers (used by hmc_chain.cpp).
#include "hmc_sampler_config.h"

} // namespace tulpa_hmc

#endif // TULPA_HMC_SAMPLER_H
