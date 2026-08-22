// hmc_sampler_config.h
// Fragment of hmc_sampler.h. Self-contained: defines symbols inside
// namespace tulpa_hmc.
// MassMatrixConfig + select/init/warm-start helpers (used by
// hmc_chain.cpp), runtime gradient verification.
#ifndef TULPA_HMC_SAMPLER_CONFIG_H
#define TULPA_HMC_SAMPLER_CONFIG_H

#include <utility>
#include <vector>

#include "hmc_sampler_decls.h"        // ModelData, ParamLayout
#include "hmc_sampler_mass_blocks.h"  // DenseMassMatrix, MassMatrixType

#include <algorithm>
#include <cmath>

namespace tulpa_hmc {

// =====================================================================
// Mass matrix configuration and helpers (used by hmc_chain.cpp)
// =====================================================================

inline constexpr int DENSE_MAX_PARAMS = 200;

struct MassMatrixConfig {
  MassMatrixType effective_metric;
  bool auto_selected_diag;
  std::vector<std::pair<int,int>> block_specs;

  // Whether the Type-IV precision-informed override runs at warmup end. The
  // GMRF request resolves to a DIAG `effective_metric` plus this flag, so the
  // per-step paths never see a fifth metric; what the request buys is one
  // block of the diagonal being computed rather than accumulated.
  bool st_gmrf = false;
  // Whether that diagonal additionally carries the interaction's two soft
  // sum-to-zero margins as an explicit low-rank term (mass_matrix =
  // "gmrf_margin"). Implies st_gmrf: the low-rank part rides on the same
  // precision-informed diagonal, and the margin directions are the part of the
  // block's stiffness a diagonal cannot reach at all (gcol33/tulpa#597).
  bool st_gmrf_margin = false;
  // Empty when st_gmrf is true; otherwise why the request could not be
  // honoured, from hmc_mass_st_gmrf.h's closed vocabulary. Non-empty with a
  // GMRF request is what a user reads instead of a silent fallback.
  const char* st_gmrf_declined = "";
};

// One clamp on an inverse-mass diagonal, shared by the structural warm start,
// a caller-supplied metric and the Type-IV override. Outside this band the
// metric is either singular or so wide that find_reasonable_epsilon cannot
// recover, and the three sites disagreeing on where that band sits is the
// copy-paste this replaces.
inline double clamp_inv_mass(double v) {
  if (!(v > 0.0) || !std::isfinite(v)) return 1.0;
  return std::max(1e-3, std::min(v, 1e3));
}

// Select mass matrix type (AUTO resolution, block detection, DENSE override)
// and initialize the DenseMassMatrix object.
MassMatrixConfig select_and_init_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    MassMatrixType metric_type,
    bool verbose
);

// Warm-start mass matrix diagonal from model structure
void warm_start_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    bool verbose
);

// Runtime gradient verification (compare active gradient vs numerical)
bool verify_gradient_runtime(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol = 1e-4
);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_SAMPLER_CONFIG_H
