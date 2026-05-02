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

namespace tulpa_hmc {

// =====================================================================
// Mass matrix configuration and helpers (used by hmc_chain.cpp)
// =====================================================================

inline constexpr int DENSE_MAX_PARAMS = 200;

struct MassMatrixConfig {
  MassMatrixType effective_metric;
  bool auto_selected_diag;
  std::vector<std::pair<int,int>> block_specs;
};

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
