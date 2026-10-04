// hmc_walnuts.h
// The engine's WALNUTS transition (tulpa/walnuts.h) over its own metric
// (DenseMassMatrix, every structure it carries) and gradient dispatch.

#ifndef TULPA_HMC_WALNUTS_H
#define TULPA_HMC_WALNUTS_H

#include <random>
#include <vector>

#include <tulpa/walnuts.h>
#include "hmc_sampler.h"

namespace tulpa_hmc {

// One WALNUTS transition from (q, grad, log_post), which are overwritten with
// the selected state. `step` is the macro step size.
tulpa::WalnutsTransitionResult walnuts_transition(
    std::vector<double>& q, std::vector<double>& grad, double& log_post,
    double step, int max_depth, const tulpa::WalnutsConfig& cfg,
    const DenseMassMatrix& mass, GradientFn gradient_fn,
    const ModelData& data, const ParamLayout& layout,
    tulpa::WalnutsWorkspace& ws, std::mt19937& rng);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_WALNUTS_H
