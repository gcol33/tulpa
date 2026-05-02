// hmc_gradient_vectorized.h
// Vectorized gradient computation with template specialization per model type.
// Replaces scalar per-observation loop in compute_gradient_analytical() for
// non-ZI, non-slopes model configurations.
//
// Architecture (three-pass):
//   Pass 1: Vectorized eta = X * beta  (Eigen matvec)
//           + dense-expanded RE/spatial/temporal additions
//   Pass 2: Scalar model-specific residual kernel (templated per ModelType)
//   Pass 3: Vectorized grad_beta = X^T * resid  (Eigen matvec)
//
// Template dispatch eliminates dead branches at compile time, reducing
// instruction cache pressure from ~30-50KB to ~5KB per instantiation.
//
// IMPORTANT: This header is included from hmc_gradients.cpp from inside
// namespace tulpa_hmc { ... }, after hmc_sampler.h, RcppEigen.h, and
// hmc_likelihood.h have been included. The included sub-fragments do not
// open or close any namespace; they expect to be expanded inside
// namespace tulpa_hmc { namespace vectorized { ... } }.

#ifndef TULPA_HMC_GRADIENT_VECTORIZED_H
#define TULPA_HMC_GRADIENT_VECTORIZED_H

#include <vector>
#include <cmath>
#include <cstring>

// Assumes hmc_sampler.h and RcppEigen.h are already included by the .cpp file.
// Assumes log_lik_binomial/negbin/poisson/gamma are available from hmc_likelihood.h.
// IMPORTANT: This file is included from within namespace tulpa_hmc {} in hmc_gradients.cpp.
// Do NOT wrap contents in namespace tulpa_hmc — it would be doubly nested.

namespace vectorized {

// Workspace struct is needed by every other section.
#include "hmc_gradient_vectorized_workspace.h"

// Pass 1 / Pass 2 (templates + dispatcher) / Pass 3 helpers.
#include "hmc_gradient_vectorized_passes.h"

// Fused single-pass gradient (uses workspace + Pass 2 kernels via direct calls).
#include "hmc_gradient_vectorized_fused.h"

// Main vectorized gradient + top-level dispatcher (uses Pass 1/2/3 helpers).
#include "hmc_gradient_vectorized_main.h"

// Shared kernel exposed to specialized gradient functions (HSGP, temporal GP, etc.).
#include "hmc_gradient_vectorized_shared.h"

}  // namespace vectorized
// Note: namespace tulpa_hmc is NOT closed here — the .cpp file handles it

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_H
