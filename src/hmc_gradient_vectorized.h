// hmc_gradient_vectorized.h
// Vectorized gradient computation with template specialization per model type.
// Used by composite/HSGP H-kernel paths for non-ZI, non-slopes model
// configurations on the legacy ratio path.
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
// Thin re-export: each fragment is self-contained and defines its symbols
// inside namespace tulpa_hmc::vectorized. Include this header from anywhere
// (top-level, not from inside an existing namespace block).

#ifndef TULPA_HMC_GRADIENT_VECTORIZED_H
#define TULPA_HMC_GRADIENT_VECTORIZED_H

#include "hmc_gradient_vectorized_workspace.h"
#include "hmc_gradient_vectorized_passes.h"
#include "hmc_gradient_vectorized_fused.h"
#include "hmc_gradient_vectorized_main.h"
#include "hmc_gradient_vectorized_shared.h"

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_H
