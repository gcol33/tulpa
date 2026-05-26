// laplace_api.h
// Shared POD result struct for the cross-package Laplace C-ABI shims.
//
// The spec-driven Laplace shims (laplace_spec_api.h / tulpa_laplace_spec_*)
// reuse LaplaceShimResult, so it lives here as a small ABI-stable POD shared
// across separately compiled DLLs (an Rcpp::List / NumericVector would not be).

#ifndef TULPA_LAPLACE_API_H
#define TULPA_LAPLACE_API_H

#include "model_data.h"  // brings in TULPA_ABI_VERSION (kept for ABI grouping)

namespace tulpa {

// ----------------------------------------------------------------------------
// Result struct shared by the Laplace shims.
//   mode:   [n_x] caller must free via free_buffers
//   n_x is set by the shim. Other scalars are filled by the shim.
// ----------------------------------------------------------------------------
struct LaplaceShimResult {
    int n_x;
    double* mode;        // [n_x]
    double log_det_Q;
    double log_marginal;
    int n_iter;
    int converged;       // 0 or 1

    void free_buffers() {
        if (mode) { delete[] mode; mode = nullptr; }
    }
};

} // namespace tulpa

#endif // TULPA_LAPLACE_API_H
