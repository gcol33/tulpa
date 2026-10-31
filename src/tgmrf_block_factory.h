// tgmrf_block_factory.h
// Build a LatentBlock for a templated user-supplied GMRF (tgmrf) plugged
// into the joint multi-arm nested-Laplace driver. Joint analogue of the
// single-arm "tgmrf" block in nested_laplace_multi.cpp.
//
// The R side precomputes Q(theta_k) at every outer-grid row plus log|Q_k|
// and log p(theta_k). The factory just reads those arrays and assembles the
// callbacks; the C++ side never indexes theta_grid for this block (Q_k is
// fully materialized R-side).
//
// Pattern contract. The SparseHessianBuilder is initialized once at fit-
// time, so the frame it holds is the UNION of the per-grid Q sparsity
// patterns, assembled by add_prior_pattern below. A grid point whose Q is
// sparser than the union writes into structural zeros, which is what the
// frame is for. Nothing has to be padded R-side.
//
// Contrib kind: INDEXED_SINGLE (one block-local DOF per obs via obs_idx).
// Prior fill: USER_CSC.
// Copy semantics not supported.

#ifndef TULPA_TGMRF_BLOCK_FACTORY_H
#define TULPA_TGMRF_BLOCK_FACTORY_H

#include "latent_block.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

LatentBlock make_tgmrf_block(
    int                                              start,
    int                                              size,
    std::function<int(int /*i*/, int /*k_arm*/)>     obs_idx_fn,
    const Rcpp::List&                                Q_csc_p_per_grid,
    const Rcpp::List&                                Q_csc_i_per_grid,
    const Rcpp::List&                                Q_csc_x_per_grid,
    const Rcpp::NumericVector&                       logdet_Q_per_grid,
    const Rcpp::NumericVector&                       log_pi_theta_per_grid,
    int                                              block_index
);

} // namespace tulpa

#endif // TULPA_TGMRF_BLOCK_FACTORY_H
