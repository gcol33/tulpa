// latent_block.h
// LatentBlock: a single rank-deficient or full-rank latent prior block plugged
// into the multi-block nested-Laplace driver (src/nested_laplace.cpp).
//
// Each block lives at [start, start + size) in the joint latent vector x[].
// An observation i sees the block via one entry x[start + idx(i) - 1] (1-based
// to match spatial_idx / temporal_idx conventions; idx returns -1 if obs i
// does not see the block, e.g. an observation with no time stamp).
//
// At outer-grid point k the block's contribution to eta is
//   eta_i += d_fac(k) * x[start + idx(i) - 1].
// d_fac is the BYM2 hook: for ICAR/RW1/RW2/AR1/CAR_proper it is constant 1.0;
// for BYM2-phi it is sigma_k * sqrt(rho_k) * scale_factor; for BYM2-theta it
// is sigma_k * sqrt(1 - rho_k). Carrying this coefficient through the block
// lets the multi-block scatter handle both single-block and BYM2-style
// reparameterizations with one code path.
//
// This header is internal to tulpa (uses DenseVec/DenseMat from laplace_types).
// Model packages do not consume it directly — they reach the multi-block
// driver through the cpp_nested_laplace_* Rcpp entries.

#ifndef TULPA_LATENT_BLOCK_H
#define TULPA_LATENT_BLOCK_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <functional>

namespace tulpa {

struct LatentBlock {
    int start = 0;
    int size  = 0;

    // i -> latent index within block (1-based). Return -1 to skip obs i.
    std::function<int(int)> idx;

    // Grid-dependent linear-mixing coefficient on the block's eta contribution.
    // 1.0 for plain indexed blocks; sigma_k * sqrt(rho_k) * scale_factor for
    // BYM2-phi; sigma_k * sqrt(1 - rho_k) for BYM2-theta.
    std::function<double(int)> d_fac;

    // Adds block's prior Q at grid point k to (grad, H). x is the full latent
    // vector; the helper indexes into [start, start + size).
    std::function<void(DenseVec&, DenseMat&, const Rcpp::NumericVector&, int)>
        add_prior;

    // log p(x_block | theta_k). Summed across blocks gives the latent log-prior
    // contribution (the driver adds RE/beta log-priors on top).
    std::function<double(const Rcpp::NumericVector&, int)> log_prior;

    // Per-grid-point feasibility (e.g. proper-CAR PD check via log|D - rho*W|).
    // Return false to short-circuit the inner solve at this k with
    // log_marginal = -inf. May be empty (treated as always true).
    std::function<bool(int)> prep;

    // Optional sum-to-zero / soft centering, applied after each Newton step
    // on the block's sub-vector. May be empty (no centering, e.g. AR1).
    std::function<void(Rcpp::NumericVector&)> center;
};

} // namespace tulpa

#endif // TULPA_LATENT_BLOCK_H
