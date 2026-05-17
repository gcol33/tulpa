// latent_block.h
// LatentBlock: a single rank-deficient or full-rank latent prior block plugged
// into the multi-block nested-Laplace driver (src/nested_laplace_multi.h) or
// the joint multi-arm driver (src/nested_laplace_joint_multi.h).
//
// Each block lives at [start, start + size) in the joint latent vector x[].
// An observation i (within its arm) sees the block via one entry
// x[start + idx(i, k_arm) - 1] (1-based to match spatial_idx / temporal_idx
// conventions; idx returns -1 if the obs does not see the block, e.g. an
// observation with no time stamp).
//
// At outer-grid point k the block's contribution to arm k_arm's eta is
//   eta_i += arm_scale(k_arm, k) * d_fac(k) * x[start + idx(i, k_arm) - 1].
// d_fac is the BYM2 hook: for ICAR/RW1/RW2/AR1/CAR_proper it is constant 1.0;
// for BYM2-phi it is sqrt(rho_k) * scale_factor; for BYM2-theta it is
// sqrt(1 - rho_k); for IID it is sigma_k. Carrying this coefficient through
// the block lets the scatter handle plain blocks, IID-as-sigma, and BYM2-style
// reparameterizations with one code path.
//
// arm_scale is the joint-driver hook for INLA-style `copy=` semantics: donor
// arms see one sigma per grid point, the copy arm sees another. Empty in the
// single-arm driver and in shared (non-copied) blocks of the joint driver.
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

    // (i, k_arm) -> latent index within block (1-based). Return -1 to skip
    // obs i. Single-arm callers pass k_arm = 0; single-arm factories ignore
    // it. Joint factories dispatch on k_arm to per-arm index vectors.
    std::function<int(int /*i*/, int /*k_arm*/)> idx;

    // Grid-dependent linear-mixing coefficient on the block's eta contribution
    // (per outer-grid point k). 1.0 for plain indexed blocks (ICAR/RW1/RW2/
    // AR1/CAR_proper in single-arm mode and shared mode); sqrt(rho_k) *
    // scale_factor for BYM2-phi; sqrt(1 - rho_k) for BYM2-theta; sigma_k for
    // IID. Combined with arm_scale (when present) to form the per-arm
    // effective coefficient on x[idx].
    std::function<double(int /*k_grid*/)> d_fac;

    // Per-arm linear scale on the block's eta contribution (joint driver
    // only). Multiplied into d_fac(k_grid) when the block contributes to arm
    // k_arm's eta. Used for INLA `copy=` semantics: donor arms see one sigma,
    // the copy arm sees another. Empty = no per-arm scaling (every arm sees
    // d_fac(k_grid) directly). The single-arm driver does not call arm_scale.
    std::function<double(int /*k_arm*/, int /*k_grid*/)> arm_scale;

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
