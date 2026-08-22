// hmc_mass_st_gmrf.h
// Precision-informed diagonal mass over the Type-IV spatiotemporal block.
//
// Warmup adaptation estimates the marginal posterior variance of every
// coordinate from the chain's own samples. For a Type-IV interaction that is
// S*T coordinates estimated from a few hundred draws, and the answer is
// already available in closed form: the block's conditional posterior is
// Gaussian to the same order the Laplace approximation is, with precision
//
//   Q = tau (Q_s (x) Q_t) + diag(h_lik) + the two sum-to-zero margins,
//
// so diag(Q^-1) is the marginal variance the Welford accumulator is trying to
// reach. This computes it once, at the warmup-end position, and installs it as
// the inverse-mass diagonal over the block. Nothing else about the metric
// changes: the result is a DIAGONAL mass, so every leapfrog, kinetic-energy
// and momentum path stays the one the diagonal metric already takes.
//
// h_lik is the per-coordinate eta-space likelihood curvature, and reaching it
// on the generic path means LikelihoodSpec::eta_weights_fn -- the IRLS
// callback laplace_mode_spec_dense drives. A spec that ships none declines.
#ifndef TULPA_HMC_MASS_ST_GMRF_H
#define TULPA_HMC_MASS_ST_GMRF_H

#include <vector>

#include "hmc_sampler_decls.h"

namespace tulpa_hmc {

// Outcome of one override attempt. `reason` is empty on success and otherwise
// one of the closed vocabulary at the top of the .cpp, naming which
// precondition failed -- a declined override is reported, never silent.
struct StGmrfMassResult {
  bool ok = false;
  const char* reason = "";

  // diag(Q^-1) over the block, in the coordinate the sampler holds. Length
  // st_delta_end - st_delta_start on success, empty otherwise.
  std::vector<double> inv_mass;

  // The soft sum-to-zero margins, as the low-rank mass term reads them
  // (hmc_mass_lowrank.h): the block is S row groups of length T plus T column
  // groups of length S, with these precisions. Carried on the result rather
  // than re-derived at the call site so the mass term and the Q the variances
  // came from cannot disagree about the parameterization's scaling.
  int    n_spatial = 0;
  int    n_times = 0;
  double lambda_row = 0.0;        // s2z precision on each spatial unit's time sum
  double lambda_col = 0.0;        // s2z precision on each time's spatial sum

  // Read by the benchmark harness, not by the sampler.
  int    n_block = 0;             // S * T
  int    n_curvature_clamped = 0; // coordinates whose likelihood curvature was < 0
  double ridge_applied = 0.0;     // relative PD backstop, 0 when the first factorization took
  double log_det_Q = 0.0;
};

// Compute the override at position `q`. Never throws: every failure comes back
// as ok = false with a reason.
StGmrfMassResult st_gmrf_inv_mass(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout
);

// Whether this model is even a candidate, checked without touching the
// position. Used at chain setup so a requested override that can never fire
// says so once rather than at every warmup end.
const char* st_gmrf_precondition(const ModelData& data, const ParamLayout& layout);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_MASS_ST_GMRF_H
