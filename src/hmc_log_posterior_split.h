// hmc_log_posterior_split.h
// Splits compute_log_post into two stages:
//   1. accumulate_log_prior_and_state — priors + structural terms;
//      also populates the obs-loop context (latent NC reconstructions,
//      collapsed-Laplace modes, derived buffers).
//   2. accumulate_obs_log_lik — the parallel observation loop.
//
// compute_log_post is the orchestrator. compute_log_lik_only runs both
// in a single pass (instead of subtracting two compute_log_post calls).

#ifndef TULPA_HMC_LOG_POSTERIOR_SPLIT_H
#define TULPA_HMC_LOG_POSTERIOR_SPLIT_H

#include <vector>
#include "hmc_observation_likelihood.h"

namespace tulpa_hmc {

// Buffers that the obs context's pointers point into. They must live
// as long as the obs loop runs, so we keep them in a single struct
// that the orchestrator owns on the stack.
struct LogPosteriorState {
  std::vector<double> re_nc_flat;       // RE values reconstructed from z
  std::vector<double> temporal_f_nc;    // NC temporal GP f reconstruction
  std::vector<double> gp_w_nc_buf;      // NC GP w reconstruction / collapsed mode
  std::vector<double> msgp_hsgp_f_local;
  std::vector<double> msgp_hsgp_f_regional;
  std::vector<double> hsgp_f;
  std::vector<double> latent_sigma;
  std::vector<double> latent_factors_vec;
  std::vector<double> tvc_eta;
};

// Returns log_prior. If populate_obs_state is true, also fills `state`
// and `ctx` so the obs loop can run; otherwise the cheap-buffer fields
// are skipped (used by gradient calls that only need log_prior).
// May early-return -INFINITY when bounds are violated; in that case
// `ctx` is partially populated and must not be consumed.
double accumulate_log_prior_and_state(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    LogPosteriorState& state,
    ObservationLikelihoodContext& ctx,
    bool populate_obs_state,
    const double* precomputed_st_log_prior,
    const double* precomputed_tgp_log_prior
);

// Runs the parallel observation loop using ctx. Caller must have
// successfully populated ctx via accumulate_log_prior_and_state with
// populate_obs_state=true.
double accumulate_obs_log_lik(
    const ObservationLikelihoodContext& ctx
);

} // namespace tulpa_hmc

#endif
