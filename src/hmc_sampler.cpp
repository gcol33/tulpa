// hmc_sampler.cpp
// Full HMC/NUTS backend with spatial, temporal, and ZI support
// Provides Stan-free Bayesian inference for all ratiod models

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_progress.h"
#include "autodiff.h"
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_car_proper.h"
#include "icar_kernel.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "lkj_chol_helpers.h"
#include "hmc_observation_likelihood.h"
#include "hmc_log_posterior_split.h"
#include <Rcpp.h>

// Include log_post_impl.h AFTER hmc_sampler.h so types are defined
#include "log_post_impl.h"
#include "tulpa/likelihood.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa_hmc {

// =====================================================================
// ICAR quadratic form: phi' Q phi  (rho = 1 special case of CAR)
// Delegates to the shared kernel in icar_kernel.h.
// =====================================================================

static inline double icar_quadratic_form_ptr(
    const double* phi, int J,
    const ModelData& data
) {
  return tulpa::car_quad_form(
      phi, J,
      data.adj_row_ptr.data(), data.adj_col_idx.data(), data.n_neighbors.data(),
      /*rho=*/1.0);
}

double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
) {
  return icar_quadratic_form_ptr(phi.data(), data.n_spatial_units, data);
}

// Shared collapsed GP workspace (used by both compute_log_post and compute_gradient_gp_collapsed)
thread_local CollapsedGPWorkspace collapsed_gp_ws;

// Shared collapsed ICAR/BYM2 workspace
thread_local CollapsedICARWorkspace collapsed_icar_ws;

// =====================================================================
// Log-posterior computation with OpenMP parallelization
// =====================================================================

double accumulate_log_prior_and_state(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LogPosteriorState& state,
    ObservationLikelihoodContext& ctx,
    bool populate_obs_state,
    const double* precomputed_st_log_prior,
    const double* precomputed_tgp_log_prior
) {
#include "hmc_sampler_log_prior_setup.h"

#include "hmc_sampler_log_prior_basic.h"

#include "hmc_sampler_log_prior_temporal_gp.h"

#include "hmc_sampler_log_prior_complex.h"

#include "hmc_sampler_log_prior_finalize.h"
}

// =====================================================================
// Observation-loop helper (parallel reduction over data.N).
// =====================================================================

double accumulate_obs_log_lik(const ObservationLikelihoodContext& ctx) {
  const auto& data = ctx.data;
  const auto& layout = ctx.layout;
  double log_lik = 0.0;

  // NOTE: Disable OpenMP for GP models to avoid race conditions.
  // The GP NNGP likelihood accesses shared data structures that may not be thread-safe.
  #ifdef _OPENMP
  int use_threads = (layout.is_gp || layout.is_multiscale_gp) ? 1 : data.n_threads;
  #pragma omp parallel for reduction(+:log_lik) schedule(static) \
          num_threads(use_threads)
  #endif
  for (int i = 0; i < data.N; i++) {
    log_lik += compute_observation_log_lik(ctx, i);
  }

  return log_lik;
}

// =====================================================================
// Orchestrators: compute_log_post / compute_log_prior / compute_log_lik_only
//
// The split satisfies the contract verified by tests/testthat/test-log-post-split.R:
//   compute_log_post == compute_log_prior + compute_log_lik_only
// (modulo IEEE arithmetic), and compute_log_lik_only now runs in a single
// pass instead of two compute_log_post calls (gcol33/tulpa#6 follow-up).
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* precomputed_st_log_prior,
    const double* precomputed_tgp_log_prior
) {
  if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
    return tulpa::compute_log_post_generic_spec_double(
        params, data, layout, skip_obs_loop);
  }

  LogPosteriorState state;
  ObservationLikelihoodContext ctx{params, data, layout};
  const bool populate = !skip_obs_loop;

  double log_prior = accumulate_log_prior_and_state(
      params, data, layout, state, ctx, populate,
      precomputed_st_log_prior, precomputed_tgp_log_prior);

  if (skip_obs_loop || (std::isinf(log_prior) && log_prior < 0.0)) {
    return log_prior;
  }
  return log_prior + accumulate_obs_log_lik(ctx);
}

double compute_log_prior(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  // Prior + structural terms only (skip the O(N) observation loop).
  return compute_log_post(params, data, layout, /*skip_obs_loop=*/true);
}

double compute_log_lik_only(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
    // Generic-spec path still derives by subtraction; only the legacy
    // (n_processes == 0) path has been split into separable stages.
    const double lp = tulpa::compute_log_post_generic_spec_double(
        params, data, layout, /*skip_obs_loop=*/false);
    const double lpr = tulpa::compute_log_post_generic_spec_double(
        params, data, layout, /*skip_obs_loop=*/true);
    return lp - lpr;
  }

  LogPosteriorState state;
  ObservationLikelihoodContext ctx{params, data, layout};
  const double log_prior = accumulate_log_prior_and_state(
      params, data, layout, state, ctx, /*populate_obs_state=*/true,
      /*precomputed_st_log_prior=*/nullptr,
      /*precomputed_tgp_log_prior=*/nullptr);

  // Match the historical (post - prior) behavior at -INFINITY: the obs
  // context may be partially populated, so the result is undefined and
  // tests don't probe this regime. Return 0 to keep the contract finite
  // when prior was finite, NaN-equivalent otherwise.
  if (std::isinf(log_prior) && log_prior < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return accumulate_obs_log_lik(ctx);
}

// =====================================================================

} // namespace tulpa_hmc
