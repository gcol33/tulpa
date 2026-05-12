// hmc_nuts_chain.cpp
// Single-chain NUTS driver: run_hmc_chain_cpp + run_hmc_chain wrapper.
//
// The body of run_hmc_chain_cpp is assembled from function-body fragment
// headers (hmc_nuts_chain_setup.h, hmc_nuts_chain_iter_*.h). Those
// fragments are intentionally non-standalone: they reference locals
// declared in run_hmc_chain_cpp's body. They will move into proper
// helper functions in a follow-up refactor.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <Rcpp.h>

#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_progress.h"
#include "hmc_sampler.h"
#include "log_post_impl.h"

namespace tulpa_hmc {

// g_gradient_mode is declared in hmc_sampler_decls.h.
extern thread_local CollapsedGPWorkspace collapsed_gp_ws;
extern thread_local CollapsedICARWorkspace collapsed_icar_ws;

// =====================================================================
// Run single HMC chain
// =====================================================================

// Pure C++ version - safe for OpenMP parallel regions
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian,
    const std::vector<double>& inv_metric_init
) {
#include "hmc_nuts_chain_setup.h"

  for (int iter = 0; iter < n_iter; iter++) {
#include "hmc_nuts_chain_iter_window.h"

#include "hmc_nuts_chain_iter_nuts.h"
#include "hmc_nuts_chain_iter_hmc.h"

#include "hmc_nuts_chain_iter_store.h"

    // Note: verbose output disabled in parallel - not thread-safe
    // Progress will be reported after parallel region
  }

  result.epsilon = da.final_epsilon();

  // Diagnostic stats - only when verbose
  if (verbose) {
    int sampling_total_lf = 0;
    for (int i = 0; i < result.n_sample; i++) sampling_total_lf += result.n_leapfrog[i];
    REprintf("  [STATS] Chain %d: metric=%s, adapted=%d, warmup_LF=%d, sampling_LF=%d, total_LF=%d, epsilon=%.6f\n",
             chain_id + 1,
             metric_name(mass.type),
             (int)mass.adapted,
             warmup_total_leapfrog, sampling_total_lf,
             warmup_total_leapfrog + sampling_total_lf, result.epsilon);
  }

  if (verbose && (softabs_retries > 0 || softabs_metric_active)) {
    REprintf("  [SoftAbs] Chain %d: metric=%s, %d divergent retried (up to %d attempts), %d resolved (%d remained)\n",
             chain_id + 1,
             softabs_metric_active ? "active" : "inactive",
             softabs_retries, SOFTABS_MAX_RETRIES, softabs_successes,
             softabs_retries - softabs_successes);
  }

  return result;
}

// R wrapper version - for single chain or non-parallel use
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian,
    const std::vector<double>& inv_metric_init
) {
  // Runtime gradient check: compare active gradient function against numerical
  if (g_gradient_mode != GradientMode::NUMERICAL) {
    bool grad_ok = verify_gradient_runtime(q_init, data, layout, 1e-4);
    if (!grad_ok) {
      g_gradient_mode = GradientMode::NUMERICAL;
      Rcpp::warning(
        "Gradient mismatch detected: active gradient function disagrees with "
        "numerical gradients (max rel diff > 1e-4). Falling back to numerical "
        "gradients (mode='N'). This is slower but correct. Please report this "
        "as a bug at https://github.com/gcol33/numdenom/issues"
      );
    }
  }

  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian, inv_metric_init
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcerr << "Chain " << (chain_id + 1) << " complete. "
                << "Divergent: " << n_div << std::endl;
  }

  return result;
}

}  // namespace tulpa_hmc
