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

#include <memory>

#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <tulpa/nested_progress.h>

#include "hmc_progress.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

// g_gradient_mode is declared in hmc_sampler_decls.h.

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

  // NUTS progress + ETA (gcol33/tulpaObs#43). A serial caller (single chain,
  // not inside the across-chain OpenMP region) with no reporter already active
  // builds its own from the scoped option; the across-chain core builds one
  // shared reporter before its parallel region, so each chain here just ticks
  // the already-active pointer. The R option is read only on the main thread
  // (guarded by !omp_in_parallel()).
  std::unique_ptr<::tulpa_progress::GridProgress> own_progress;
  bool in_parallel = false;
#ifdef _OPENMP
  in_parallel = (omp_in_parallel() != 0);
#endif
  if (g_active_grid_progress == nullptr && !in_parallel) {
    own_progress = make_nuts_progress(n_iter, 1);
    if (own_progress) g_active_grid_progress = own_progress.get();
  }

  for (int iter = 0; iter < n_iter; iter++) {
#include "hmc_nuts_chain_iter_window.h"

#include "hmc_nuts_chain_iter_nuts.h"
#include "hmc_nuts_chain_iter_hmc.h"

#include "hmc_nuts_chain_iter_store.h"

    // Note: verbose console output disabled in parallel - not thread-safe.
    // The progress tick only touches a counter + clock + (throttled) file
    // write, has no effect on the sampler RNG or state, and serialises across
    // chains via the named critical; the console line self-suppresses in a
    // parallel region (the heartbeat file is the across-chain channel).
    if (g_active_grid_progress != nullptr) {
#ifdef _OPENMP
      #pragma omp critical(tulpa_nuts_progress)
#endif
      g_active_grid_progress->tick();
    }
  }

  if (own_progress) {
    g_active_grid_progress->finish();
    g_active_grid_progress = nullptr;
  }

  result.epsilon = da.final_epsilon();

  // Warm-start / resume outputs (gcol33/tulpa#29): the adapted inverse-mass
  // diagonal and the final raw sampler position. `q` holds the last sampler
  // state in the sampling parameterization (z for NC GP, not the stored w);
  // resuming from it is what continues the trajectory.
  result.inv_metric_diag = mass.inv_mass_diag;
  result.final_position = q;

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
        "as a bug at https://github.com/gcol33/tulpa/issues"
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
