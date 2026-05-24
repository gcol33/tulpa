// hmc_nuts_parallel.cpp
// run_hmc_parallel_chains_cpp: pure-C++ OpenMP across-chain core.
// run_hmc_parallel_chains:     thin Rcpp-returning wrapper over it.

#include <algorithm>
#include <cmath>
#include <vector>

#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hmc_progress.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

// g_gradient_mode is declared in hmc_sampler_decls.h.

// =====================================================================
// Pure-C++ across-chain core (OpenMP). Per-chain initial position and
// per-chain initial inverse-mass diagonal, so it serves both a fresh fit
// (caller broadcasts one init) and a warm-started resume (caller passes
// each chain's final_position + inv_metric from a previous run, with
// n_warmup = 0 — gcol33/tulpa#29 + #30). Returns the pure-C++ results
// (no Rcpp types), so it is safe to call from the C ABI as well.
//
// inv_metric_per_chain may be empty (all chains use the structural
// default) or length n_chains (entry c may itself be empty for chain c).
// =====================================================================
std::vector<HMCResultCpp> run_hmc_parallel_chains_cpp(
    const std::vector<std::vector<double>>& q_init_per_chain,
    const std::vector<std::vector<double>>& inv_metric_per_chain,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  ParamLayout layout = compute_param_layout(data);

  static const std::vector<double> kNoMetric;
  auto metric_for = [&](int c) -> const std::vector<double>& {
    return ((int)inv_metric_per_chain.size() == n_chains)
               ? inv_metric_per_chain[c]
               : kNoMetric;
  };

  // Runtime gradient check on chain 0's init: compare the active gradient
  // against numerical BEFORE spawning parallel chains. Single-threaded here,
  // so the R API and g_gradient_mode mutation are safe; the decision then
  // applies to every chain.
  if (n_chains > 0 && g_gradient_mode != GradientMode::NUMERICAL) {
    bool grad_ok = verify_gradient_runtime(q_init_per_chain[0], data, layout, 1e-4);
    if (!grad_ok) {
      g_gradient_mode = GradientMode::NUMERICAL;
      REprintf("[tulpa] Falling back to numerical gradients for all chains.\n");
    }
  }

  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Thread-safe autodiff: each chain creates its own tape via TapeScope (RAII),
  // so all gradient modes (N, A, A_t, H) run in parallel.
#ifdef _OPENMP
  if (n_chains > 1) {
    #pragma omp parallel for schedule(static) num_threads(n_chains)
    for (int c = 0; c < n_chains; c++) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init_per_chain[c], data, layout,
        n_iter, n_warmup, L, c, seed, false, max_treedepth,
        metric_type, adapt_delta, riemannian, metric_for(c)
      );
    }
  } else {
    cpp_results[0] = run_hmc_chain_cpp(
      q_init_per_chain[0], data, layout,
      n_iter, n_warmup, L, 0, seed, verbose, max_treedepth,
      metric_type, adapt_delta, riemannian, metric_for(0)
    );
  }
#else
  for (int c = 0; c < n_chains; c++) {
    cpp_results[c] = run_hmc_chain_cpp(
      q_init_per_chain[c], data, layout,
      n_iter, n_warmup, L, c, seed, verbose, max_treedepth,
      metric_type, adapt_delta, riemannian, metric_for(c)
    );
  }
#endif

  if (verbose && n_chains > 1) {
    // Verbose was disabled during the parallel run; report per-chain now.
    for (int c = 0; c < n_chains; c++) {
      int n_div = 0;
      for (int i = 0; i < cpp_results[c].n_sample; i++) {
        n_div += cpp_results[c].divergent[i];
      }
      Rcpp::Rcerr << "Chain " << (c + 1) << " complete. "
                  << "Divergent: " << n_div << std::endl;
    }
  }

  return cpp_results;
}

// =====================================================================
// Rcpp-returning wrapper: broadcast a single q_init across n_chains
// (the classic fresh-fit signature) and convert to R result structs.
// =====================================================================
std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  std::vector<std::vector<double>> q_init_per_chain(n_chains, q_init);
  std::vector<HMCResultCpp> cpp_results = run_hmc_parallel_chains_cpp(
    q_init_per_chain, {}, data,
    n_iter, n_warmup, L, n_chains, seed, verbose,
    max_treedepth, metric_type, adapt_delta, riemannian
  );

  int n_params = compute_param_layout(data).total_params;
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);
  }
  return results;
}

}  // namespace tulpa_hmc
