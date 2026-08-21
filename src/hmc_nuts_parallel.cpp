// hmc_nuts_parallel.cpp
// run_hmc_parallel_chains_cpp: pure-C++ OpenMP across-chain core.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <memory>

#include <tulpa/nested_progress.h>

#include "hmc_sampler.h"
#include "hmc_chain_checkpoint.h"
#include "omp_threads.h"

namespace tulpa_hmc {

// g_gradient_mode is declared in hmc_sampler_decls.h.

// Active NUTS progress reporter. Declared in
// hmc_sampler_decls.h; see there for the contract.
::tulpa_progress::GridProgress* g_active_grid_progress = nullptr;

std::unique_ptr<::tulpa_progress::GridProgress> make_nuts_progress(int total, int width) {
  SEXP optS = Rf_GetOption1(Rf_install("tulpa.nl_progress"));
  if (optS == R_NilValue || TYPEOF(optS) != VECSXP) return nullptr;
  Rcpp::List opt(optS);
  bool on = opt.containsElementNamed("progress") &&
            Rcpp::as<bool>(opt["progress"]);
  std::string file = opt.containsElementNamed("progress_file")
                       ? Rcpp::as<std::string>(opt["progress_file"])
                       : std::string();
  int every = opt.containsElementNamed("progress_every")
                ? Rcpp::as<int>(opt["progress_every"]) : 0;
  double throttle = opt.containsElementNamed("progress_throttle")
                      ? Rcpp::as<double>(opt["progress_throttle"]) : 2.0;
  if (!on && file.empty()) return nullptr;
  std::unique_ptr<::tulpa_progress::GridProgress> gp(
      new ::tulpa_progress::GridProgress("nuts", total, every, throttle,
                                         file, /*emit_console=*/on,
                                         /*unit=*/"iter"));
  gp->set_width(width > 0 ? width : 1);
  return gp;
}

// =====================================================================
// Pure-C++ across-chain core (OpenMP). Per-chain initial position and
// per-chain initial inverse-mass diagonal, so it serves both a fresh fit
// (caller broadcasts one init) and a warm-started resume (caller passes
// each chain's final_position + inv_metric from a previous run, with
// n_warmup = 0. Returns the pure-C++ results
// (no Rcpp types), so it is safe to call from the C ABI as well.
//
// inv_metric_per_chain may be empty (all chains use the structural
// default) or length n_chains (entry c may itself be empty for chain c).
// =====================================================================
// Rhat compares between-chain variance against within-chain variance, so it
// only reports on agreement between chains when the chains start overdispersed
// relative to the posterior. Chains that all start at the same point make the
// between-chain variance a measure of how far the sampler wandered from one
// origin, and a well-mixing sampler drives that toward the within-chain
// variance whether or not the chains found the same mode.
//
// So a broadcast start is dispersed here, at the one point every multi-chain
// entry funnels through. Chain 0 keeps the caller's position exactly, so a warm
// start is not thrown away; every other chain is offset per coordinate by
// Uniform(-2, 2) times that coordinate's own scale. The scale is the posterior
// SD implied by a caller-supplied inverse-mass diagonal (which is a variance),
// and 1 otherwise: a fixed absolute offset would be wrong across a layout that
// mixes coefficients, log-variances and field coordinates, and every coordinate
// here is on the sampler's unconstrained scale. The stream is keyed on
// (seed, chain), so a run stays reproducible under a fixed seed.
//
// A caller who supplied distinct rows is left alone, and so is a run with no
// warmup: there the reported draws start at the first iteration, and an
// overdispersed start would be reported rather than burnt in.
static void disperse_broadcast_inits(
    std::vector<std::vector<double>>& q_init_per_chain,
    const std::vector<std::vector<double>>& inv_metric_per_chain,
    int n_chains,
    int n_warmup,
    unsigned int seed
) {
  if (n_chains < 2 || n_warmup < 1) return;
  if ((int)q_init_per_chain.size() != n_chains) return;
  const std::vector<double> q0 = q_init_per_chain[0];
  const int n = (int)q0.size();
  if (n == 0) return;
  for (int c = 1; c < n_chains; c++) {
    if (q_init_per_chain[c] != q0) return;
  }

  const bool have_metric = ((int)inv_metric_per_chain.size() == n_chains);
  for (int c = 1; c < n_chains; c++) {
    std::mt19937 jitter_rng(seed + 7919u * static_cast<unsigned int>(c + 1));
    std::uniform_real_distribution<double> jitter(-2.0, 2.0);
    const bool scaled = have_metric && (int)inv_metric_per_chain[c].size() == n;
    for (int j = 0; j < n; j++) {
      double sd = 1.0;
      if (scaled && inv_metric_per_chain[c][j] > 0.0) {
        sd = std::sqrt(inv_metric_per_chain[c][j]);
      }
      q_init_per_chain[c][j] = q0[j] + sd * jitter(jitter_rng);
    }
  }
}

std::vector<HMCResultCpp> run_hmc_parallel_chains_cpp(
    const std::vector<std::vector<double>>& q_init_broadcast,
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
    int riemannian,
    const std::string& checkpoint_path,
    const ParamLayout* layout_override
) {
  // Honour a caller-supplied layout: a FullGradFn model carries
  // a minimal ModelData (total_params set by hand, no processes), so recomputing
  // here would size the chains' position buffers wrong and overrun. The
  // single-chain runner already threads the passed layout through.
  ParamLayout layout = layout_override ? *layout_override
                                       : compute_param_layout(data);

  if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");
  if ((int)q_init_broadcast.size() != n_chains) {
    Rcpp::stop("run_hmc_parallel_chains_cpp: %d initial positions for %d chains",
               (int)q_init_broadcast.size(), n_chains);
  }

  std::vector<std::vector<double>> q_init_per_chain = q_init_broadcast;
  disperse_broadcast_inits(q_init_per_chain, inv_metric_per_chain,
                           n_chains, n_warmup, seed);

  static const std::vector<double> kNoMetric;
  auto metric_for = [&](int c) -> const std::vector<double>& {
    return ((int)inv_metric_per_chain.size() == n_chains)
               ? inv_metric_per_chain[c]
               : kNoMetric;
  };

  // Any fallback below is scoped to this fit; the entry mode is restored on
  // return so it does not leak into the next fit.
  GradientModeFitScope grad_mode_scope;

  // Runtime gradient check on chain 0's init: compare the active gradient
  // against numerical BEFORE spawning parallel chains. Single-threaded here,
  // so the R API and g_gradient_mode mutation are safe; the decision then
  // applies to every chain.
  if (g_gradient_mode != GradientMode::NUMERICAL) {
    bool grad_ok = verify_gradient_runtime(q_init_per_chain[0], data, layout, 1e-4);
    if (!grad_ok) {
      g_gradient_mode = GradientMode::NUMERICAL;
      REprintf("[tulpa] Falling back to numerical gradients for all chains.\n");
    }
  }

  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Per-chain checkpoint/resume. A chain is the checkpoint
  // unit: deterministic in (seed, chain_id, data, settings), so a resumed chain
  // is bit-for-bit identical to the uninterrupted one. The fingerprint folds the
  // sampler settings, the seed, the per-chain init + metric, and the latent /
  // data dimensions, so a resume onto a file from a different fit errors. Done
  // chains are loaded SERIALLY here (has()/get() are unsynchronized reads); the
  // parallel loop then skips them and only the missing chains run + save()
  // (save() is mutex-guarded).
  std::unique_ptr<tulpa::ChainCheckpoint> ckpt;
  std::vector<unsigned char> done(n_chains, 0);
  if (!checkpoint_path.empty()) {
    tulpa::Fingerprint fp;
    fp.fold_str("nuts_chains");
    fp.fold_pod(n_iter);
    fp.fold_pod(n_warmup);
    fp.fold_pod(L);
    fp.fold_pod(n_chains);
    fp.fold_pod(seed);
    fp.fold_pod(max_treedepth);
    int mt = static_cast<int>(metric_type);
    fp.fold_pod(mt);
    fp.fold_pod(adapt_delta);
    fp.fold_pod(riemannian);
    fp.fold_pod(layout.total_params);
    fp.fold_pod(data.N);
    for (const auto& q : q_init_per_chain)   fp.fold_vec(q);
    for (const auto& m : inv_metric_per_chain) fp.fold_vec(m);
    ckpt.reset(new tulpa::ChainCheckpoint(
        checkpoint_path, fp.value(), tulpa::chain_checkpoint_keys(n_chains)));
    for (int c = 0; c < n_chains; c++) {
      if (ckpt->has(c)) { cpp_results[c] = ckpt->get(c); done[c] = 1; }
    }
  }

  // Shared NUTS progress reporter for this across-chain run.
  // Built on the main thread (reads the scoped option); each chain ticks the
  // same pointer under an omp critical inside run_hmc_chain_cpp. Console output
  // auto-suppresses in the parallel region (the heartbeat file is the
  // parallel/detached channel); a single serial chain keeps the console bar.
  std::unique_ptr<::tulpa_progress::GridProgress> shared_progress;
  if (!g_active_grid_progress) {
    shared_progress = make_nuts_progress(n_iter * n_chains, n_chains);
    if (shared_progress) g_active_grid_progress = shared_progress.get();
  }

  // Thread-safe autodiff: each chain creates its own tape via TapeScope (RAII),
  // so all gradient modes (N, A, A_t, H) run in parallel.
  //
  // This is the outermost region in a sampler run, so its team is the one that
  // has to respect the environment: OMP_NUM_THREADS / omp_set_num_threads()
  // through omp_get_max_threads(), OMP_THREAD_LIMIT, and the two-core cap R CMD
  // check sets through _R_CHECK_LIMIT_CORES_. A team narrower than n_chains
  // still runs every chain, several to a worker, and a chain is deterministic
  // in (seed, chain_id, data, settings), so the results do not depend on the
  // width.
#ifdef _OPENMP
  const int chain_team = tulpa_omp_team_size_req(n_chains, n_chains);
  if (chain_team > 1) {
    // Exception barrier: an exception escaping the omp for body (Rcpp::stop
    // from the log-posterior, bad_alloc from per-chain buffers) is
    // std::terminate. Each chain catches; the first message is re-raised
    // from the main thread after the region.
    std::atomic<bool> chain_failed{false};
    std::string       chain_err;
    #pragma omp parallel for schedule(static) num_threads(chain_team)
    for (int c = 0; c < n_chains; c++) {
      if (done[c]) continue;
      try {
        cpp_results[c] = run_hmc_chain_cpp(
          q_init_per_chain[c], data, layout,
          n_iter, n_warmup, L, c, seed, false, max_treedepth,
          metric_type, adapt_delta, riemannian, metric_for(c)
        );
        if (ckpt) ckpt->save(c, cpp_results[c]);
      } catch (const std::exception& e) {
        if (!chain_failed.exchange(true)) chain_err = e.what();
      } catch (...) {
        chain_failed.exchange(true);
      }
    }
    if (chain_failed.load()) {
      if (shared_progress) {
        g_active_grid_progress->finish();
        g_active_grid_progress = nullptr;
      }
      Rcpp::stop("NUTS chain failed: %s",
                 chain_err.empty() ? "unknown error" : chain_err.c_str());
    }
  } else {
    // A one-wide team runs the chains one after another. The console bar stays
    // on for a single chain; several chains report per-chain divergences after
    // the loop, the same channel the parallel route uses.
    for (int c = 0; c < n_chains; c++) {
      if (done[c]) continue;
      cpp_results[c] = run_hmc_chain_cpp(
        q_init_per_chain[c], data, layout,
        n_iter, n_warmup, L, c, seed, verbose && n_chains == 1, max_treedepth,
        metric_type, adapt_delta, riemannian, metric_for(c)
      );
      if (ckpt) ckpt->save(c, cpp_results[c]);
    }
  }
#else
  for (int c = 0; c < n_chains; c++) {
    if (done[c]) continue;
    cpp_results[c] = run_hmc_chain_cpp(
      q_init_per_chain[c], data, layout,
      n_iter, n_warmup, L, c, seed, verbose, max_treedepth,
      metric_type, adapt_delta, riemannian, metric_for(c)
    );
    if (ckpt) ckpt->save(c, cpp_results[c]);
  }
#endif

  if (shared_progress) {
    g_active_grid_progress->finish();
    g_active_grid_progress = nullptr;
  }

  if (verbose && n_chains > 1) {
    // Per-chain verbose is off for a multi-chain run; report per-chain now.
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

}  // namespace tulpa_hmc
