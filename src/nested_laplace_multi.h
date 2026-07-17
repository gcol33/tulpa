// nested_laplace_multi.h
// Shared helpers for the multi-block nested-Laplace driver. Used by both
// nested_laplace.cpp (single-block kernels routing through length-1
// LatentBlock vectors) and nested_laplace_multi.cpp (genuine multi-block
// dispatch from R).
//
// The driver assembles the inner solve at each outer-grid point k as:
//
//   eta_i  = X_i β + RE_g(i) + Σ_b d_fac_b(k) * x[start_b + idx_b(i) - 1]
//   grad/H from scatter_obs_grad_hess_base (β, RE)
//          + accumulate_latent_cross_terms (latent, latent x β, latent x RE)
//          + Σ_b add_prior_b(k)
//          + add_re_beta_priors
//   center : each block's centerer applied to its sub-vector
//   log_prior = Σ_b log_prior_b(k) + compute_log_prior_re
//
// Per-block prep is invoked once at grid point k before the inner solve; if
// any block reports infeasible (e.g. proper CAR with rho outside the PD
// interval), the inner solve short-circuits with log_marginal = -inf.

#ifndef TULPA_NESTED_LAPLACE_MULTI_H
#define TULPA_NESTED_LAPLACE_MULTI_H

#include "laplace_builtin_family_spec.h"  // builtin_family_spec, BuiltinFamilyResponse
#include "laplace_core.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_spec_solve.h"           // spec_inner_solve (the unified inner solve)
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Generic outer-grid driver over a vector of LatentBlocks.
//
// `n_threads_outer` controls the outer-grid parallelism (1 = serial, the
// pre-refactor default). `n_threads_inner` controls per-observation parallel
// kernels (compute_eta, scatter, log-lik reduction) inside each cell. When
// `n_threads_outer > 1`, the driver pre-allocates n_threads_outer worth of
// NewtonScratch (one per OpenMP thread) and one SparseCholeskySolver per
// thread, then parallelises the grid via run_nested_laplace_grid.
inline Rcpp::List run_multi_block_nested_laplace(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const std::vector<LatentBlock>& blocks,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    bool store_Q = false,
    int n_threads_outer = 1,
    double prune_tol = 0.0,
    const LikelihoodSpec* ext_spec = nullptr,
    void* ext_response = nullptr,
    tulpa_progress::GridProgress* progress = nullptr,
    GridCheckpoint* ckpt = nullptr
) {
    int n_x = p + n_re_groups;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    // ---- Likelihood: built-in family or model-supplied spec -----------------
    // Each single-block (and np==1 multi-block) nested kernel routes its inner
    // Laplace solve through spec_inner_solve, which reads the per-observation
    // score / Fisher weight / log-lik entirely from a single-process
    // LikelihoodSpec; scatter_spec assembles the latent gradient + Hessian. The
    // driver therefore carries no copy of the obs + latent-cross scatter.
    //
    // By default tulpa builds builtin_family_spec(family) over (y, n_trials,
    // family, phi). A model package (tulpaObs occupancy, tulpaGlmm custom
    // families) instead supplies its own spec + response via ext_spec /
    // ext_response (passed from R as an XPtr<NestedLikelihood>); then `family` /
    // `phi` / `y` / `n_trials` are unused here and the likelihood -- including any
    // per-observation scaling, e.g. the marginalized occupancy state -- lives
    // wholly in the model's spec. Built once; read-only inside the (possibly
    // parallel) grid. sigma_beta = 100 makes the spec's beta ridge tau_beta =
    // 1e-4, identical to the nested kernel's DEFAULT_TAU_BETA; the spec also
    // folds the beta-prior log-density into the log-marginal.
    const bool has_re = (n_re_groups > 0) && (re_idx.size() == N);

    ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < p; j++)
            proc.X_flat[(size_t)i * p + j] = X(i, j);

    LikelihoodSpec builtin_spec;
    std::vector<int> n_trials_vec;
    BuiltinFamilyResponse resp;
    const LikelihoodSpec* spec_ptr = ext_spec;
    void*                 resp_ptr = ext_response;
    if (!ext_spec) {
        builtin_spec  = builtin_family_spec(family);
        n_trials_vec.assign(n_trials.begin(), n_trials.end());
        resp.y        = y.begin();
        resp.n_trials = n_trials_vec.data();
        resp.N        = N;
        resp.family   = family;
        resp.phi      = phi;
        spec_ptr      = &builtin_spec;
        resp_ptr      = &resp;
    }

    ModelData data;
    data.n_processes         = 1;
    data.processes.push_back(proc);
    data.N                   = N;
    data.sigma_beta          = 100.0;   // tau_beta = 1e-4 == DEFAULT_TAU_BETA
    data.likelihood_spec     = spec_ptr;
    data.model_response_data = resp_ptr;
    data.sharing.init(1);
    std::vector<int> re_group_1based;
    if (has_re) {
        data.n_re_groups = n_re_groups;
        re_group_1based.resize(N);
        for (int i = 0; i < N; i++) re_group_1based[i] = static_cast<int>(re_idx[i]);
        data.re_group = re_group_1based;
    }

    ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    if (has_re) {
        layout.has_re           = true;
        layout.re_start         = p;
        layout.re_end           = p + n_re_groups;
        layout.log_sigma_re_idx = n_x;   // hyperparam slot AFTER all latent
    }
    const int total_params = n_x + (has_re ? 1 : 0);
    layout.total_params = total_params;
    const double log_sigma_re = std::log(sigma_re);

    // Per-outer-thread NewtonScratch. omp_get_thread_num() returns 0 outside
    // parallel regions so the serial path correctly picks scratch_pool[0].
    int n_outer = std::max(1, n_threads_outer);
    std::vector<NewtonScratch> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, N);

    // When the outer grid is parallel, force inner kernels to single-thread
    // OpenMP (compute_eta / scatter / log-lik reduction). Nested OpenMP at
    // this problem size is overhead-dominated — see
    // dev_notes/issue_body_adaptive_grid_cost.md for the empirical sweep.
    int n_threads_inner_eff = (n_outer > 1) ? 1 : n_threads;

    // Per-cell per-row predictive variance of the linear predictor,
    // var(eta_i | theta_k) = a_i' H_k^{-1} a_i, where a_i is the row's loading
    // vector (the same beta + RE + sum_b d_fac_b * e_{idx_b(i)} accumulation as
    // compute_eta, but as coefficients rather than multiplied by the mode). It
    // is read off the live Cholesky factor laplace_newton_solve leaves resident
    // at the converged mode (the same factor inv_block_layout reuses for the RE
    // posterior-covariance blocks), so there is no refactorization. Filled only
    // on the full solves (not the max_iter=1 cheap screen) and only when modes
    // are stored; packed into `fitted_eta_var` [n_grid x N] below so callers can
    // marginalise psi intervals over the hyperparameter grid. Buffer is
    // k-sliced (disjoint per cell) so the outer-parallel writes do not race.
    std::vector<double> fitted_var_buf(
        store_modes ? static_cast<std::size_t>(n_grid) * N : 0, 0.0);

    // Inner implementation: takes max_iter as a parameter so the cheap-pass
    // path can call with max_iter=1 for a one-Newton-step screen. See the
    // joint analogue in nested_laplace_joint_multi.h for the rationale.
    auto solve_at_theta_impl = [&](int k,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* solver,
                                   int max_iter_use,
                                   NewtonScratch* scratch_override
                                   = nullptr,
                                   bool want_var = false) -> LaplaceResult
    {
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(k)) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                           ? prev_mode
                           : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        std::vector<double> d_fac_cache(blocks.size());
        for (size_t b = 0; b < blocks.size(); b++) {
            d_fac_cache[b] = blocks[b].d_fac(k);
        }

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif

        NewtonScratch& scratch = scratch_override ? *scratch_override
                                                  : scratch_pool[tid];

        // base_params: latent warm start in [0, n_x); log_sigma_re hyperparam
        // after. spec_inner_solve builds the [beta | RE | blocks] layout,
        // wraps the spec helpers as the shared Newton loop's closures, and leaves
        // the live Cholesky factor resident in scratch/solver for the predictive-
        // variance back-solves below.
        std::vector<double> base_params(total_params, 0.0);
        if (static_cast<int>(prev_mode.size()) == n_x)
            std::copy(prev_mode.begin(), prev_mode.end(), base_params.begin());
        if (has_re) base_params[layout.log_sigma_re_idx] = log_sigma_re;

        LaplaceResult res = spec_inner_solve(
            data, layout, &blocks, k, *spec_ptr, resp_ptr, re_group_1based,
            max_iter_use, tol, n_threads_inner_eff, base_params,
            scratch, solver, store_Q, /*inv_block_layout=*/nullptr
        );

        // Per-row predictive variance, var(eta_i | theta_k) = a_i' H^{-1} a_i,
        // read off the live Cholesky factor laplace_newton_solve left resident
        // at the converged mode (sparse path solves the CHOLMOD factor in
        // `solver`, dense path back-substitutes scratch.chol.L) -- N back-solves,
        // no refactorization. H is the spec's Fisher curvature at the mode; when
        // the model spec already returns the marginal information (e.g. tulpaObs'
        // occupancy q*sigma*(1-sigma)^2/(1-q*sigma)), no rescaling is needed --
        // the calibrated variance falls out directly.
        if (want_var && std::isfinite(res.log_marginal) &&
            static_cast<std::size_t>(k + 1) * N <= fitted_var_buf.size()) {
            const bool use_sparse = (n_x >= SPARSE_THRESHOLD);
            const bool used_sparse_factor =
                use_sparse && solver && solver->factored();
            std::vector<double> a(n_x, 0.0), z(n_x, 0.0), zwork;
            if (!used_sparse_factor) zwork.assign(n_x, 0.0);
            std::vector<std::pair<int,double>> a_multi;
            const std::size_t base = static_cast<std::size_t>(k) * N;
            for (int i = 0; i < N; i++) {
                std::fill(a.begin(), a.end(), 0.0);
                for (int j = 0; j < p; j++) a[j] = X(i, j);
                if (n_re_groups > 0) {
                    int g = static_cast<int>(re_idx[i]) - 1;
                    if (g >= 0 && g < n_re_groups) a[p + g] += 1.0;
                }
                for (size_t b = 0; b < blocks.size(); b++) {
                    if (blocks[b].contrib_kind
                            == BlockContribKind::INDEXED_MULTI) {
                        blocks[b].obs_indices(i, /*k_arm=*/0, a_multi);
                        for (const auto& nw : a_multi) {
                            int l = nw.first;
                            if (l > 0 && l <= blocks[b].size) {
                                a[blocks[b].start + l - 1] +=
                                    d_fac_cache[b] * nw.second;
                            }
                        }
                    } else {
                        int l = blocks[b].idx(i, /*k_arm=*/0);
                        if (l > 0 && l <= blocks[b].size) {
                            double w = blocks[b].row_weight
                                       ? blocks[b].row_weight(i, /*k_arm=*/0)
                                       : 1.0;
                            a[blocks[b].start + l - 1] += d_fac_cache[b] * w;
                        }
                    }
                }
                bool ok = true;
                if (used_sparse_factor) {
                    solver->solve(a.data(), z.data(), n_x);
                } else {
                    ok = chol_substitute_raw(scratch.chol.L.data(), n_x,
                                             a.data(), z.data(), zwork.data());
                }
                double v = 0.0;
                if (ok) for (int j = 0; j < n_x; j++) v += a[j] * z[j];
                fitted_var_buf[base + i] = (ok && v > 0.0) ? v : 0.0;
            }
        }

        return res;
    };

    // 3-arg adapter for run_nested_laplace_grid.
    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k, prev_mode, solver, max_iter, nullptr,
                                   /*want_var=*/store_modes);
    };

    // Cheap-pass screening: a short inner Newton run warm-started from the
    // neighbour quasi-mode the driver chains across the lattice, returning
    // the quasi-mode and the Laplace log-marginal at it. Calling
    // solve_at_theta_impl with the driver-supplied `n_steps` reuses all the
    // per-cell callbacks without duplicating scatter/eta logic. Dedicated
    // thread-local solver + scratch keep cheap_eval independent of the
    // parallel fan-out's pool.
    // One cheap solver + scratch per outer worker slot: the cheap screen may
    // run per-tile chains concurrently, and CHOLMOD's
    // cholmod_common is not thread-safe, so each worker needs its own. This
    // path passes no tile metadata, so the screen stays serial (worker 0), but
    // the pool keeps the CheapEval contract uniform across call sites.
    const int n_cheap_workers = std::max(1, n_outer);
    std::vector<SparseCholeskySolver> cheap_solvers(n_cheap_workers);
    std::vector<NewtonScratch> cheap_scratches(n_cheap_workers);
    for (auto& cs : cheap_scratches) cs.allocate(n_x, N);
    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& warm,
                          int n_steps, int worker) -> LaplaceResult {
        return solve_at_theta_impl(
            k_grid, warm, &cheap_solvers[worker], n_steps,
            &cheap_scratches[worker]);
    };

    Rcpp::List out = run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        cheap_eval, prune_tol, progress, ckpt
    );

    // Per-row fitted linear predictor at every grid cell, reconstructed from the
    // stored modes with the SAME accumulation as the inner solve's compute_eta
    // (beta + RE + sum_b d_fac_b(k) * x_block[idx_b(i)]). `run_nested_laplace_
    // grid` stores modes row k for grid cell k, so d_fac(k) aligns. Returned as
    // `fitted_eta` [n_grid x N] so callers can marginalise mu/psi over the
    // hyperparameter grid for every prior -- including those whose predictor
    // mixes components with hyperparameter-dependent scales (e.g. BYM2) -- and
    // for held-out (n_trials = 0) rows, without re-deriving each prior's scales.
    if (store_modes && out.containsElementNamed("modes")) {
        Rcpp::NumericMatrix modes = out["modes"];
        int ng = modes.nrow();
        Rcpp::NumericMatrix fitted_eta(ng, N);
        std::vector<double> dfac(blocks.size());
        std::vector<std::pair<int,double>> e_multi;
        for (int k = 0; k < ng; k++) {
            for (size_t b = 0; b < blocks.size(); b++) dfac[b] = blocks[b].d_fac(k);
            for (int i = 0; i < N; i++) {
                double e = 0.0;
                for (int j = 0; j < p; j++) e += X(i, j) * modes(k, j);
                if (n_re_groups > 0) {
                    int g = static_cast<int>(re_idx[i]) - 1;
                    if (g >= 0 && g < n_re_groups) e += modes(k, p + g);
                }
                for (size_t b = 0; b < blocks.size(); b++) {
                    if (blocks[b].contrib_kind
                            == BlockContribKind::INDEXED_MULTI) {
                        blocks[b].obs_indices(i, /*k_arm=*/0, e_multi);
                        for (const auto& nw : e_multi) {
                            int l = nw.first;
                            if (l > 0 && l <= blocks[b].size) {
                                e += dfac[b] * nw.second
                                   * modes(k, blocks[b].start + l - 1);
                            }
                        }
                    } else {
                        int l = blocks[b].idx(i, /*k_arm=*/0);
                        if (l > 0 && l <= blocks[b].size) {
                            double w = blocks[b].row_weight
                                       ? blocks[b].row_weight(i, /*k_arm=*/0)
                                       : 1.0;
                            e += dfac[b] * w * modes(k, blocks[b].start + l - 1);
                        }
                    }
                }
                fitted_eta(k, i) = e;
            }
        }
        out["fitted_eta"] = fitted_eta;

        // Per-cell predictive variance of eta, var(eta_i | theta_k) = a_i'
        // H_k^{-1} a_i, filled by the solves above (0 for pruned cells, which
        // carry zero grid weight). Paired with `fitted_eta` so callers can
        // marginalise the per-row eta posterior as a Gaussian mixture over the
        // grid (mean fitted_eta[k,i], variance fitted_eta_var[k,i], weight w_k)
        // and map monotone through plogis for calibrated psi intervals.
        if (fitted_var_buf.size() ==
            static_cast<std::size_t>(ng) * N) {
            Rcpp::NumericMatrix fitted_eta_var(ng, N);
            for (int k = 0; k < ng; k++) {
                const std::size_t base = static_cast<std::size_t>(k) * N;
                for (int i = 0; i < N; i++) {
                    fitted_eta_var(k, i) = fitted_var_buf[base + i];
                }
            }
            out["fitted_eta_var"] = fitted_eta_var;
        }
    }
    return out;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_MULTI_H
