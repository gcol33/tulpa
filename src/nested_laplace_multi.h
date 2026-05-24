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

#include "laplace_core.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
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

// Per-observation latent-block scatter for the multi-block driver.
//
// Generalizes the BYM2 hand-rolled cross-term loop to a vector of blocks. For
// each observation i, resolves the active subset of blocks (those whose
// idx(i) is in [1, size]), then accumulates the latent grad and the latent-
// {self,β,RE} Hessian cross-terms with d_fac coefficients.
//
// β/RE x β/RE blocks are NOT touched here — those are produced by
// scatter_obs_grad_hess_base. This helper only adds the latent-side
// contributions.
//
// Serial across observations. The latent block count B is small (typically
// 1-3), so the inner cost is dominated by the base scatter's β x β block.
inline void accumulate_latent_cross_terms(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta,
    const std::string& family, double phi,
    const std::vector<LatentBlock>& blocks, int k,
    DenseVec& grad, DenseMat& H,
    const double* det_prob = nullptr
) {
    const int B = static_cast<int>(blocks.size());
    if (B == 0) return;

    std::vector<double> d_fac_cache(B);
    for (int b = 0; b < B; b++) d_fac_cache[b] = blocks[b].d_fac(k);

    std::vector<int> active_idx;
    std::vector<double> active_d;
    active_idx.reserve(B);
    active_d.reserve(B);

    for (int i = 0; i < N; i++) {
        active_idx.clear();
        active_d.clear();
        for (int b = 0; b < B; b++) {
            int l_b = blocks[b].idx(i, /*k_arm=*/0);
            if (l_b > 0 && l_b <= blocks[b].size) {
                active_idx.push_back(blocks[b].start + l_b - 1);
                active_d.push_back(d_fac_cache[b]);
            }
        }
        if (active_idx.empty()) continue;

        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi,
                                       det_prob ? det_prob[i] : 1.0);

        int re_g = -1;
        if (n_re_groups > 0) {
            int g = static_cast<int>(re_idx[i]) - 1;
            if (g >= 0 && g < n_re_groups) re_g = g;
        }

        const int A = static_cast<int>(active_idx.size());
        for (int a = 0; a < A; a++) {
            const int idx_a = active_idx[a];
            const double d_a = active_d[a];

            grad[idx_a] += gh.grad * d_a;

            for (int j = 0; j < p; j++) {
                double v = gh.neg_hess * X(i, j) * d_a;
                H[j][idx_a]     += v;
                H[idx_a][j]     += v;
            }
            if (re_g >= 0) {
                double v = gh.neg_hess * d_a;
                H[p + re_g][idx_a] += v;
                H[idx_a][p + re_g] += v;
            }

            for (int b = 0; b < A; b++) {
                H[idx_a][active_idx[b]] += gh.neg_hess * d_a * active_d[b];
            }
        }
    }
}

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
    const Rcpp::NumericVector& det_prob = Rcpp::NumericVector()
) {
    int n_x = p + n_re_groups;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    // Per-observation detection probability q_i = 1 - (1-p)^J for the
    // `occupancy` family (mu_i = q_i * sigma(eta_i)); nullptr for every other
    // family. Threaded into the likelihood scatter, the step-halving objective,
    // and the per-cell log-marginal so the inner solve fits the marginalized
    // occupancy state likelihood directly. With this family the converged
    // Hessian is already the marginal curvature, so the predictive variance is
    // read off the live factor with no rescaling.
    const double* det_prob_ptr =
        (det_prob.size() == N) ? &det_prob[0] : nullptr;

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

        auto compute_eta = [&](const Rcpp::NumericVector& x,
                               Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) \
                num_threads(n_threads_inner_eff > 0 ? n_threads_inner_eff : 1) \
                if(n_threads_inner_eff > 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = static_cast<int>(re_idx[i]) - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                for (size_t b = 0; b < blocks.size(); b++) {
                    int l = blocks[b].idx(i, /*k_arm=*/0);
                    if (l > 0 && l <= blocks[b].size) {
                        eta[i] += d_fac_cache[b] * x[blocks[b].start + l - 1];
                    }
                }
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x,
                           const Rcpp::NumericVector& eta,
                           DenseVec& grad, DenseMat& H) {
            scatter_obs_grad_hess_base(y, n_trials, X, re_idx,
                                        N, p, n_re_groups,
                                        eta, family, phi,
                                        grad, H, n_threads_inner_eff,
                                        det_prob_ptr);
            accumulate_latent_cross_terms(y, n_trials, X, re_idx,
                                           N, p, n_re_groups,
                                           eta, family, phi,
                                           blocks, k, grad, H,
                                           det_prob_ptr);
            for (const auto& b : blocks) {
                if (b.add_prior) b.add_prior(grad, H, x, k);
            }
            add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            for (const auto& b : blocks) {
                if (b.center) b.center(x);
            }
        };

        auto log_prior = [&](const Rcpp::NumericVector& x,
                             const Rcpp::NumericVector&) {
            double lp = compute_log_prior_re(x, p, n_re_groups, tau_re);
            for (const auto& b : blocks) {
                if (b.log_prior) lp += b.log_prior(x, k);
            }
            return lp;
        };

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif

        NewtonScratch& scratch = scratch_override ? *scratch_override
                                                  : scratch_pool[tid];

        LaplaceResult res = laplace_newton_solve(
            y, n_trials, family, phi, N, n_x,
            max_iter_use, tol, n_threads_inner_eff,
            compute_eta, scatter, center, log_prior,
            scratch, prev_mode, solver, store_Q,
            /*inv_block_layout=*/nullptr, det_prob_ptr
        );

        // Per-row predictive variance, var(eta_i | theta_k) = a_i' H^{-1} a_i,
        // read off the live Cholesky factor laplace_newton_solve left resident
        // at the converged mode (sparse path solves the CHOLMOD factor in
        // `solver`, dense path back-substitutes scratch.chol.L) -- N back-solves,
        // no refactorization. H is the fitting Hessian at the mode; for the
        // `occupancy` family that is already the marginal curvature (the
        // expected information q*sigma*(1-sigma)^2/(1-q*sigma)), so no rescaling
        // is needed -- the calibrated variance falls out directly.
        if (want_var && std::isfinite(res.log_marginal) &&
            static_cast<std::size_t>(k + 1) * N <= fitted_var_buf.size()) {
            const bool use_sparse = (n_x >= SPARSE_THRESHOLD);
            const bool used_sparse_factor =
                use_sparse && solver && solver->factored();
            std::vector<double> a(n_x, 0.0), z(n_x, 0.0), zwork;
            if (!used_sparse_factor) zwork.assign(n_x, 0.0);
            const std::size_t base = static_cast<std::size_t>(k) * N;
            for (int i = 0; i < N; i++) {
                std::fill(a.begin(), a.end(), 0.0);
                for (int j = 0; j < p; j++) a[j] = X(i, j);
                if (n_re_groups > 0) {
                    int g = static_cast<int>(re_idx[i]) - 1;
                    if (g >= 0 && g < n_re_groups) a[p + g] += 1.0;
                }
                for (size_t b = 0; b < blocks.size(); b++) {
                    int l = blocks[b].idx(i, /*k_arm=*/0);
                    if (l > 0 && l <= blocks[b].size) {
                        a[blocks[b].start + l - 1] += d_fac_cache[b];
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

    // Cheap-pass screening (Phase 3, dev_notes/speedup.md): one Newton step
    // from the pilot mode, return the Laplace log-marginal at that quasi-
    // mode. Same trick as the joint kernel — calling solve_at_theta_impl
    // with max_iter=1 reuses all the per-cell callbacks without
    // duplicating scatter/eta logic. Dedicated thread-local solver +
    // scratch keep cheap_eval independent of the parallel fan-out's pool.
    SparseCholeskySolver cheap_solver;
    NewtonScratch cheap_scratch;
    cheap_scratch.allocate(n_x, N);
    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& x_pilot) -> double {
        LaplaceResult r = solve_at_theta_impl(
            k_grid, x_pilot, &cheap_solver,
            /*max_iter_use=*/1, &cheap_scratch);
        return r.log_marginal;
    };

    Rcpp::List out = run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        cheap_eval, prune_tol
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
                    int l = blocks[b].idx(i, /*k_arm=*/0);
                    if (l > 0 && l <= blocks[b].size) {
                        e += dfac[b] * modes(k, blocks[b].start + l - 1);
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
