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
    DenseVec& grad, DenseMat& H
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

        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

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
    double prune_tol = 0.0
) {
    int n_x = p + n_re_groups;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

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

    // Inner implementation: takes max_iter as a parameter so the cheap-pass
    // path can call with max_iter=1 for a one-Newton-step screen. See the
    // joint analogue in nested_laplace_joint_multi.h for the rationale.
    auto solve_at_theta_impl = [&](int k,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* solver,
                                   int max_iter_use,
                                   NewtonScratch* scratch_override
                                   = nullptr) -> LaplaceResult
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
                                        grad, H, n_threads_inner_eff);
            accumulate_latent_cross_terms(y, n_trials, X, re_idx,
                                           N, p, n_re_groups,
                                           eta, family, phi,
                                           blocks, k, grad, H);
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

        return laplace_newton_solve(
            y, n_trials, family, phi, N, n_x,
            max_iter_use, tol, n_threads_inner_eff,
            compute_eta, scatter, center, log_prior,
            scratch, prev_mode, solver, store_Q
        );
    };

    // 3-arg adapter for run_nested_laplace_grid.
    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k, prev_mode, solver, max_iter, nullptr);
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

    return run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        cheap_eval, prune_tol
    );
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_MULTI_H
