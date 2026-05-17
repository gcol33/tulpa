// nested_laplace_joint_multi.h
// Shared driver for joint multi-likelihood nested Laplace with one or more
// latent prior blocks.
//
// This is the joint analogue of run_multi_block_nested_laplace (single-arm,
// see nested_laplace_multi.h). For each outer-grid point k the inner Newton
// solves
//
//   eta_{k_arm,i} = X_{k_arm} beta_{k_arm} + RE_{k_arm}[g(i)]
//                 + Σ_b arm_scale_b(k_arm, k) * d_fac_b(k)
//                       * x[start_b + idx_b(i, k_arm) - 1]
//
//   grad/H from per-arm scatter (β/β, β/RE, RE/RE diagonal blocks
//          per arm, plus latent/{β, RE, latent} cross-terms per block)
//          + Σ_b add_prior_b(k)
//          + add_per_arm_beta_re_priors
//
//   center: each block's centerer applied to its sub-vector, with each
//           arm's first beta column shifted by
//           arm_scale_b(k_arm, k) * d_fac_b(k) * delta_b
//           to preserve eta after centering rank-deficient blocks.
//
//   log_prior: Σ_b log_prior_b(k) + log_prior_per_arm_re(x, parsed)
//
// Per-block prep is invoked once at grid point k before the inner solve. If
// any block reports infeasible (e.g. proper-CAR with rho outside the PD
// interval), the inner solve short-circuits with log_marginal = -inf.

#ifndef TULPA_NESTED_LAPLACE_JOINT_MULTI_H
#define TULPA_NESTED_LAPLACE_JOINT_MULTI_H

#include "laplace_core.h"
#include "laplace_family_link.h"
#include "laplace_newton_joint.h"
#include "laplace_re_priors.h"
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Per-observation latent-block scatter for one arm at one grid point.
//
// Variable-length analogue of the single-arm multi-block scatter
// (accumulate_latent_cross_terms in nested_laplace_multi.h), with the
// β/β and β/RE diagonal blocks evaluated *per arm* using ParsedArm
// offsets so multiple likelihood arms can share the same latent vector.
// Each block's eta contribution carries an optional per-arm scaling
// factor (arm_scale) for INLA `copy=` semantics.
//
// Accumulates β/RE/RE×β diagonal blocks (per arm, unchanged from
// joint_core), then for each obs i resolves the active subset of blocks
// (those with idx_b(i, k_arm) in [1, size_b]) and adds the latent
// gradient and β×latent / RE×latent / latent×latent Hessian cross-terms
// with the effective coefficient d_eff_b = arm_scale_b(k_arm, k_grid) *
// d_fac_b(k_grid).
inline void scatter_arm_obs_joint_multi(
    const Rcpp::NumericVector& /*x*/,
    const Rcpp::NumericVector&    eta,
    const ParsedArm&              pa,
    const JointArm&               arm,
    int                           k_arm,
    const std::vector<LatentBlock>& blocks,
    int                           k_grid,
    DenseVec&                     grad,
    DenseMat&                     H
) {
    const int p_k      = pa.p;
    const int n_re_k   = pa.n_re_groups;
    const int bstart   = pa.beta_start;
    const int rstart   = pa.re_start;
    const std::string& family = arm.family;
    const double phi_disp     = arm.phi;
    const int B = static_cast<int>(blocks.size());

    // Cache per-block effective coefficient for this (k_arm, k_grid).
    std::vector<double> d_eff_cache(B);
    for (int b = 0; b < B; b++) {
        double s = blocks[b].arm_scale
                    ? blocks[b].arm_scale(k_arm, k_grid)
                    : 1.0;
        d_eff_cache[b] = s * blocks[b].d_fac(k_grid);
    }

    std::vector<int>    active_idx;
    std::vector<double> active_d;
    active_idx.reserve(B);
    active_d.reserve(B);

    for (int i = 0; i < arm.N; i++) {
        auto gh = grad_hess_for_family(
            arm.y[i], arm.n_trials[i], eta[i], family, phi_disp);

        int g_re = -1;
        if (n_re_k > 0) {
            int gi = static_cast<int>(pa.re_idx[i]) - 1;
            if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
        }

        // Resolve active blocks for obs i. -1 from idx means "this obs
        // doesn't see this block" (e.g. an obs with no spatial unit).
        active_idx.clear();
        active_d.clear();
        for (int b = 0; b < B; b++) {
            int l_b = blocks[b].idx(i, k_arm);
            if (l_b > 0 && l_b <= blocks[b].size) {
                active_idx.push_back(blocks[b].start + l_b - 1);
                active_d.push_back(d_eff_cache[b]);
            }
        }
        const int A = static_cast<int>(active_idx.size());

        // β block: gradient + diagonal-block Hessian + cross with RE and
        // every active latent block.
        for (int j = 0; j < p_k; j++) {
            const double Xij = pa.X(i, j);
            grad[bstart + j] += gh.grad * Xij;
            for (int l = 0; l < p_k; l++) {
                H[bstart + j][bstart + l] += gh.neg_hess * Xij * pa.X(i, l);
            }
            if (g_re >= 0) {
                H[bstart + j][g_re] += gh.neg_hess * Xij;
                H[g_re][bstart + j] += gh.neg_hess * Xij;
            }
            for (int a = 0; a < A; a++) {
                H[bstart + j][active_idx[a]] += gh.neg_hess * Xij * active_d[a];
                H[active_idx[a]][bstart + j] += gh.neg_hess * Xij * active_d[a];
            }
        }

        // RE block: gradient + diagonal + cross with active latent indices.
        if (g_re >= 0) {
            grad[g_re] += gh.grad;
            H[g_re][g_re] += gh.neg_hess;
            for (int a = 0; a < A; a++) {
                H[g_re][active_idx[a]] += gh.neg_hess * active_d[a];
                H[active_idx[a]][g_re] += gh.neg_hess * active_d[a];
            }
        }

        // Latent x latent block (intra-block + inter-block). Includes both
        // the diagonal at (idx_a, idx_a) and the off-diagonal (idx_a, idx_b)
        // for a != b.
        for (int a = 0; a < A; a++) {
            grad[active_idx[a]] += gh.grad * active_d[a];
            for (int b = 0; b < A; b++) {
                H[active_idx[a]][active_idx[b]] +=
                    gh.neg_hess * active_d[a] * active_d[b];
            }
        }
    }
}

// Outer-grid driver. n_x_after_re is the latent dimension after all per-arm
// (β + RE) blocks; each LatentBlock's start field must point above that
// offset (typically built by appending sizes as blocks are constructed).
//
// `prep_at_grid` is an optional per-grid-point callback that runs before
// block.prep and the inner Newton at each outer-grid index. Joint kernels
// use it to apply per-grid dispersion overrides on `arms` (e.g.
// phi_grid_per_arm rewrites arm.phi for the current outer-grid index) or
// any other grid-dependent state that doesn't fit cleanly inside a
// LatentBlock callback. Pass `nullptr` (default) to disable.
inline Rcpp::List run_multi_block_nested_laplace_joint(
    int                              n_grid,
    std::vector<JointArm>&           arms,
    const std::vector<ParsedArm>&    parsed,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x_after_re,
    int                              max_iter,
    double                           tol,
    int                              n_threads,
    bool                             store_modes,
    const Rcpp::NumericVector&       x_init,
    bool                             store_Q = false,
    std::function<void(int)>         prep_at_grid = nullptr
) {
    const int n_arms = static_cast<int>(arms.size());
    if (static_cast<int>(parsed.size()) != n_arms) {
        Rcpp::stop("parsed and arms vectors must have the same length.");
    }

    int n_x = n_x_after_re;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }

    SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k_grid, const Rcpp::NumericVector& prev_mode)
        -> LaplaceResult
    {
        if (prep_at_grid) prep_at_grid(k_grid);
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(k_grid)) {
                LaplaceResult bad;
                bad.mode = (prev_mode.size() == n_x) ? prev_mode :
                           Rcpp::NumericVector(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        // Cache per-block (k_arm, k_grid) -> d_eff. Per-block d_fac(k_grid)
        // is evaluated once; per-arm scaling is re-evaluated inside the
        // per-arm loops because compute_eta is called from inside the
        // Newton step many times.
        const int B = static_cast<int>(blocks.size());
        std::vector<double> d_fac_cache(B);
        for (int b = 0; b < B; b++) {
            d_fac_cache[b] = blocks[b].d_fac(k_grid);
        }

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                const ParsedArm& pa = parsed[k_arm];
                const int N_k    = arms[k_arm].N;
                const int p_k    = pa.p;
                const int n_re_k = pa.n_re_groups;
                const int bstart = pa.beta_start;
                const int rstart = pa.re_start;

                // Per-arm effective coefficients per block.
                std::vector<double> d_eff(B);
                for (int b = 0; b < B; b++) {
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    d_eff[b] = s * d_fac_cache[b];
                }

                #ifdef _OPENMP
                #pragma omp parallel for schedule(static) \
                    num_threads(n_threads > 0 ? n_threads : 1)
                #endif
                for (int i = 0; i < N_k; i++) {
                    double e = 0.0;
                    for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
                    if (n_re_k > 0) {
                        int g = static_cast<int>(pa.re_idx[i]) - 1;
                        if (g >= 0 && g < n_re_k) e += x[rstart + g];
                    }
                    for (int b = 0; b < B; b++) {
                        int l = blocks[b].idx(i, k_arm);
                        if (l > 0 && l <= blocks[b].size) {
                            e += d_eff[b] * x[blocks[b].start + l - 1];
                        }
                    }
                    etas[k_arm][i] = e;
                }
            }
        };

        auto scatter_joint = [&](const Rcpp::NumericVector& x,
                                 const std::vector<Rcpp::NumericVector>& etas,
                                 DenseVec& grad, DenseMat& H) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                scatter_arm_obs_joint_multi(
                    x, etas[k_arm], parsed[k_arm], arms[k_arm], k_arm,
                    blocks, k_grid, grad, H
                );
            }
            for (const auto& b : blocks) {
                if (b.add_prior) b.add_prior(grad, H, x, k_grid);
            }
            add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center_joint = [&](Rcpp::NumericVector& x) {
            for (int b = 0; b < B; b++) {
                if (!blocks[b].center) continue;
                double c_b = blocks[b].center(x);
                if (std::abs(c_b) < 1e-15) continue;
                // Per-arm intercept compensation so eta is preserved when a
                // rank-deficient block is re-centered after a Newton step.
                // arm k's first beta column absorbs the constant
                // arm_scale_b(k_arm, k_grid) * d_fac_b(k_grid) * c_b that
                // the centerer removed from x[block]. See the BYM2 / ICAR
                // joint kernel centerers in nested_laplace_joint.cpp for
                // the load-bearing rationale.
                for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                    if (parsed[k_arm].p == 0) continue;
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    x[parsed[k_arm].beta_start] += s * d_fac_cache[b] * c_b;
                }
            }
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                    const std::vector<Rcpp::NumericVector>&)
            -> double {
            double lp = log_prior_per_arm_re(x, parsed);
            for (const auto& b : blocks) {
                if (b.log_prior) lp += b.log_prior(x, k_grid);
            }
            return lp;
        };

        return laplace_newton_solve_joint(
            arms, n_x,
            max_iter, tol, n_threads,
            compute_eta_joint, scatter_joint, center_joint, log_prior_joint,
            prev_mode, &shared_solver,
            store_Q
        );
    };

    return run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes
    );
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_MULTI_H
