// laplace_spec.cpp
// LikelihoodSpec-driven Laplace mode finder.
//
// Mirrors the laplace_mode_dense pattern in laplace_core.cpp but routes
// the per-observation log-likelihood and IRLS weights through a
// LikelihoodSpec rather than the family-enum dispatch in
// laplace_family_link.h. Lets a downstream package (tulpaGlmm,
// tulpaOcc, tulpaRatio, ...) pin its own log-likelihood through Laplace
// without adding a family enum to tulpa for every new family it ships.
//
// Covers n_processes >= 1 with at most one iid RE term. The shared iid RE
// is added (as +u_g) into every process whose data.sharing.re[k] is true;
// when sharing.re[k] is false the RE does not contribute to that process's
// linear predictor. Single-process is a special case of the general path
// (np == 1), not a separate branch.
//
// Random-slope / spatial / temporal variants remain follow-on work and
// raise a clear error.

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_newton_loop.h"
#include "linalg_fast.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

namespace {

// Layout helper: sub-vector that Newton actually moves on. For n_processes
// >= 1 the latent slice is the concatenation of every process's beta block
// followed by the (re_start .. re_end) span when has_re is true. Extra
// parameters (sigma_re, dispersion, ZI betas, ...) are held fixed at their
// input value — Laplace integrates over the latent field, not the
// hyperparameters.
//
// `latent_offset[k]` gives the offset in the n_x-long latent vector where
// process k's beta block lives. `latent_offset[np]` is where the RE block
// starts (or n_x when has_re is false). This single index lets the scatter
// step write straight into the (latent_x, latent_x) Hessian without a per-
// process branch.
struct SpecLatentLayout {
    int np = 1;
    std::vector<int> beta_start;        // [np] absolute index into params
    std::vector<int> beta_count;        // [np] block size
    std::vector<int> latent_offset;     // [np + 1] prefix into n_x
    int re_start = -1;
    int re_end = -1;                    // exclusive (-1 if no RE)
    int re_offset = 0;                  // offset of RE block in latent vec
    bool has_re = false;
    int n_re_groups = 0;
    int n_x = 0;                        // sum of beta_counts + n_re_groups
};

inline SpecLatentLayout build_latent_layout(
    const ModelData& data,
    const ParamLayout& layout
) {
    SpecLatentLayout L;
    if (layout.process_beta_start.empty()) {
        Rcpp::stop("laplace_spec_dense: ParamLayout.process_beta_start is empty");
    }
    L.np = data.n_processes;
    if (L.np <= 0) {
        Rcpp::stop("laplace_spec_dense: requires n_processes >= 1 (got %d)", L.np);
    }
    if ((int)layout.process_beta_start.size() < L.np ||
        (int)layout.process_beta_count.size() < L.np) {
        Rcpp::stop("laplace_spec_dense: ParamLayout has %d/%d process_beta blocks "
                   "but data.n_processes == %d",
                   (int)layout.process_beta_start.size(),
                   (int)layout.process_beta_count.size(), L.np);
    }
    L.beta_start.assign(L.np, 0);
    L.beta_count.assign(L.np, 0);
    L.latent_offset.assign(L.np + 1, 0);
    int running = 0;
    for (int k = 0; k < L.np; k++) {
        L.beta_start[k] = layout.process_beta_start[k];
        L.beta_count[k] = layout.process_beta_count[k];
        L.latent_offset[k] = running;
        running += L.beta_count[k];
    }
    L.latent_offset[L.np] = running;

    L.has_re = layout.has_re && layout.re_start >= 0;
    if (L.has_re) {
        L.re_start = layout.re_start;
        L.re_end = layout.re_end;
        L.n_re_groups = L.re_end - L.re_start;
        L.re_offset = running;
        running += L.n_re_groups;
    }
    L.n_x = running;
    return L;
}

// True iff the iid RE contributes to process k's linear predictor. Falls back
// to "share into every process" when SharingSpec.re hasn't been initialised
// (e.g. legacy callers building ModelData without sharing.init()).
inline bool re_shared_into(const ModelData& data, int k) {
    if ((int)data.sharing.re.size() != data.n_processes) return true;
    return data.sharing.re[k];
}

// Pull the sigma_re hyperparameter out of the params vector when the layout
// records its log-precision slot. Mirrors the convention used by the legacy
// Laplace path (sigma_re is the standard deviation of the iid RE prior).
inline double current_sigma_re(
    const std::vector<double>& params,
    const ParamLayout& layout
) {
    if (layout.log_sigma_re_idx < 0) return 1.0;
    return std::exp(params[layout.log_sigma_re_idx]);
}

// Build eta_flat for all processes. eta_flat is laid out as [N x np] in
// observation-major order so eta_flat[i*np + k] is process k's linear
// predictor at observation i — the same layout the LikelihoodSpec eta
// pointer expects (one obs at a time, &eta_flat[i*np]).
//
// Per process: eta_k_i = (X_k beta_k)_i + offset_k_i + (RE if shared into k).
inline void compute_eta_spec(
    const ModelData& data,
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    const std::vector<int>& re_group_1based,
    int N,
    std::vector<double>& eta_flat,
    int n_threads
) {
    const int np = L.np;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N; i++) {
        // Optional shared iid RE contribution at observation i.
        double re_eff = 0.0;
        bool re_used = false;
        if (L.has_re && !re_group_1based.empty()) {
            int g = re_group_1based[i] - 1;
            if (g >= 0 && g < L.n_re_groups) {
                re_eff = params[L.re_start + g];
                re_used = true;
            }
        }

        for (int k = 0; k < np; k++) {
            const ProcessData& proc = data.processes[k];
            double e = 0.0;
            if (proc.p > 0) {
                const double* row = proc.X_flat.data() + (std::ptrdiff_t)i * proc.p;
                const double* beta = &params[L.beta_start[k]];
                for (int j = 0; j < proc.p; j++) e += row[j] * beta[j];
            }
            if (!proc.offset.empty()) e += proc.offset[i];
            if (re_used && re_shared_into(data, k)) e += re_eff;
            eta_flat[(std::ptrdiff_t)i * np + k] = e;
        }
    }
}

// Per-observation gradient + neg-Hessian assembled into the latent gradient
// and Hessian in (beta_0..beta_{np-1}, re) space. eta_weights_fn writes
//   grad_eta[k]                       = d log_lik_i / d eta_i_k
//   neg_hess_eta[k * np + l]          = -d^2 log_lik_i / (d eta_i_k d eta_i_l)
// Adds the iid RE prior contribution at the end (-tau_re * u for grad,
// +tau_re on the diagonal of H).
inline void scatter_spec(
    const std::vector<double>& params,
    const std::vector<double>& eta_flat,
    const std::vector<int>& re_group_1based,
    const SpecLatentLayout& L,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N,
    double tau_re,
    DenseVec& grad,
    DenseMat& H,
    int /*n_threads*/
) {
    std::fill(grad.begin(), grad.end(), 0.0);
    for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);

    const int np = L.np;
    std::vector<double> grad_eta(np, 0.0);
    std::vector<double> neg_hess_eta((size_t)np * np, 0.0);

    for (int i = 0; i < N; i++) {
        std::fill(grad_eta.begin(), grad_eta.end(), 0.0);
        std::fill(neg_hess_eta.begin(), neg_hess_eta.end(), 0.0);

        spec.eta_weights_fn(
            i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
            params, data, layout, response_data,
            grad_eta.data(), neg_hess_eta.data()
        );

        // Resolve the RE group at observation i once.
        int g_re = -1;
        if (L.has_re && !re_group_1based.empty()) {
            int g = re_group_1based[i] - 1;
            if (g >= 0 && g < L.n_re_groups) g_re = g;
        }

        // ============= GRADIENT scatter =============
        // Per-process beta gradient: g_{beta_k_j} += X_k(i, j) * grad_eta[k]
        // Shared RE gradient: g_{u_g} += sum_{k in shared} grad_eta[k]
        double re_grad_acc = 0.0;
        for (int k = 0; k < np; k++) {
            const ProcessData& proc = data.processes[k];
            const double gk = grad_eta[k];
            if (proc.p > 0) {
                const double* row = proc.X_flat.data() + (std::ptrdiff_t)i * proc.p;
                double* gbeta = &grad[L.latent_offset[k]];
                for (int j = 0; j < proc.p; j++) gbeta[j] += row[j] * gk;
            }
            if (g_re >= 0 && re_shared_into(data, k)) re_grad_acc += gk;
        }
        if (g_re >= 0) grad[L.re_offset + g_re] += re_grad_acc;

        // ============= HESSIAN scatter (lower triangle) =============
        // Block layout: H[(beta_k_j, beta_l_m)] += X_k(i,j) * X_l(i,m) * w_{kl}
        // RE blocks: shared RE acts as a "design column" of 1s into each shared
        // process. So for shared processes k, l:
        //   H[(beta_k_j, u_g)]  += X_k(i, j) * w_{kl}  (sum over l shared)
        //   H[(u_g, u_g)]       += sum_{k,l shared} w_{kl}
        // (lower-triangle: only writes to entries with row >= col)

        // Helper: weight w_{kl} = neg_hess_eta[k*np + l]
        for (int k = 0; k < np; k++) {
            const ProcessData& pk = data.processes[k];
            const double* xk = (pk.p > 0)
                ? pk.X_flat.data() + (std::ptrdiff_t)i * pk.p
                : nullptr;
            const int off_k = L.latent_offset[k];

            for (int l = 0; l <= k; l++) {
                const ProcessData& pl = data.processes[l];
                const double* xl = (pl.p > 0)
                    ? pl.X_flat.data() + (std::ptrdiff_t)i * pl.p
                    : nullptr;
                const int off_l = L.latent_offset[l];
                const double w_kl = neg_hess_eta[(size_t)k * np + l];

                if (xk && xl) {
                    if (k == l) {
                        // diagonal block: only j >= m
                        for (int j = 0; j < pk.p; j++) {
                            const double w_xj = w_kl * xk[j];
                            double* row_j = H[off_k + j].data();
                            for (int m = 0; m <= j; m++) {
                                row_j[off_l + m] += w_xj * xl[m];
                            }
                        }
                    } else {
                        // off-diagonal block (k > l): write H[off_k + j, off_l + m]
                        for (int j = 0; j < pk.p; j++) {
                            const double w_xj = w_kl * xk[j];
                            double* row_j = H[off_k + j].data();
                            for (int m = 0; m < pl.p; m++) {
                                row_j[off_l + m] += w_xj * xl[m];
                            }
                        }
                    }
                }
            }
        }

        // RE × beta and RE × RE blocks. The RE row index is L.re_offset + g_re,
        // which is the largest index in the latent vector by construction
        // (re_offset == sum of all beta_counts), so it is always a "row >= col"
        // write relative to any beta block.
        if (g_re >= 0) {
            // Sum of weights into the shared-into set, used for the (RE, RE)
            // diagonal: H[u_g, u_g] += sum_{k,l shared} w_{kl}.
            double w_re_re = 0.0;
            for (int k = 0; k < np; k++) {
                if (!re_shared_into(data, k)) continue;
                for (int l = 0; l < np; l++) {
                    if (!re_shared_into(data, l)) continue;
                    w_re_re += neg_hess_eta[(size_t)k * np + l];
                }
            }
            const int re_row = L.re_offset + g_re;
            // (RE, beta_l_m) for each l: weight = sum_{k shared} w_{kl}
            for (int l = 0; l < np; l++) {
                const ProcessData& pl = data.processes[l];
                if (pl.p == 0) continue;
                const double* xl = pl.X_flat.data() + (std::ptrdiff_t)i * pl.p;
                double w_l = 0.0;
                for (int k = 0; k < np; k++) {
                    if (!re_shared_into(data, k)) continue;
                    w_l += neg_hess_eta[(size_t)k * np + l];
                }
                double* row = H[re_row].data();
                const int off_l = L.latent_offset[l];
                for (int m = 0; m < pl.p; m++) row[off_l + m] += w_l * xl[m];
            }
            H[re_row][re_row] += w_re_re;
        }
    }

    // Symmetrise lower → upper triangle.
    for (int j = 0; j < L.n_x; j++) {
        for (int k = j + 1; k < L.n_x; k++) {
            H[j][k] = H[k][j];
        }
    }

    // beta prior: N(0, sigma_beta^2 I), applied per process.
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta + 1e-300);
    for (int k = 0; k < np; k++) {
        const int off_k = L.latent_offset[k];
        for (int j = 0; j < L.beta_count[k]; j++) {
            grad[off_k + j] += -tau_beta * params[L.beta_start[k] + j];
            H[off_k + j][off_k + j] += tau_beta;
        }
    }

    // iid RE prior: N(0, sigma_re^2 I)
    if (L.has_re) {
        for (int g = 0; g < L.n_re_groups; g++) {
            grad[L.re_offset + g] += -tau_re * params[L.re_start + g];
            H[L.re_offset + g][L.re_offset + g] += tau_re;
        }
    }
}

inline double total_log_lik_spec(
    const std::vector<double>& params,
    const std::vector<double>& eta_flat,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N
) {
    const int np = data.n_processes;
    double ll = 0.0;
    for (int i = 0; i < N; i++) {
        ll += spec.ll_double(
            i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
            params, data, layout, response_data
        );
    }
    return ll;
}

inline double log_prior_latent(
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    double sigma_beta,
    double tau_re
) {
    double lp = 0.0;
    double tau_beta = 1.0 / (sigma_beta * sigma_beta + 1e-300);
    int p_total = 0;
    for (int k = 0; k < L.np; k++) {
        for (int j = 0; j < L.beta_count[k]; j++) {
            double b = params[L.beta_start[k] + j];
            lp += -0.5 * tau_beta * b * b;
        }
        p_total += L.beta_count[k];
    }
    lp += 0.5 * p_total * std::log(tau_beta / (2.0 * M_PI));
    if (L.has_re) {
        for (int g = 0; g < L.n_re_groups; g++) {
            double u = params[L.re_start + g];
            lp += -0.5 * tau_re * u * u;
        }
        lp += 0.5 * L.n_re_groups * std::log(tau_re / (2.0 * M_PI));
    }
    return lp;
}

} // namespace (anonymous)

// ----------------------------------------------------------------------------
// Public entry. Mirrors laplace_mode_dense's contract but reads the
// log-likelihood and IRLS weights from data.likelihood_spec.
//
//   data, layout: filled exactly as for the NUTS shim (n_processes == 1).
//   params_inout: full parameter vector. On entry, hyperparameter slots
//     (sigma_re, extras) supply their fixed values; latent slots may carry
//     a warm start. On exit, the latent slots are overwritten with the
//     mode; hyperparameter slots are untouched.
//   re_group_1based: per-obs 1-based RE group index. Empty / nullptr if no RE.
//
// Returns the same LaplaceShimResult shape as the family-enum shims so
// downstream code can switch between paths without re-wiring its result
// handling.
// ----------------------------------------------------------------------------
struct LaplaceShimResult;

// Helper: write a trial latent step into the parameter vector. Used by both
// the Newton step and the gradient-only fall-back when factorisation fails.
inline void apply_latent_step(
    const SpecLatentLayout& L,
    const std::vector<double>& base,
    const std::vector<double>& step,
    double scale,
    std::vector<double>& out
) {
    out = base;
    for (int k = 0; k < L.np; k++) {
        const int off_k = L.latent_offset[k];
        for (int j = 0; j < L.beta_count[k]; j++) {
            out[L.beta_start[k] + j] = base[L.beta_start[k] + j]
                                       + scale * step[off_k + j];
        }
    }
    if (L.has_re) {
        for (int g = 0; g < L.n_re_groups; g++) {
            out[L.re_start + g] = base[L.re_start + g]
                                  + scale * step[L.re_offset + g];
        }
    }
}

void laplace_mode_spec_dense_impl(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& params_inout,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    int* n_iter_out,
    int* converged_out,
    double* log_det_Q_out,
    double* log_marginal_out
) {
    if (data.n_processes < 1) {
        Rcpp::stop("laplace_spec_dense: requires n_processes >= 1 (got %d)",
                   data.n_processes);
    }
    const LikelihoodSpec* spec =
        static_cast<const LikelihoodSpec*>(data.likelihood_spec);
    if (spec == nullptr) {
        Rcpp::stop("laplace_spec_dense: data.likelihood_spec is null");
    }
    if (spec->eta_weights_fn == nullptr) {
        Rcpp::stop("laplace_spec_dense: LikelihoodSpec.eta_weights_fn is null "
                   "(required for spec-driven Laplace)");
    }
    if (spec->ll_double == nullptr) {
        Rcpp::stop("laplace_spec_dense: LikelihoodSpec.ll_double is null");
    }
    if ((int)data.processes.size() != data.n_processes) {
        Rcpp::stop("laplace_spec_dense: data.processes.size() (%d) != n_processes (%d)",
                   (int)data.processes.size(), data.n_processes);
    }

    SpecLatentLayout L = build_latent_layout(data, layout);
    const int np = L.np;
    for (int k = 0; k < np; k++) {
        const ProcessData& proc = data.processes[k];
        if (proc.p != L.beta_count[k]) {
            Rcpp::stop("laplace_spec_dense: process[%d].p (%d) != "
                       "ParamLayout.process_beta_count[%d] (%d)",
                       k, proc.p, k, L.beta_count[k]);
        }
        if (proc.p > 0 && (int)proc.X_flat.size() != data.N * proc.p) {
            Rcpp::stop("laplace_spec_dense: process[%d] design matrix shape "
                       "(%d) inconsistent with N * p (%d * %d)",
                       k, (int)proc.X_flat.size(), data.N, proc.p);
        }
    }
    if (L.has_re && (int)re_group_1based.size() != data.N) {
        Rcpp::stop("laplace_spec_dense: re_group has length %d but N = %d",
                   (int)re_group_1based.size(), data.N);
    }

    int N = data.N;
    int n_x = L.n_x;
    double sigma_re = current_sigma_re(params_inout, layout);
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-300);

    bool use_sparse = (n_x >= 200);
    SparseCholeskySolver sparse_solver;
    const void* response_data = data.model_response_data;

    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif

    std::vector<double> eta_flat((size_t)N * np, 0.0);

    auto eval_objective = [&](const std::vector<double>& trial_params) -> double {
        compute_eta_spec(data, trial_params, L, re_group_1based, N, eta_flat, n_threads);
        double ll = total_log_lik_spec(
            trial_params, eta_flat, data, layout, *spec, response_data, N
        );
        double lp = log_prior_latent(trial_params, L, data.sigma_beta, tau_re);
        return ll + lp;
    };

    double obj_current = eval_objective(params_inout);
    int n_iter = 0;
    bool converged = false;

    DenseVec grad(n_x, 0.0);
    DenseMat H(n_x, DenseVec(n_x, 0.0));
    std::vector<double> delta(n_x, 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta_spec(data, params_inout, L, re_group_1based, N, eta_flat, n_threads);
        scatter_spec(params_inout, eta_flat, re_group_1based, L,
                     data, layout, *spec, response_data, N, tau_re,
                     grad, H, n_threads);

        std::fill(delta.begin(), delta.end(), 0.0);
        bool solve_ok = dispatch_factor_solve(H, grad, delta, n_x,
                                              sparse_solver, use_sparse);
        if (!solve_ok) {
            // Damped fall-back: tiny gradient step keeps the search moving so a
            // bad starting point doesn't kill the run.
            std::vector<double> grad_step(n_x, 0.0);
            for (int j = 0; j < n_x; j++) {
                grad_step[j] = std::isfinite(grad[j]) ? 0.01 * grad[j] : 0.0;
            }
            std::vector<double> trial;
            apply_latent_step(L, params_inout, grad_step, 1.0, trial);
            params_inout = trial;
            obj_current = eval_objective(params_inout);
            n_iter = iter + 1;
            continue;
        }

        // Step halving in latent slots only — hyperparameter slots are pinned.
        double step_scale = 1.0;
        std::vector<double> trial;
        for (int half = 0; half <= 12; half++) {
            apply_latent_step(L, params_inout, delta, step_scale, trial);
            double obj_try = eval_objective(trial);
            if (obj_try >= obj_current - 1e-8 || half == 12) {
                params_inout = trial;
                obj_current = obj_try;
                break;
            }
            step_scale *= 0.5;
        }

        n_iter = iter + 1;
        double m = 0.0;
        for (int j = 0; j < n_x; j++) {
            m = std::max(m, std::abs(step_scale * delta[j]));
        }
        if (m < tol) {
            converged = true;
            break;
        }
    }

    // Final Hessian + log-marginal at the mode.
    compute_eta_spec(data, params_inout, L, re_group_1based, N, eta_flat, n_threads);
    scatter_spec(params_inout, eta_flat, re_group_1based, L,
                 data, layout, *spec, response_data, N, tau_re,
                 grad, H, n_threads);
    double log_det_Q = 0.0;
    dispatch_factor_log_det(H, n_x, sparse_solver, use_sparse, log_det_Q);

    double ll = total_log_lik_spec(
        params_inout, eta_flat, data, layout, *spec, response_data, N
    );
    double lp = log_prior_latent(params_inout, L, data.sigma_beta, tau_re);
    double log_marginal = finalize_log_marginal(ll, lp, log_det_Q, n_x);

    if (n_iter_out)      *n_iter_out      = n_iter;
    if (converged_out)   *converged_out   = converged ? 1 : 0;
    if (log_det_Q_out)   *log_det_Q_out   = log_det_Q;
    if (log_marginal_out)*log_marginal_out= log_marginal;
}

} // namespace tulpa

// ============================================================================
// Test harness: Gaussian LikelihoodSpec + iid RE driven through
// laplace_mode_spec_dense_impl. Lets the testthat suite exercise the
// spec-driven Laplace path against the family-enum reference without
// requiring a downstream package. Internal — not part of the public ABI.
// ============================================================================

namespace {

struct GaussianResponse {
    const double* y;
    int N;
    double phi;   // observation sd
};

template<typename T>
T gaussian_ll_T(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data
) {
    auto* resp = static_cast<const GaussianResponse*>(model_data);
    T r = T(resp->y[i]) - eta[0];
    T phi2 = T(resp->phi * resp->phi);
    return T(-0.5) * std::log(T(2.0 * M_PI) * phi2) - r * r / (T(2.0) * phi2);
}

void gaussian_eta_weights(
    int i,
    const double* eta,
    double /*logit_zi*/,
    double /*logit_oi*/,
    const std::vector<double>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data,
    double* grad_eta,
    double* neg_hess_eta
) {
    auto* resp = static_cast<const GaussianResponse*>(model_data);
    double inv_phi2 = 1.0 / (resp->phi * resp->phi);
    grad_eta[0]     = (resp->y[i] - eta[0]) * inv_phi2;
    neg_hess_eta[0] = inv_phi2;
}

} // namespace (anonymous, test only)

// [[Rcpp::export]]
Rcpp::List cpp_laplace_spec_test_gaussian(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_idx,    // 1-based; pass integer(0) for no RE
    int n_re_groups,
    double sigma_re,
    double sigma_beta,
    double phi,
    int max_iter = 100,
    double tol = 1e-8,
    int n_threads = 1
) {
    int N = y.size();
    int p = X.ncol();
    bool has_re = (n_re_groups > 0) && (re_idx.size() == N);

    // Build ProcessData (row-major X_flat).
    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[(size_t)i * p + j] = X(i, j);
        }
    }

    // Build LikelihoodSpec.
    tulpa::LikelihoodSpec spec;
    spec.n_processes      = 1;
    spec.name             = "gaussian_spec_test";
    spec.ll_double        = &gaussian_ll_T<double>;
    spec.eta_weights_fn   = &gaussian_eta_weights;
    spec.n_extra_params   = 0;

    // Build response payload.
    GaussianResponse resp{y.begin(), N, phi};

    // Build ModelData.
    tulpa::ModelData data;
    data.n_processes          = 1;
    data.processes.push_back(proc);
    data.N                    = N;
    data.sigma_beta           = sigma_beta;
    data.likelihood_spec      = &spec;
    data.model_response_data  = &resp;
    data.sharing.init(1);
    if (has_re) {
        data.n_re_groups = n_re_groups;
        data.re_group.assign(re_idx.begin(), re_idx.end());
    }

    // Build ParamLayout (manually — bypasses tulpa_compute_param_layout
    // because the test pins the latent layout we want to exercise).
    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    int next = p;
    if (has_re) {
        layout.has_re = true;
        layout.log_sigma_re_idx = next++;
        layout.re_start = next;
        layout.re_end   = next + n_re_groups;
        next = layout.re_end;
    }
    layout.total_params = next;

    // Initialise params: zeros for latent, log(sigma_re) for the precision slot.
    std::vector<double> params(layout.total_params, 0.0);
    if (has_re) params[layout.log_sigma_re_idx] = std::log(sigma_re);

    // Build re_group_1based slice for the impl.
    std::vector<int> re_group_1based;
    if (has_re) re_group_1based.assign(re_idx.begin(), re_idx.end());

    int n_iter = 0;
    int converged = 0;
    double log_det_Q = 0.0;
    double log_marginal = 0.0;
    tulpa::laplace_mode_spec_dense_impl(
        data, layout, params, re_group_1based,
        max_iter, tol, n_threads,
        &n_iter, &converged, &log_det_Q, &log_marginal
    );

    // Return mode + diagnostics in the same shape as cpp_laplace_fit's list.
    Rcpp::NumericVector mode(p + (has_re ? n_re_groups : 0));
    for (int j = 0; j < p; j++) mode[j] = params[j];
    if (has_re) {
        for (int g = 0; g < n_re_groups; g++) mode[p + g] = params[layout.re_start + g];
    }
    return Rcpp::List::create(
        Rcpp::Named("mode")         = mode,
        Rcpp::Named("log_det_Q")    = log_det_Q,
        Rcpp::Named("log_marginal") = log_marginal,
        Rcpp::Named("n_iter")       = n_iter,
        Rcpp::Named("converged")    = (converged != 0)
    );
}

// ============================================================================
// Multi-process Gaussian test fixture: n_processes == 2 independent Gaussian
// likelihoods on (y1, y2) with separate (X1, X2) and a shared iid RE that
// enters both processes' linear predictors. Exercises the generalized
// laplace_mode_spec_dense_impl path (np >= 2 + cross-process Hessian zeros
// + RE shared into multiple processes) against a hand-derived Newton step
// the test computes in R.
// ============================================================================

namespace {

struct Gaussian2pResponse {
    const double* y1;
    const double* y2;
    int N;
    double phi1;
    double phi2;
};

template<typename T>
T gaussian2p_ll_T(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data
) {
    auto* resp = static_cast<const Gaussian2pResponse*>(model_data);
    T r1 = T(resp->y1[i]) - eta[0];
    T r2 = T(resp->y2[i]) - eta[1];
    T phi1_2 = T(resp->phi1 * resp->phi1);
    T phi2_2 = T(resp->phi2 * resp->phi2);
    T ll = T(-0.5) * std::log(T(2.0 * M_PI) * phi1_2)
           - r1 * r1 / (T(2.0) * phi1_2)
           + T(-0.5) * std::log(T(2.0 * M_PI) * phi2_2)
           - r2 * r2 / (T(2.0) * phi2_2);
    return ll;
}

// neg_hess_eta is row-major [np x np]. Independent Gaussians: cross terms 0.
void gaussian2p_eta_weights(
    int i,
    const double* eta,
    double /*logit_zi*/,
    double /*logit_oi*/,
    const std::vector<double>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data,
    double* grad_eta,
    double* neg_hess_eta
) {
    auto* resp = static_cast<const Gaussian2pResponse*>(model_data);
    double inv1 = 1.0 / (resp->phi1 * resp->phi1);
    double inv2 = 1.0 / (resp->phi2 * resp->phi2);
    grad_eta[0] = (resp->y1[i] - eta[0]) * inv1;
    grad_eta[1] = (resp->y2[i] - eta[1]) * inv2;
    neg_hess_eta[0] = inv1;   // (0, 0)
    neg_hess_eta[1] = 0.0;    // (0, 1)
    neg_hess_eta[2] = 0.0;    // (1, 0)
    neg_hess_eta[3] = inv2;   // (1, 1)
}

} // namespace (anonymous, test only)

// [[Rcpp::export]]
Rcpp::List cpp_laplace_spec_test_gaussian2p(
    Rcpp::NumericVector y1,
    Rcpp::NumericVector y2,
    Rcpp::NumericMatrix X1,
    Rcpp::NumericMatrix X2,
    Rcpp::NumericVector offset1,
    Rcpp::NumericVector offset2,
    Rcpp::IntegerVector re_idx,
    int n_re_groups,
    double sigma_re,
    double sigma_beta,
    double phi1,
    double phi2,
    bool re_into_proc0 = true,
    bool re_into_proc1 = true,
    int max_iter = 100,
    double tol = 1e-10,
    int n_threads = 1
) {
    int N = y1.size();
    int p1 = X1.ncol();
    int p2 = X2.ncol();
    bool has_re = (n_re_groups > 0) && (re_idx.size() == N);

    auto build_proc = [N](Rcpp::NumericMatrix Xr,
                          Rcpp::NumericVector offr) {
        tulpa::ProcessData proc;
        proc.p = Xr.ncol();
        proc.X_flat.resize((size_t)N * proc.p);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < proc.p; j++) {
                proc.X_flat[(size_t)i * proc.p + j] = Xr(i, j);
            }
        }
        if (offr.size() == N) {
            proc.offset.assign(offr.begin(), offr.end());
        }
        return proc;
    };

    tulpa::ProcessData proc1 = build_proc(X1, offset1);
    tulpa::ProcessData proc2 = build_proc(X2, offset2);

    Gaussian2pResponse resp{y1.begin(), y2.begin(), N, phi1, phi2};

    tulpa::LikelihoodSpec spec;
    spec.n_processes      = 2;
    spec.name             = "gaussian2p_spec_test";
    spec.ll_double        = &gaussian2p_ll_T<double>;
    spec.eta_weights_fn   = &gaussian2p_eta_weights;

    tulpa::ModelData data;
    data.n_processes         = 2;
    data.processes.push_back(proc1);
    data.processes.push_back(proc2);
    data.N                   = N;
    data.sigma_beta          = sigma_beta;
    data.likelihood_spec     = &spec;
    data.model_response_data = &resp;
    data.sharing.init(2);
    data.sharing.re[0] = re_into_proc0;
    data.sharing.re[1] = re_into_proc1;
    if (has_re) {
        data.n_re_groups = n_re_groups;
        data.re_group.assign(re_idx.begin(), re_idx.end());
    }

    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p1);
    layout.process_beta_start.push_back(p1);
    layout.process_beta_count.push_back(p2);
    int next = p1 + p2;
    if (has_re) {
        layout.has_re = true;
        layout.log_sigma_re_idx = next++;
        layout.re_start = next;
        layout.re_end   = next + n_re_groups;
        next = layout.re_end;
    }
    layout.total_params = next;

    std::vector<double> params(layout.total_params, 0.0);
    if (has_re) params[layout.log_sigma_re_idx] = std::log(sigma_re);

    std::vector<int> re_group_1based;
    if (has_re) re_group_1based.assign(re_idx.begin(), re_idx.end());

    int n_iter = 0;
    int converged = 0;
    double log_det_Q = 0.0;
    double log_marginal = 0.0;
    tulpa::laplace_mode_spec_dense_impl(
        data, layout, params, re_group_1based,
        max_iter, tol, n_threads,
        &n_iter, &converged, &log_det_Q, &log_marginal
    );

    int n_x = p1 + p2 + (has_re ? n_re_groups : 0);
    Rcpp::NumericVector mode(n_x);
    for (int j = 0; j < p1; j++) mode[j] = params[j];
    for (int j = 0; j < p2; j++) mode[p1 + j] = params[p1 + j];
    if (has_re) {
        for (int g = 0; g < n_re_groups; g++) {
            mode[p1 + p2 + g] = params[layout.re_start + g];
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("mode")         = mode,
        Rcpp::Named("log_det_Q")    = log_det_Q,
        Rcpp::Named("log_marginal") = log_marginal,
        Rcpp::Named("n_iter")       = n_iter,
        Rcpp::Named("converged")    = (converged != 0)
    );
}
