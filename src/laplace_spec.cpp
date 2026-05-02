// laplace_spec.cpp
// LikelihoodSpec-driven Laplace mode finder.
//
// Mirrors the laplace_mode_dense pattern in laplace_core.cpp but routes
// the per-observation log-likelihood and IRLS weights through a
// LikelihoodSpec rather than the family-enum dispatch in
// laplace_family_link.h. Lets a downstream package (tulpaGlmm,
// tulpaOcc, ...) pin its own log-likelihood through Laplace without
// adding a family enum to tulpa for every new family it ships.
//
// First cut covers n_processes == 1 with at most one iid RE term.
// Multi-process / random-slope / spatial / temporal variants are
// follow-on work — intentionally not built here so the spec contract
// can stabilise on real downstream traffic before being widened.

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

// Layout helper: sub-vector that Newton actually moves on. For the first cut
// (n_processes == 1, optional single iid RE term, no spatial/temporal/etc.)
// the latent slice is the contiguous span [beta_start .. beta_end) plus the
// (re_start .. re_end) span when has_re is true. Extra parameters (sigma_re,
// dispersion, ZI betas, ...) are held fixed at their input value — Laplace
// integrates over the latent field, not the hyperparameters.
struct SpecLatentLayout {
    int beta_start = 0;
    int beta_end = 0;       // exclusive
    int re_start = -1;
    int re_end = -1;        // exclusive (-1 if no RE)
    bool has_re = false;
    int p = 0;
    int n_re_groups = 0;
    int n_x = 0;            // p + n_re_groups
};

inline SpecLatentLayout build_latent_layout(const ParamLayout& layout) {
    SpecLatentLayout L;
    if (layout.process_beta_start.empty()) {
        Rcpp::stop("laplace_spec_dense: ParamLayout.process_beta_start is empty");
    }
    L.beta_start = layout.process_beta_start[0];
    L.p = layout.process_beta_count[0];
    L.beta_end = L.beta_start + L.p;
    L.has_re = layout.has_re && layout.re_start >= 0;
    if (L.has_re) {
        L.re_start = layout.re_start;
        L.re_end = layout.re_end;
        L.n_re_groups = L.re_end - L.re_start;
    }
    L.n_x = L.p + L.n_re_groups;
    return L;
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

// Build eta for n_processes == 1. process 0's beta lives in
// params[beta_start..beta_end), iid RE in params[re_start..re_end).
inline void compute_eta_spec(
    const ProcessData& proc,
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    const std::vector<int>& re_group_1based,
    int N,
    Rcpp::NumericVector& eta_out,
    int n_threads
) {
    const double* X = proc.X_flat.data();
    const double* beta = &params[L.beta_start];

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N; i++) {
        double e = 0.0;
        const double* row = X + (std::ptrdiff_t)i * L.p;
        for (int j = 0; j < L.p; j++) e += row[j] * beta[j];
        if (L.has_re && !re_group_1based.empty()) {
            int g = re_group_1based[i] - 1;
            if (g >= 0 && g < L.n_re_groups) {
                e += params[L.re_start + g];
            }
        }
        eta_out[i] = e;
    }
}

// Per-observation gradient + neg-Hessian assembled into the latent gradient
// and Hessian in (beta, re) space. Adds the iid RE prior contribution at the
// end (-tau_re * u for grad, +tau_re on the diagonal of H).
inline void scatter_spec(
    const std::vector<double>& params,
    const Rcpp::NumericVector& eta,
    const ProcessData& proc,
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

    const double* X = proc.X_flat.data();
    double g_eta = 0.0;
    double w = 0.0;

    // Single-process implementation: grad_eta and neg_hess_eta are scalars.
    for (int i = 0; i < N; i++) {
        spec.eta_weights_fn(
            i, &eta[i], 0.0, 0.0, params, data, layout, response_data,
            &g_eta, &w
        );
        const double* row = X + (std::ptrdiff_t)i * L.p;

        for (int j = 0; j < L.p; j++) {
            grad[j] += row[j] * g_eta;
        }
        if (L.has_re && !re_group_1based.empty()) {
            int g = re_group_1based[i] - 1;
            if (g >= 0 && g < L.n_re_groups) {
                grad[L.p + g] += g_eta;
            }
        }

        for (int j = 0; j < L.p; j++) {
            double w_xj = w * row[j];
            for (int k = 0; k <= j; k++) {
                H[j][k] += w_xj * row[k];
            }
            if (L.has_re && !re_group_1based.empty()) {
                int g = re_group_1based[i] - 1;
                if (g >= 0 && g < L.n_re_groups) {
                    H[L.p + g][j] += w_xj;
                }
            }
        }
        if (L.has_re && !re_group_1based.empty()) {
            int g = re_group_1based[i] - 1;
            if (g >= 0 && g < L.n_re_groups) {
                H[L.p + g][L.p + g] += w;
            }
        }
    }

    // Symmetrise lower → upper triangle.
    for (int j = 0; j < L.n_x; j++) {
        for (int k = j + 1; k < L.n_x; k++) {
            H[j][k] = H[k][j];
        }
    }

    // beta prior: N(0, sigma_beta^2 I)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta + 1e-300);
    for (int j = 0; j < L.p; j++) {
        grad[j] += -tau_beta * params[L.beta_start + j];
        H[j][j] += tau_beta;
    }

    // iid RE prior: N(0, sigma_re^2 I)
    if (L.has_re) {
        for (int g = 0; g < L.n_re_groups; g++) {
            grad[L.p + g] += -tau_re * params[L.re_start + g];
            H[L.p + g][L.p + g] += tau_re;
        }
    }
}

inline double total_log_lik_spec(
    const std::vector<double>& params,
    const Rcpp::NumericVector& eta,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N
) {
    double ll = 0.0;
    for (int i = 0; i < N; i++) {
        double eta_i = eta[i];
        ll += spec.ll_double(
            i, &eta_i, 0.0, 0.0, params, data, layout, response_data
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
    for (int j = 0; j < L.p; j++) {
        double b = params[L.beta_start + j];
        lp += -0.5 * tau_beta * b * b;
    }
    lp += 0.5 * L.p * std::log(tau_beta / (2.0 * M_PI));
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
    if (data.n_processes != 1) {
        Rcpp::stop("laplace_spec_dense: requires n_processes == 1 (got %d)",
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

    SpecLatentLayout L = build_latent_layout(layout);
    const ProcessData& proc = data.processes[0];
    if ((int)proc.X_flat.size() != data.N * L.p) {
        Rcpp::stop("laplace_spec_dense: process[0] design matrix shape "
                   "(%d) inconsistent with N * p (%d * %d)",
                   (int)proc.X_flat.size(), data.N, L.p);
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

    Rcpp::NumericVector eta(N, 0.0);

    auto eval_objective = [&](const std::vector<double>& trial_params) -> double {
        compute_eta_spec(proc, trial_params, L, re_group_1based, N, eta, n_threads);
        double ll = total_log_lik_spec(
            trial_params, eta, data, layout, *spec, response_data, N
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
        compute_eta_spec(proc, params_inout, L, re_group_1based, N, eta, n_threads);
        scatter_spec(params_inout, eta, proc, re_group_1based, L,
                     data, layout, *spec, response_data, N, tau_re,
                     grad, H, n_threads);

        std::fill(delta.begin(), delta.end(), 0.0);
        bool solve_ok = dispatch_factor_solve(H, grad, delta, n_x,
                                              sparse_solver, use_sparse);
        if (!solve_ok) {
            // Damped fall-back: tiny gradient step keeps the search moving so a
            // bad starting point doesn't kill the run.
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(grad[j])) {
                    if (j < L.p) params_inout[L.beta_start + j] += 0.01 * grad[j];
                    else         params_inout[L.re_start + (j - L.p)] += 0.01 * grad[j];
                }
            }
            obj_current = eval_objective(params_inout);
            n_iter = iter + 1;
            continue;
        }

        // Step halving in latent slots only — hyperparameter slots are pinned.
        double step_scale = 1.0;
        bool accepted = false;
        std::vector<double> trial = params_inout;
        for (int half = 0; half <= 12; half++) {
            for (int j = 0; j < L.p; j++) {
                trial[L.beta_start + j] = params_inout[L.beta_start + j]
                                          + step_scale * delta[j];
            }
            if (L.has_re) {
                for (int g = 0; g < L.n_re_groups; g++) {
                    trial[L.re_start + g] = params_inout[L.re_start + g]
                                            + step_scale * delta[L.p + g];
                }
            }
            double obj_try = eval_objective(trial);
            if (obj_try >= obj_current - 1e-8 || half == 12) {
                params_inout = trial;
                obj_current = obj_try;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
        }
        (void)accepted;

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
    compute_eta_spec(proc, params_inout, L, re_group_1based, N, eta, n_threads);
    scatter_spec(params_inout, eta, proc, re_group_1based, L,
                 data, layout, *spec, response_data, N, tau_re,
                 grad, H, n_threads);
    double log_det_Q = 0.0;
    dispatch_factor_log_det(H, n_x, sparse_solver, use_sparse, log_det_Q);

    double ll = total_log_lik_spec(
        params_inout, eta, data, layout, *spec, response_data, N
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
