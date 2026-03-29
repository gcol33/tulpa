// laplace_helpers.h
// Shared helpers for Laplace approximation engine
// Eliminates duplication across model-specific Laplace functions

#ifndef TULPA_LAPLACE_HELPERS_H
#define TULPA_LAPLACE_HELPERS_H

#include "laplace_core.h"
#include "linalg_fast.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Type aliases for the dense Hessian representation
using DenseVec = std::vector<double>;
using DenseMat = std::vector<std::vector<double>>;

// =====================================================================
// Dense Cholesky: factorize, solve, log-det
// =====================================================================

struct CholeskyResult {
    Rcpp::NumericMatrix L;
    Rcpp::NumericVector delta;
    double log_det;
    bool success;
};

// Factorize H = L L', solve L L' delta = rhs, compute log|H| in one pass.
// This replaces ~9 inline Cholesky triple-loops across laplace_core.cpp.
inline CholeskyResult dense_cholesky_solve(
    const DenseMat& H, const DenseVec& rhs, int n
) {
    CholeskyResult res;
    res.L = Rcpp::NumericMatrix(n, n);
    res.delta = Rcpp::NumericVector(n);
    res.log_det = 0.0;
    res.success = true;

    // Cholesky factorization: L such that H = L L'
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = H[j][k];
            for (int i = 0; i < k; i++) sum -= res.L(j, i) * res.L(k, i);
            if (j == k) {
                if (sum <= 0) sum = 1e-6;
                res.L(j, k) = std::sqrt(sum);
            } else {
                res.L(j, k) = sum / res.L(k, k);
            }
        }
    }

    // Log-determinant: log|H| = 2 * sum(log(diag(L)))
    for (int j = 0; j < n; j++) {
        res.log_det += std::log(res.L(j, j));
    }
    res.log_det *= 2.0;

    // Forward solve: L z = rhs
    Rcpp::NumericVector z(n);
    for (int j = 0; j < n; j++) {
        double sum = rhs[j];
        for (int k = 0; k < j; k++) sum -= res.L(j, k) * z[k];
        z[j] = sum / res.L(j, j);
    }

    // Back solve: L' delta = z
    for (int j = n - 1; j >= 0; j--) {
        double sum = z[j];
        for (int k = j + 1; k < n; k++) sum -= res.L(k, j) * res.delta[k];
        res.delta[j] = sum / res.L(j, j);
    }

    // Check for NaN
    for (int j = 0; j < n; j++) {
        if (!std::isfinite(res.delta[j])) {
            res.success = false;
            break;
        }
    }

    return res;
}

// Factorize only (for finalize / sampling — no solve needed).
// Returns L and log|H|.
inline void dense_cholesky_factorize(
    const DenseMat& H, int n,
    Rcpp::NumericMatrix& L, double& log_det
) {
    log_det = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = H[j][k];
            for (int i = 0; i < k; i++) sum -= L(j, i) * L(k, i);
            if (j == k) {
                if (sum <= 0) sum = 1e-6;
                L(j, k) = std::sqrt(sum);
            } else {
                L(j, k) = sum / L(k, k);
            }
        }
    }
    for (int j = 0; j < n; j++) {
        log_det += std::log(L(j, j));
    }
    log_det *= 2.0;
}

// Overload accepting NumericMatrix H (for the finalize paths that use it).
inline void dense_cholesky_factorize(
    const Rcpp::NumericMatrix& H, int n,
    Rcpp::NumericMatrix& L, double& log_det
) {
    log_det = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = H(j, k);
            for (int i = 0; i < k; i++) sum -= L(j, i) * L(k, i);
            if (j == k) {
                if (sum <= 0) sum = 1e-6;
                L(j, k) = std::sqrt(sum);
            } else {
                L(j, k) = sum / L(k, k);
            }
        }
    }
    for (int j = 0; j < n; j++) {
        log_det += std::log(L(j, j));
    }
    log_det *= 2.0;
}

// =====================================================================
// Generic family + link dispatch
// =====================================================================
//
// Decouples family from link via chain rule. Any family × any link works.
// Family string format: "family" (uses canonical/default link) or
// "family_link" (explicit link). Parsed once per Newton solve.
//
// Families: gaussian, binomial, poisson, neg_binomial_2, gamma, inverse_gaussian
// Links: identity, log, inverse, logit, probit, cauchit, cloglog, sqrt, 1/mu^2

struct FamilyLink {
    std::string family;
    std::string link;
};

inline FamilyLink parse_family_link(const std::string& code) {
    FamilyLink fl;
    // Default links per family
    static const std::pair<std::string, std::string> defaults[] = {
        {"gaussian",          "identity"},
        {"binomial",          "logit"},
        {"poisson",           "log"},
        {"neg_binomial_2",    "log"},
        {"gamma",             "log"},
        {"inverse_gaussian",  "log"},
    };
    // Check for explicit link suffix: "family_link"
    // Handle multi-word families first (neg_binomial_2, inverse_gaussian)
    for (auto& [fam, def_link] : defaults) {
        if (code == fam) { fl.family = fam; fl.link = def_link; return fl; }
        std::string prefix = fam + "_";
        if (code.substr(0, prefix.size()) == prefix && code.size() > prefix.size()) {
            // Check that the suffix is a known link, not part of the family name
            std::string suffix = code.substr(prefix.size());
            static const char* links[] = {
                "identity", "log", "inverse", "logit", "probit",
                "cauchit", "cloglog", "sqrt", "1mu2", nullptr
            };
            for (int i = 0; links[i]; i++) {
                if (suffix == links[i]) {
                    fl.family = fam; fl.link = suffix; return fl;
                }
            }
        }
    }
    // Fallback: treat whole string as family with default link
    fl.family = code;
    for (auto& [fam, def_link] : defaults) {
        if (code == fam) { fl.link = def_link; return fl; }
    }
    fl.link = "log"; // ultimate fallback
    return fl;
}

// --- Link function evaluators ---

inline double linkinv(double eta, const std::string& link) {
    if (link == "identity") return eta;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") return 1.0 / eta;
    if (link == "logit") {
        if (eta > 0) return 1.0 / (1.0 + std::exp(-eta));
        double e = std::exp(eta); return e / (1.0 + e);
    }
    if (link == "probit") return R::pnorm(eta, 0.0, 1.0, 1, 0);
    if (link == "cauchit") return 0.5 + std::atan(eta) / M_PI;
    if (link == "cloglog") return 1.0 - std::exp(-std::exp(eta));
    if (link == "sqrt") return eta * eta;
    if (link == "1mu2") return 1.0 / std::sqrt(eta);  // eta = 1/mu^2 > 0
    return tulpa_linalg::safe_exp(eta);
}

// dmu/deta
inline double mu_eta(double eta, const std::string& link) {
    if (link == "identity") return 1.0;
    if (link == "log") return tulpa_linalg::safe_exp(eta);
    if (link == "inverse") return -1.0 / (eta * eta);
    if (link == "logit") {
        double p;
        if (eta > 0) { double e = std::exp(-eta); p = 1.0 / (1.0 + e); }
        else { double e = std::exp(eta); p = e / (1.0 + e); }
        return p * (1.0 - p);
    }
    if (link == "probit") return R::dnorm(eta, 0.0, 1.0, 0);
    if (link == "cauchit") return 1.0 / (M_PI * (1.0 + eta * eta));
    if (link == "cloglog") return std::exp(eta - std::exp(eta));
    if (link == "sqrt") return 2.0 * eta;
    if (link == "1mu2") {
        // mu = eta^{-1/2}, dmu/deta = -1/(2*eta^{3/2})
        return -0.5 / (eta * std::sqrt(eta));
    }
    return tulpa_linalg::safe_exp(eta);
}

// --- Per-family functions on the mu scale ---

// Variance function V(mu, phi, n_trials): actual Var(Y) = V(mu, phi, n_trials)
inline double variance_fn(double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian")          return phi * phi;
    if (family == "binomial")          return n_trials * mu * (1.0 - mu);
    if (family == "poisson")           return mu;
    if (family == "neg_binomial_2")    return mu + mu * mu / phi;
    if (family == "gamma")             return mu * mu / phi;       // phi = shape
    if (family == "inverse_gaussian")  return phi * mu * mu * mu;  // phi = dispersion
    return mu;  // default: Poisson
}

// d(logL)/d(mu): gradient of log-likelihood w.r.t. the mean
inline double grad_mu(double y, double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian")          return (y - mu) / (phi * phi);
    if (family == "binomial")          return ((int)y - n_trials * mu) / (mu * (1.0 - mu));
    if (family == "poisson")           return (int)y / mu - 1.0;
    if (family == "neg_binomial_2") {
        return (int)y / mu - ((int)y + phi) / (mu + phi);
    }
    if (family == "gamma")             return phi * (y - mu) / (mu * mu);
    if (family == "inverse_gaussian")  return (y - mu) / (phi * mu * mu * mu);
    return (int)y / mu - 1.0;
}

// Log-likelihood on the mu scale (full, including normalization constants)
inline double log_lik_mu(double y, double mu, double phi, const std::string& family, int n_trials) {
    if (family == "gaussian") {
        double r = y - mu;
        return -0.5 * std::log(2.0 * M_PI * phi * phi) - r * r / (2.0 * phi * phi);
    }
    if (family == "binomial") {
        double p = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
        return (int)y * std::log(p) + (n_trials - (int)y) * std::log(1.0 - p);
    }
    if (family == "poisson") {
        double safe_mu = std::max(mu, 1e-15);
        return (int)y * std::log(safe_mu) - safe_mu - R::lgammafn((int)y + 1.0);
    }
    if (family == "neg_binomial_2") {
        double safe_mu = std::max(mu, 1e-15);
        return R::lgammafn((int)y + phi) - R::lgammafn(phi) - R::lgammafn((int)y + 1.0)
               + phi * std::log(phi / (safe_mu + phi))
               + (int)y * std::log(safe_mu / (safe_mu + phi));
    }
    if (family == "gamma") {
        // phi = shape
        return phi * std::log(phi) - R::lgammafn(phi) + (phi - 1.0) * std::log(y)
               - phi * std::log(mu) - phi * y / mu;
    }
    if (family == "inverse_gaussian") {
        double r = y - mu;
        return -0.5 * std::log(2.0 * M_PI * phi * y * y * y)
               - r * r / (2.0 * phi * mu * mu * y);
    }
    // default: Poisson
    double safe_mu = std::max(mu, 1e-15);
    return (int)y * std::log(safe_mu) - safe_mu - R::lgammafn((int)y + 1.0);
}

// =====================================================================
// Combined dispatch: grad, neg-hess, and log-lik per observation
// =====================================================================

struct GradHess {
    double grad;
    double neg_hess;
};

inline GradHess grad_hess_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi
) {
    // Fast path for the legacy family strings (canonical links, optimized)
    if (family == "binomial") {
        GradHess gh;
        gh.grad = grad_log_lik_binomial((int)y, n_trials, eta);
        gh.neg_hess = neg_hess_log_lik_binomial((int)y, n_trials, eta);
        return gh;
    }
    if (family == "poisson") {
        GradHess gh;
        gh.grad = grad_log_lik_poisson((int)y, eta);
        gh.neg_hess = neg_hess_log_lik_poisson((int)y, eta);
        return gh;
    }
    if (family == "neg_binomial_2") {
        GradHess gh;
        gh.grad = grad_log_lik_negbin((int)y, eta, phi);
        gh.neg_hess = neg_hess_log_lik_negbin((int)y, eta, phi);
        return gh;
    }

    // Generic path: family + link via chain rule
    // Parse is O(1) string comparisons — called per observation but branches predict well
    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    double dmu = mu_eta(eta, fl.link);

    // Clamp mu to valid range for the family
    if (fl.family == "binomial") mu = std::max(std::min(mu, 1.0 - 1e-7), 1e-7);
    else if (fl.family != "gaussian") mu = std::max(mu, 1e-10);

    double g = grad_mu(y, mu, phi, fl.family, n_trials);
    double V = variance_fn(mu, phi, fl.family, n_trials);

    GradHess gh;
    gh.grad = g * dmu;
    gh.neg_hess = dmu * dmu / V;    // Fisher scoring
    return gh;
}

// Log-likelihood for a single observation (for finalize).
inline double log_lik_for_family(
    double y, int n_trials, double eta,
    const std::string& family, double phi
) {
    // Fast path for canonical links
    if (family == "binomial") return log_lik_binomial((int)y, n_trials, eta);
    if (family == "poisson") return log_lik_poisson((int)y, eta);
    if (family == "neg_binomial_2") return log_lik_negbin((int)y, eta, phi);

    // Generic path
    FamilyLink fl = parse_family_link(family);
    double mu = linkinv(eta, fl.link);
    if (fl.family == "binomial") mu = std::max(std::min(mu, 1.0 - 1e-15), 1e-15);
    else if (fl.family != "gaussian") mu = std::max(mu, 1e-15);
    return log_lik_mu(y, mu, phi, fl.family, n_trials);
}

// Sum log-likelihood over all observations.
inline double compute_total_log_lik(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericVector& eta, int N,
    const std::string& family, double phi, int n_threads
) {
    double log_lik = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:log_lik) schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N; i++) {
        log_lik += log_lik_for_family(y[i], n_trials[i], eta[i], family, phi);
    }
    return log_lik;
}

// =====================================================================
// RE + beta prior helpers (already existed, made more generic)
// =====================================================================

// Add RE and beta priors to gradient and Hessian.
inline void add_re_beta_priors(
    DenseVec& grad, DenseMat& H,
    const Rcpp::NumericVector& x,
    int p, int n_re_groups, double tau_re
) {
    for (int g = 0; g < n_re_groups; g++) {
        grad[p + g] -= tau_re * x[p + g];
        H[p + g][p + g] += tau_re;
    }
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
        grad[j] -= tau_beta * x[j];
        H[j][j] += tau_beta;
    }
}

// Overload for NumericVector grad / NumericMatrix H (used by some variants).
inline void add_re_beta_priors(
    Rcpp::NumericVector& grad, Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int p, int n_re_groups, double tau_re
) {
    for (int g = 0; g < n_re_groups; g++) {
        grad[p + g] -= tau_re * x[p + g];
        H(p + g, p + g) += tau_re;
    }
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
        grad[j] -= tau_beta * x[j];
        H(j, j) += tau_beta;
    }
}

// Log prior for RE (for finalize).
inline double compute_log_prior_re(
    const Rcpp::NumericVector& x, int p, int n_re_groups, double tau_re
) {
    double log_prior = 0.0;
    for (int g = 0; g < n_re_groups; g++) {
        log_prior += -0.5 * tau_re * x[p + g] * x[p + g];
    }
    if (n_re_groups > 0) {
        log_prior += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
    }
    return log_prior;
}

// Center a block of effects (sum-to-zero constraint).
inline void center_effects(Rcpp::NumericVector& x, int start, int length) {
    if (length <= 0) return;
    double mean = 0.0;
    for (int i = 0; i < length; i++) mean += x[start + i];
    mean /= length;
    for (int i = 0; i < length; i++) x[start + i] -= mean;
}

// =====================================================================
// Shared scatter and prior functions (defined in laplace_core.cpp)
// =====================================================================

void scatter_obs_grad_hess_base(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta, const std::string& family, double phi,
    DenseVec& grad, DenseMat& H, int n_threads
);

void scatter_obs_with_latent(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta, const std::string& family, double phi,
    const std::vector<int>& effect_idx, const std::vector<double>& d_factors,
    DenseVec& grad, DenseMat& H, int n_threads
);

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

double log_prior_icar(
    const Rcpp::NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

void add_rw1_precision(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
);

void add_rw2_precision(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
);

void add_ar1_precision(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, double rho
);

// =====================================================================
// PIRLS-equivalent solver (Fisher scoring + step halving)
// =====================================================================
//
// Equivalent to lme4's PIRLS inner loop: Fisher scoring with deviance-
// based step halving for robust convergence on non-canonical links.
//
// Each model variant provides:
//   ComputeEta(x, eta)     — fill eta = X*beta + RE + model-specific effects
//   ScatterGradHess(x, eta, grad, H) — likelihood scatter + all priors
//   CenterEffects(x)       — post-convergence centering (sum-to-zero etc.)
//   ComputeLogPrior(x, eta) — log prior for finalize
//
// The solver handles: Fisher scoring, step halving, Cholesky solve,
// convergence, finalize (log-det, log-marginal).

constexpr int SPARSE_THRESHOLD = 200;
constexpr double SPARSE_DROP_TOL = 1e-12;
constexpr int MAX_HALVING = 12;  // max step halvings per iteration

template<typename ComputeEta, typename ScatterGradHess,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterGradHess scatter_grad_hess,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    const Rcpp::NumericVector& x_init = Rcpp::NumericVector(),
    SparseCholeskySolver* shared_solver = nullptr
) {
    LaplaceResult result;
    result.mode = Rcpp::NumericVector(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector x(n_x, 0.0);
    if (x_init.size() == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    }
    bool use_sparse = (n_x >= SPARSE_THRESHOLD);

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& sparse_solver = shared_solver ? *shared_solver : local_solver;

    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif

    // Penalized log-posterior: J(x) = log_lik(y|eta(x)) + log_prior(x)
    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        Rcpp::NumericVector eta_tmp(N, 0.0);
        compute_eta(xv, eta_tmp);
        double ll = compute_total_log_lik(y, n_trials, eta_tmp, N, family, phi, n_threads);
        double lp = compute_log_prior(xv, eta_tmp);
        return ll + lp;
    };

    // Cholesky solve helper (returns success flag, fills delta)
    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        bool ok = false;
        if (use_sparse) {
            cholmod_sparse* A = dense_to_cholmod_sparse_drop(
                H, n_x, SPARSE_DROP_TOL, &sparse_solver.common());
            if (A) {
                if (!sparse_solver.analyzed()) sparse_solver.analyze(A);
                if (sparse_solver.factorize(A)) {
                    sparse_solver.solve(grad.data(), delta.data(), n_x);
                    ok = true;
                    for (int j = 0; j < n_x; j++) {
                        if (!std::isfinite(delta[j])) { ok = false; break; }
                    }
                }
                M_cholmod_free_sparse(&A, &sparse_solver.common());
            }
            if (!ok) {
                auto chol = dense_cholesky_solve(H, grad, n_x);
                ok = chol.success;
                for (int j = 0; j < n_x; j++) delta[j] = chol.delta[j];
            }
        } else {
            auto chol = dense_cholesky_solve(H, grad, n_x);
            ok = chol.success;
            for (int j = 0; j < n_x; j++) delta[j] = chol.delta[j];
        }
        return ok;
    };

    // Current objective (computed lazily on first use)
    double obj_current = -1e300;
    bool obj_valid = false;

    // --- PIRLS iteration (Fisher scoring + step halving) ---
    for (int iter = 0; iter < max_iter; iter++) {
        // 1. Compute linear predictor
        Rcpp::NumericVector eta(N, 0.0);
        compute_eta(x, eta);

        // 2. Compute gradient and Hessian (Fisher-scored likelihood + priors)
        DenseVec grad(n_x, 0.0);
        DenseMat H(n_x, DenseVec(n_x, 0.0));
        scatter_grad_hess(x, eta, grad, H);

        // 3. Solve for Newton step: H * delta = grad
        std::vector<double> delta(n_x, 0.0);
        bool solve_ok = cholesky_solve(H, grad, delta);

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(delta[j])) x[j] += 0.1 * delta[j];
            }
            obj_valid = false;
            result.n_iter = iter + 1;
            continue;
        }

        // 4. Step halving: ensure penalized deviance decreases
        //    (equivalent to lme4's PIRLS step control)
        if (!obj_valid) {
            obj_current = eval_objective(x);
            obj_valid = true;
        }

        double step_scale = 1.0;
        bool accepted = false;
        for (int half = 0; half <= MAX_HALVING; half++) {
            Rcpp::NumericVector x_try(n_x);
            for (int j = 0; j < n_x; j++) x_try[j] = x[j] + step_scale * delta[j];

            double obj_try = eval_objective(x_try);

            if (obj_try >= obj_current - 1e-8 || half == MAX_HALVING) {
                // Accept step
                for (int j = 0; j < n_x; j++) x[j] = x_try[j];
                obj_current = obj_try;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
        }

        // 5. Check convergence
        double max_delta = 0.0;
        for (int j = 0; j < n_x; j++) {
            max_delta = std::max(max_delta, std::abs(step_scale * delta[j]));
        }

        result.n_iter = iter + 1;
        if (max_delta < tol) {
            result.converged = true;
            break;
        }
    }

    // --- Post-convergence centering ---
    center_effects_fn(x);

    // --- Finalize: log-det and log-marginal ---
    result.mode = x;

    Rcpp::NumericVector eta_final(N, 0.0);
    compute_eta(x, eta_final);

    DenseVec grad_final(n_x, 0.0);
    DenseMat H_final(n_x, DenseVec(n_x, 0.0));
    scatter_grad_hess(x, eta_final, grad_final, H_final);

    if (use_sparse) {
        cholmod_sparse* A_final = dense_to_cholmod_sparse_drop(
            H_final, n_x, SPARSE_DROP_TOL, &sparse_solver.common());
        if (A_final) {
            if (!sparse_solver.analyzed()) sparse_solver.analyze(A_final);
            if (sparse_solver.factorize(A_final)) {
                result.log_det_Q = sparse_solver.log_determinant();
            } else {
                Rcpp::NumericMatrix L_final(n_x, n_x);
                dense_cholesky_factorize(H_final, n_x, L_final, result.log_det_Q);
            }
            M_cholmod_free_sparse(&A_final, &sparse_solver.common());
        } else {
            Rcpp::NumericMatrix L_final(n_x, n_x);
            dense_cholesky_factorize(H_final, n_x, L_final, result.log_det_Q);
        }
    } else {
        Rcpp::NumericMatrix L_final(n_x, n_x);
        dense_cholesky_factorize(H_final, n_x, L_final, result.log_det_Q);
    }

    double log_lik = compute_total_log_lik(y, n_trials, eta_final, N, family, phi, n_threads);
    double log_prior = compute_log_prior(x, eta_final);

    result.log_marginal = log_lik + log_prior
                          - 0.5 * result.log_det_Q
                          + 0.5 * n_x * std::log(2.0 * M_PI);

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_HELPERS_H
