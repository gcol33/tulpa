// laplace_core.cpp
// Core Laplace approximation engine for tulpa
// Implements Laplace approximation for latent Gaussian models.
// All model-specific Laplace functions run through the laplace_newton_solve()
// template in laplace_newton.h.

#include "laplace_core.h"
#include "laplace_cholesky.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_spec_fit.h"     // spec-solver marshalling for the single-point fits
#include "re_structure.h"         // shared multi-term RE ModelData marshalling
#include "laplace_scatter.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "linalg_fast.h"
#include "gpu_nngp_laplace.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

// =====================================================================
// GP / NNGP helpers (moved from inline, unchanged)
// =====================================================================

inline double compute_cov_exp_laplace(double d, double sigma2, double phi) {
    if (d < 1e-10) return sigma2;
    return sigma2 * std::exp(-d / phi);
}

inline double compute_cov_matern15_laplace(double d, double sigma2, double phi) {
    if (d < 1e-10) return sigma2;
    double x = std::sqrt(3.0) * d / phi;
    return sigma2 * (1.0 + x) * std::exp(-x);
}

inline double compute_cov_matern25_laplace(double d, double sigma2, double phi) {
    if (d < 1e-10) return sigma2;
    double x = std::sqrt(5.0) * d / phi;
    return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

inline double compute_cov_laplace(double d, double sigma2, double phi, int cov_type) {
    if (cov_type == 0) return compute_cov_exp_laplace(d, sigma2, phi);
    if (cov_type == 1) return compute_cov_matern15_laplace(d, sigma2, phi);
    return compute_cov_matern25_laplace(d, sigma2, phi);
}

inline void nngp_conditional_laplace(
    int obs_idx, int i,
    const std::vector<double>& w,
    double sigma2, double phi_gp, int cov_type,
    const NumericMatrix& coords,
    const IntegerMatrix& nn_idx, const NumericMatrix& nn_dist,
    const IntegerVector& nn_order, int nn,
    double& cond_mean, double& cond_var
) {
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
        if (nn_idx(i, j) > 0) n_neighbors++;
    }
    if (n_neighbors == 0) {
        cond_mean = 0.0;
        cond_var = sigma2;
        return;
    }
    // The positive neighbour indices are front-contiguous by construction; the
    // loops below index columns 0..n_neighbors-1 as the neighbours. Guard an
    // interior zero (a malformed neighbour row) that would index nn_order[-1].
    for (int j = 0; j < n_neighbors; j++) {
        if (nn_idx(i, j) <= 0) { cond_mean = 0.0; cond_var = sigma2; return; }
    }

    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    for (int j = 0; j < n_neighbors; j++) {
        c_vec[j] = compute_cov_laplace(nn_dist(i, j), sigma2, phi_gp, cov_type);
    }
    for (int j1 = 0; j1 < n_neighbors; j1++) {
        int nn_orig1 = nn_order[nn_idx(i, j1) - 1];
        for (int j2 = 0; j2 < n_neighbors; j2++) {
            int nn_orig2 = nn_order[nn_idx(i, j2) - 1];
            if (j1 == j2) {
                C_mat[j1 * n_neighbors + j2] = sigma2;
            } else {
                double d12 = std::sqrt(
                    std::pow(coords(nn_orig1, 0) - coords(nn_orig2, 0), 2) +
                    std::pow(coords(nn_orig1, 1) - coords(nn_orig2, 1), 2)
                );
                C_mat[j1 * n_neighbors + j2] = compute_cov_laplace(d12, sigma2, phi_gp, cov_type);
            }
        }
    }

    // Gather neighbor values in c_vec order, then shared factor/solve core
    std::vector<double> w_nb(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
        int nn_orig = nn_order[nn_idx(i, j) - 1];
        w_nb[j] = w[nn_orig];
    }
    tulpa_linalg::nngp_conditional_moments(
        C_mat.data(), c_vec.data(), w_nb.data(), n_neighbors, sigma2,
        tulpa_linalg::kCholJitter, tulpa_linalg::kCholJitter,
        cond_mean, cond_var);
}

} // namespace tulpa

// =====================================================================
// R exports (call into tulpa:: functions defined above)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    // Fixed effects + optional single iid RE, through the unified spec solver
    // (the family-enum laplace_mode_dense was retired in B2-live). sigma_beta =
    // 100 reproduces the historical weak ridge tau_beta = 1e-4 = DEFAULT_TAU_BETA.
    const int N = y.size();
    std::vector<int> re_group;
    if (n_re_groups > 0) {
        re_group.resize(N);
        for (int i = 0; i < N; i++) re_group[i] = (int)re_idx[i];
    }
    tulpa::SpecFamilyInputs in;
    tulpa::build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/100.0, /*n_block_latent=*/0);
    std::vector<double> params(in.layout.total_params, 0.0);
    if (in.layout.has_re) params[in.layout.log_sigma_re_idx] = std::log(sigma_re);
    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads);
    return tulpa::laplace_result_to_list(res);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multi_re(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::List re_idx_list,
    Rcpp::IntegerVector re_ngroups,
    Rcpp::List re_sigma_list,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::List> re_Z_list = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> x_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> beta_prior_mean = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> beta_prior_sd = R_NilValue,
    bool return_re_cov = false,
    double phi2 = NA_REAL
) {
    // Multi-term RE (intercept / slopes / correlated) + built-in family through
    // the unified spec solver (the family-enum laplace_mode_dense_multi_re was
    // retired in B2-live). Marshals the R-facing inputs into the multi-term
    // ModelData / ParamLayout the spec path consumes, converting each term's
    // `pack` (marginal SDs or a packed Sigma-Cholesky) into the spec log-Cholesky
    // parameterization, and preserves every input: weights, offset, the full
    // per-coef beta prior, the warm start, and the EM M-step covariance blocks.
    const int N = y.size();
    const int p = X.ncol();
    const int K = re_ngroups.size();

    // --- Optional per-coef Gaussian fixed-effect prior (mean / sd -> tau). ---
    tulpa::BetaPrior bp;
    bool has_bp = false;
    if (beta_prior_sd.isNotNull()) {
        Rcpp::NumericVector sd = Rcpp::as<Rcpp::NumericVector>(beta_prior_sd);
        if ((int)sd.size() != p) {
            Rcpp::stop("beta_prior_sd has length %d but X has %d columns.",
                       (int)sd.size(), p);
        }
        bp.tau.resize(p);
        for (int j = 0; j < p; j++) {
            if (!(sd[j] > 0.0) || ISNAN(sd[j])) {
                Rcpp::stop("beta_prior_sd[%d] must be positive (Inf allowed = no penalty).", j + 1);
            }
            // sd = +Inf -> tau = 0 (no penalty); BetaPrior::tau_at returns it
            // and the spec beta-prior treats tau = 0 as a no-op.
            bp.tau[j] = R_finite(sd[j]) ? 1.0 / (sd[j] * sd[j]) : 0.0;
        }
        has_bp = true;
    }
    if (beta_prior_mean.isNotNull()) {
        Rcpp::NumericVector mn = Rcpp::as<Rcpp::NumericVector>(beta_prior_mean);
        if ((int)mn.size() != p) {
            Rcpp::stop("beta_prior_mean has length %d but X has %d columns.",
                       (int)mn.size(), p);
        }
        bp.mean.assign(mn.begin(), mn.end());
        has_bp = true;
    }

    // --- Per-obs likelihood weights (borrowed; the family adapter scales the
    //     score + Fisher weight). Stable storage outlives the solve. ---
    std::vector<double> w_store;
    const double* w_ptr = nullptr;
    if (weights.isNotNull()) {
        Rcpp::NumericVector wv = Rcpp::as<Rcpp::NumericVector>(weights);
        w_store.assign(wv.begin(), wv.end());
        w_ptr = w_store.data();
    }

    // --- n_coefs per term (default 1 = intercept only). ---
    std::vector<int> ncoefs(K, 1);
    if (re_ncoefs.isNotNull()) {
        Rcpp::IntegerVector nc = Rcpp::as<Rcpp::IntegerVector>(re_ncoefs);
        for (int t = 0; t < K; t++) ncoefs[t] = nc[t];
    }

    // --- Process design + built-in family spec + response. ---
    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < p; j++)
            proc.X_flat[(size_t)i * p + j] = X(i, j);
    if (offset.isNotNull()) {
        Rcpp::NumericVector ov = Rcpp::as<Rcpp::NumericVector>(offset);
        proc.offset.assign(ov.begin(), ov.end());
    }

    tulpa::LikelihoodSpec spec = tulpa::builtin_family_spec(family);
    std::vector<int> n_trials(n.begin(), n.end());
    tulpa::BuiltinFamilyResponse resp;
    resp.y        = y.begin();
    resp.n_trials = n_trials.data();
    resp.N        = N;
    resp.family   = family;
    resp.phi      = phi;
    resp.phi2     = phi2;   // NA_REAL is a NaN => family default (e.g. t df = 4)
    resp.weights  = w_ptr;

    tulpa::ModelData data;
    data.n_processes         = 1;
    data.processes.push_back(proc);
    data.N                   = N;
    data.sigma_beta          = 100.0;   // default ridge (tau = 1e-4); overridden by bp
    data.likelihood_spec     = &spec;
    data.model_response_data = &resp;
    data.sharing.init(1);

    // --- Multi-term RE structure (shared marshalling, re_structure.h). ---
    data.re_parameterization = 1;       // unused on the centered Laplace path
    // Correlated when a packed Sigma-Cholesky (length q(q+1)/2) is supplied;
    // otherwise diagonal (length-q marginal SDs).
    std::vector<bool> corr_flags(K, false);
    for (int t = 0; t < K; t++) {
        Rcpp::NumericVector sig = Rcpp::as<Rcpp::NumericVector>(re_sigma_list[t]);
        corr_flags[t] = (ncoefs[t] > 1) && ((int)sig.size() == ncoefs[t] * (ncoefs[t] + 1) / 2);
    }
    tulpa::populate_re_structure(
        data, N, re_idx_list,
        std::vector<int>(re_ngroups.begin(), re_ngroups.end()),
        ncoefs, re_Z_list, corr_flags);

    // Per-term sigma packing: convert each term's `pack` (marginal SDs or a
    // packed Sigma-Cholesky) into the spec log-Cholesky parameterization the
    // params vector below carries.
    std::vector<std::vector<double>> term_log_sigma(K), term_tanh_raw(K);
    for (int t = 0; t < K; t++) {
        Rcpp::NumericVector sig = Rcpp::as<Rcpp::NumericVector>(re_sigma_list[t]);
        tulpa::pack_to_spec_re_params(sig.begin(), ncoefs[t], data.re_correlated[t],
                                      term_log_sigma[t], term_tanh_raw[t]);
    }

    // --- ParamLayout: [beta | sigma slots | chol slots | RE effects], the
    //     schema build_latent_layout reads (mirrors hmc_param_layout). ---
    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    int next = p;
    layout.has_re                   = (K > 0);   // fixed-effects-only when no RE terms
    layout.has_re_slopes            = data.has_re_slopes;
    layout.has_re_correlated_slopes = data.has_re_correlated_slopes;
    layout.log_sigma_re_multi.resize(K);
    layout.log_sigma_re_slopes.resize(K);
    layout.re_start_multi.resize(K);
    layout.re_end_multi.resize(K);
    layout.re_n_coefs_multi.resize(K);
    layout.re_correlated_multi.resize(K);
    layout.chol_re_start_multi.assign(K, -1);
    layout.chol_re_end_multi.assign(K, -1);
    for (int t = 0; t < K; t++) {
        const int q = ncoefs[t];
        layout.re_n_coefs_multi[t]    = q;
        layout.re_correlated_multi[t] = data.re_correlated[t];
        layout.log_sigma_re_slopes[t].resize(q);
        for (int c = 0; c < q; c++) layout.log_sigma_re_slopes[t][c] = next++;
        layout.log_sigma_re_multi[t]  = layout.log_sigma_re_slopes[t][0];
    }
    for (int t = 0; t < K; t++) {
        if (data.re_n_chol[t] > 0) {
            layout.chol_re_start_multi[t] = next;
            next += data.re_n_chol[t];
            layout.chol_re_end_multi[t] = next;
        }
    }
    for (int t = 0; t < K; t++) {
        layout.re_start_multi[t] = next;
        next += data.re_n_groups_multi[t] * data.re_n_coefs[t];
        layout.re_end_multi[t] = next;
    }
    layout.log_sigma_re_idx = K > 0 ? layout.log_sigma_re_multi[0] : -1;
    layout.re_start = K > 0 ? layout.re_start_multi[0] : -1;
    layout.re_end   = K > 0 ? layout.re_end_multi[0]   : -1;
    layout.total_params = next;

    // --- params: hyperparameter slots from each term's converted spec params,
    //     latent slots from the optional warm start (mode order [beta | terms]). ---
    std::vector<double> params(layout.total_params, 0.0);
    for (int t = 0; t < K; t++) {
        const int q = ncoefs[t];
        for (int c = 0; c < q; c++)
            params[layout.log_sigma_re_slopes[t][c]] = term_log_sigma[t][c];
        if (data.re_n_chol[t] > 0)
            for (int j = 0; j < data.re_n_chol[t]; j++)
                params[layout.chol_re_start_multi[t] + j] = term_tanh_raw[t][j];
    }
    if (x_init.isNotNull()) {
        Rcpp::NumericVector xi = Rcpp::as<Rcpp::NumericVector>(x_init);
        int off = 0;
        for (int j = 0; j < p; j++) params[j] = xi[off++];
        for (int t = 0; t < K; t++) {
            const int sz = data.re_n_groups_multi[t] * data.re_n_coefs[t];
            for (int j = 0; j < sz; j++) params[layout.re_start_multi[t] + j] = xi[off++];
        }
    }

    std::vector<int> re_group_empty;   // groups come from re_group_multi_flat
    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        data, layout, params, re_group_empty, max_iter, tol, n_threads,
        /*blocks=*/nullptr, /*k_grid=*/0,
        has_bp ? &bp : nullptr, return_re_cov);
    return tulpa::laplace_result_to_list(res);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_laplace_sample(
    Rcpp::NumericVector mode, Rcpp::NumericMatrix H, int n_samples
) {
    int n_x = mode.size();
    Rcpp::NumericMatrix samples(n_samples, n_x);

    // Cholesky of H + ridge*I. The same uniform upstream regularization
    // every Laplace solve uses (see LAPLACE_UNIFORM_RIDGE in
    // laplace_cholesky.h); guarantees PD on rank-deficient priors so the
    // sampler never hits a non-positive pivot.
    for (int j = 0; j < n_x; j++) H(j, j) += tulpa::LAPLACE_UNIFORM_RIDGE;
    Rcpp::NumericMatrix L(n_x, n_x);
    double log_det;
    tulpa::dense_cholesky_factorize(H, n_x, L, log_det);

    // Sample: z ~ N(0, I), x = mode + L^{-T} z
    for (int s = 0; s < n_samples; s++) {
        Rcpp::NumericVector z(n_x);
        for (int j = 0; j < n_x; j++) z[j] = R::rnorm(0.0, 1.0);

        // Solve L' x_centered = z (back substitution)
        Rcpp::NumericVector x_centered(n_x);
        for (int j = n_x - 1; j >= 0; j--) {
            double sum = z[j];
            for (int k = j + 1; k < n_x; k++) sum -= L(k, j) * x_centered[k];
            x_centered[j] = sum / L(j, j);
        }

        for (int j = 0; j < n_x; j++) samples(s, j) = mode[j] + x_centered[j];
    }
    return samples;
}

// Spatial / BYM2 / RSR mode finders + their R exports moved to laplace_core_spatial.cpp.
// GP / Multiscale GP / Multiscale temporal mode finders + their R exports moved to laplace_core_gp.cpp.
// Nested Laplace and SPDE code lives in nested_laplace.cpp and spde_laplace.cpp.
