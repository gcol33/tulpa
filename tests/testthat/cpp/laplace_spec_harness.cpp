// laplace_spec_harness.cpp
#include <RcppEigen.h>

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_spec_fit.h"
#include "nested_laplace_grid.h"
#include "laplace_spec_solve.h"
#include <cmath>
#include <vector>

// ============================================================================
// Test harness: Gaussian LikelihoodSpec + iid RE driven through
// laplace_mode_spec_dense_impl. Lets the testthat suite exercise the
// spec-driven Laplace path against the family-enum reference without
// requiring a downstream package.
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
    if (has_re) {
        tulpa::nl_check_positive("sigma_re", sigma_re);
        params[layout.log_sigma_re_idx] = std::log(sigma_re);
    }

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

    // Return mode + diagnostics in the same shape as cpp_laplace_fit_multi_re's list.
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
    int n_threads = 1,
    int sparse_override = 0
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
    if (has_re) {
        tulpa::nl_check_positive("sigma_re", sigma_re);
        params[layout.log_sigma_re_idx] = std::log(sigma_re);
    }

    std::vector<int> re_group_1based;
    if (has_re) re_group_1based.assign(re_idx.begin(), re_idx.end());

    const tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        data, layout, params, re_group_1based, max_iter, tol, n_threads,
        /*blocks=*/nullptr, /*k_grid=*/0, /*beta_prior=*/nullptr,
        /*return_re_cov=*/false, sparse_override
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
        Rcpp::Named("log_det_Q")    = res.log_det_Q,
        Rcpp::Named("log_marginal") = res.log_marginal,
        Rcpp::Named("n_iter")       = res.n_iter,
        Rcpp::Named("converged")    = res.converged
    );
}

// ============================================================================
// Multi-term + slope RE test fixture. Accepts a list of RE terms
// (group_idx, n_groups, n_coefs, sigma vector, correlated, optional slope
// matrix, optional chol_raw vector) and runs the spec-Laplace path on a
// Gaussian likelihood. Returns the mode in the canonical concatenation
// order [beta | term_0 | term_1 | ...]. `sparse_override` forces the dense
// (< 0) or sparse (> 0) Newton path; `store_Q` adds the exported lower-CSC
// Hessian and `return_re_cov` the per-group marginal covariance blocks.
// ============================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_spec_test_multi_re(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::List re_terms,         // list of lists per term (see below)
    double sigma_beta,
    double phi,
    int max_iter = 200,
    double tol = 1e-12,
    int n_threads = 1,
    int sparse_override = 0,
    bool store_Q = false,
    bool return_re_cov = false
) {
    // Each element of re_terms is a list with fields:
    //   group_idx   IntegerVector length N, 1-based
    //   n_groups    int
    //   n_coefs     int (1 = intercept-only, >1 = intercept + slopes)
    //   sigma       NumericVector length n_coefs (positive sds)
    //   correlated  logical scalar
    //   slope_mat   NumericMatrix N x (n_coefs - 1) (or NULL if n_coefs == 1)
    //   chol_raw    NumericVector length n_coefs*(n_coefs-1)/2 (only when correlated)
    int N = y.size();
    int p = X.ncol();
    int K = re_terms.size();

    // ---- ProcessData ----
    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[(size_t)i * p + j] = X(i, j);
        }
    }

    tulpa::LikelihoodSpec spec;
    spec.n_processes    = 1;
    spec.name           = "gaussian_multire_test";
    spec.ll_double      = &gaussian_ll_T<double>;
    spec.eta_weights_fn = &gaussian_eta_weights;

    GaussianResponse resp{y.begin(), N, phi};

    tulpa::ModelData data;
    data.n_processes         = 1;
    data.processes.push_back(proc);
    data.N                   = N;
    data.sigma_beta          = sigma_beta;
    data.likelihood_spec     = &spec;
    data.model_response_data = &resp;
    data.sharing.init(1);

    // ---- Multi-term RE structure ----
    data.n_re_terms = K;
    data.re_parameterization = 1; // unused on the Laplace path (centered)
    data.re_n_groups_multi.assign(K, 0);
    data.re_n_coefs.assign(K, 1);
    data.re_n_slopes.assign(K, 0);
    data.re_has_intercept.assign(K, 1);  // this harness always carries intercept
    data.re_correlated.assign(K, false);
    data.re_n_chol.assign(K, 0);
    data.re_offsets.assign(K, 0);
    data.re_slope_matrices.assign(K, std::vector<double>());
    data.re_group_multi_flat.assign((size_t)N * K, 0);

    // Precompute total RE shape for asserts; also build manual ParamLayout.
    int total_re_groups = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;
    bool any_slopes = false;
    bool any_correlated = false;

    // First pass: pull static term shape from each list element.
    std::vector<Rcpp::NumericVector> sigma_per_term(K);
    std::vector<Rcpp::NumericVector> chol_per_term(K);
    for (int t = 0; t < K; t++) {
        Rcpp::List term = Rcpp::as<Rcpp::List>(re_terms[t]);
        Rcpp::IntegerVector gi = Rcpp::as<Rcpp::IntegerVector>(term["group_idx"]);
        int n_g = Rcpp::as<int>(term["n_groups"]);
        int n_c = Rcpp::as<int>(term["n_coefs"]);
        bool corr = Rcpp::as<bool>(term["correlated"]);
        if ((int)gi.size() != N) {
            Rcpp::stop("term %d: group_idx length %d != N (%d)", t + 1,
                       (int)gi.size(), N);
        }
        data.re_n_groups_multi[t] = n_g;
        data.re_n_coefs[t]        = n_c;
        data.re_n_slopes[t]       = std::max(0, n_c - 1);
        data.re_correlated[t]     = (n_c > 1) && corr;
        data.re_n_chol[t]         = data.re_correlated[t] ? n_c * (n_c - 1) / 2 : 0;
        data.re_offsets[t]        = total_re_groups;
        for (int i = 0; i < N; i++) {
            data.re_group_multi_flat[(size_t)i * K + t] = gi[i];
        }
        if (data.re_n_slopes[t] > 0) {
            any_slopes = true;
            Rcpp::NumericMatrix sm = Rcpp::as<Rcpp::NumericMatrix>(term["slope_mat"]);
            if (sm.nrow() != N || sm.ncol() != data.re_n_slopes[t]) {
                Rcpp::stop("term %d: slope_mat is %dx%d, expected %dx%d",
                           t + 1, sm.nrow(), sm.ncol(), N, data.re_n_slopes[t]);
            }
            data.re_slope_matrices[t].assign(
                (size_t)N * data.re_n_slopes[t], 0.0);
            for (int i = 0; i < N; i++) {
                for (int s = 0; s < data.re_n_slopes[t]; s++) {
                    data.re_slope_matrices[t][(size_t)i * data.re_n_slopes[t] + s]
                        = sm(i, s);
                }
            }
        }
        if (data.re_correlated[t]) any_correlated = true;
        sigma_per_term[t] = Rcpp::as<Rcpp::NumericVector>(term["sigma"]);
        if ((int)sigma_per_term[t].size() != n_c) {
            Rcpp::stop("term %d: sigma length %d != n_coefs (%d)", t + 1,
                       (int)sigma_per_term[t].size(), n_c);
        }
        if (data.re_correlated[t]) {
            chol_per_term[t] = Rcpp::as<Rcpp::NumericVector>(term["chol_raw"]);
            if ((int)chol_per_term[t].size() != data.re_n_chol[t]) {
                Rcpp::stop("term %d: chol_raw length %d != n_coefs*(n_coefs-1)/2 (%d)",
                           t + 1, (int)chol_per_term[t].size(), data.re_n_chol[t]);
            }
        }
        total_re_groups    += n_g;
        total_sigma_params += n_c;
        total_chol_params  += data.re_n_chol[t];
    }
    data.total_re_groups    = total_re_groups;
    data.total_re_params    = 0;
    for (int t = 0; t < K; t++) {
        data.total_re_params += data.re_n_groups_multi[t] * data.re_n_coefs[t];
    }
    data.total_sigma_params = total_sigma_params;
    data.total_chol_params  = total_chol_params;
    data.has_re_slopes              = any_slopes;
    data.has_re_correlated_slopes   = any_correlated;

    // ---- Build ParamLayout manually to mirror hmc_param_layout's
    // multi-term branches. We use the slopes-or-not branch because that's
    // the schema the new spec-Laplace path consumes.
    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    int next = p;
    layout.has_re                 = true;
    layout.has_re_slopes          = any_slopes;
    layout.has_re_correlated_slopes = any_correlated;
    layout.log_sigma_re_multi.resize(K);
    layout.log_sigma_re_slopes.resize(K);
    layout.re_start_multi.resize(K);
    layout.re_end_multi.resize(K);
    layout.re_n_coefs_multi.resize(K);
    layout.re_correlated_multi.resize(K);
    layout.chol_re_start_multi.assign(K, -1);
    layout.chol_re_end_multi.assign(K, -1);

    // 1. sigma slots (q_t per term)
    for (int t = 0; t < K; t++) {
        int q = data.re_n_coefs[t];
        layout.re_n_coefs_multi[t] = q;
        layout.re_correlated_multi[t] = data.re_correlated[t];
        layout.log_sigma_re_slopes[t].resize(q);
        for (int c = 0; c < q; c++) {
            layout.log_sigma_re_slopes[t][c] = next++;
        }
        layout.log_sigma_re_multi[t] = layout.log_sigma_re_slopes[t][0];
    }
    // 2. chol slots (only for correlated terms)
    for (int t = 0; t < K; t++) {
        if (data.re_n_chol[t] > 0) {
            layout.chol_re_start_multi[t] = next;
            next += data.re_n_chol[t];
            layout.chol_re_end_multi[t] = next;
        }
    }
    // 3. RE effects (group * coef per term)
    for (int t = 0; t < K; t++) {
        layout.re_start_multi[t] = next;
        next += data.re_n_groups_multi[t] * data.re_n_coefs[t];
        layout.re_end_multi[t] = next;
    }
    // legacy mirrors
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end   = layout.re_end_multi[0];
    layout.total_params = next;

    // ---- params: hyperparams from term spec, latent zero-init ----
    std::vector<double> params(layout.total_params, 0.0);
    for (int t = 0; t < K; t++) {
        int q = data.re_n_coefs[t];
        for (int c = 0; c < q; c++) {
            params[layout.log_sigma_re_slopes[t][c]] =
                std::log(sigma_per_term[t][c]);
        }
        if (data.re_correlated[t]) {
            for (int j = 0; j < data.re_n_chol[t]; j++) {
                params[layout.chol_re_start_multi[t] + j] = chol_per_term[t][j];
            }
        }
    }

    std::vector<int> empty_group;
    const tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        data, layout, params, empty_group, max_iter, tol, n_threads,
        /*blocks=*/nullptr, /*k_grid=*/0, /*beta_prior=*/nullptr,
        return_re_cov, sparse_override, store_Q
    );

    // Build mode = [beta | term_0_block | term_1_block | ...]
    int n_x = p;
    for (int t = 0; t < K; t++) n_x += data.re_n_groups_multi[t] * data.re_n_coefs[t];
    Rcpp::NumericVector mode(n_x);
    int off = 0;
    for (int j = 0; j < p; j++) mode[off++] = params[j];
    for (int t = 0; t < K; t++) {
        int q = data.re_n_coefs[t];
        int n_g = data.re_n_groups_multi[t];
        for (int j = 0; j < n_g * q; j++) {
            mode[off++] = params[layout.re_start_multi[t] + j];
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("mode")         = mode,
        Rcpp::Named("log_det_Q")    = res.log_det_Q,
        Rcpp::Named("log_marginal") = res.log_marginal,
        Rcpp::Named("n_iter")       = res.n_iter,
        Rcpp::Named("converged")    = res.converged,
        Rcpp::Named("Q_p")          = Rcpp::wrap(res.Q_csc_p),
        Rcpp::Named("Q_i")          = Rcpp::wrap(res.Q_csc_i),
        Rcpp::Named("Q_x")          = Rcpp::wrap(res.Q_csc_x),
        Rcpp::Named("re_cov")       = Rcpp::wrap(res.re_cov_flat)
    );
}
