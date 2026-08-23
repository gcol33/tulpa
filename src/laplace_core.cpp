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
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "linalg_fast.h"
#include "gpu_nngp_laplace.h"
#include "sparse_hessian.h"
#include "nested_laplace_grid.h"   // nl_check_positive
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// =====================================================================
// R exports
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue
) {
    // Fixed effects + optional single iid RE, through the unified spec solver.
    // sigma_beta = 100 is the weak ridge tau_beta = 1e-4 = DEFAULT_TAU_BETA.
    const int N = y.size();
    std::vector<int> re_group =
        tulpa::as_re_group_vec(re_idx, n_re_groups, N);
    tulpa::SpecFamilyInputs in;
    tulpa::build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/100.0, /*n_block_latent=*/0);
    std::vector<double> params(in.layout.total_params, 0.0);
    if (in.layout.has_re) {
        tulpa::nl_check_positive("sigma_re", sigma_re);
        params[in.layout.log_sigma_re_idx] = std::log(sigma_re);
    }
    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);
    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads,
        /*blocks=*/nullptr, /*k_grid=*/0, /*beta_prior=*/nullptr,
        /*return_re_cov=*/false, /*sparse_override=*/0, /*store_Q=*/false,
        compute_skew, skew_idx_ptr);
    return tulpa::laplace_result_to_list(res);
}

// `debias` is the subspace-debias request: a list carrying
// `idx` (the 1-based latent index set to correct by Metropolis along the
// Gaussian-conditional-mean surface) and the optional sweep budget `n_iter` /
// `warmup` / `thin`. An absent or empty index set leaves the solve bit-for-bit
// as it was and consumes no random number.
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
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::NumericMatrix> X_zi = R_NilValue,
    double zi_prior_sd = 2.5,
    bool return_joint_hessian = false,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue,
    Rcpp::Nullable<Rcpp::List> debias = R_NilValue
) {
    // Multi-term RE (intercept / slopes / correlated) + built-in family through
    // the unified spec solver. Marshals the R-facing inputs into the multi-term
    // ModelData / ParamLayout the spec path consumes, converting each term's
    // `pack` (marginal SDs or a packed Sigma-Cholesky) into the spec log-Cholesky
    // parameterization, and preserves every input: weights, offset, the full
    // per-coef beta prior, the warm start, and the EM M-step covariance blocks.
    const int N = y.size();
    const int p = X.ncol();
    const int K = re_ngroups.size();

    // This entry marshals ModelData by hand rather than through
    // build_spec_family_inputs, so the length checks that path carries are made
    // here. Everything below indexes by N or by K, both read off arguments the
    // caller supplies separately.
    tulpa::check_arg_length(X.nrow(), N, "nrow(X)", "length(y)");
    tulpa::check_arg_length(n.size(), N, "length(n)", "length(y)");
    tulpa::check_arg_length(re_sigma_list.size(), K, "length(re_sigma_list)",
                            "length(re_ngroups)");

    // --- Zero inflation: the spec-Laplace shim passes 0.0 for the `logit_zi`
    //     callback argument, so the ZI linear predictor is carried as a second
    //     PROCESS (eta[1] = X_zi beta_zi) rather than through data.zi_type's
    //     side channel. data.zi_type therefore stays NONE here -- it means "the
    //     side channel is live", which is the sampler paths' mechanism, not
    //     this one. The beta block becomes [beta_count | beta_zi]. ---
    Rcpp::NumericMatrix X_zi_mat;
    int p_zi = 0;
    if (X_zi.isNotNull()) {
        X_zi_mat = Rcpp::as<Rcpp::NumericMatrix>(X_zi);
        if (X_zi_mat.nrow() != N) {
            Rcpp::stop("X_zi has %d rows but y has length %d.",
                       (int)X_zi_mat.nrow(), N);
        }
        p_zi = X_zi_mat.ncol();
    }
    const bool has_zi = (p_zi > 0);
    if (has_zi && !tulpa::zi::compiled_zi_supported(family)) {
        Rcpp::stop("family '%s' has no compiled zero-inflated kernel "
                   "(supported: %s).",
                   family.c_str(),
                   tulpa::zi::compiled_zi_supported_families().c_str());
    }
    const int p_total = p + p_zi;

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
    // The beta prior is supplied for the count block only; the ZI block is
    // appended with a weakly-informative N(0, zi_prior_sd^2) matching
    // ModelData::zi_prior_sd, which keeps the logit identified when a level has
    // no zeros (where the likelihood alone would send beta_zi to -Inf). It is
    // applied whenever the ZI block exists: it is the ZI block's own prior, not
    // an extension of the caller's count-block prior, and the sampler paths
    // carry it unconditionally through ModelData::zi_prior_sd. Materializing
    // `tau` here means the count block keeps DEFAULT_TAU_BETA explicitly rather
    // than through the empty-vector fallback, which cannot represent two
    // different precisions.
    if (has_zi) {
        if (!(zi_prior_sd > 0.0) || ISNAN(zi_prior_sd)) {
            Rcpp::stop("zi_prior_sd must be positive (Inf allowed = no penalty).");
        }
        const double tau_zi = R_finite(zi_prior_sd)
            ? 1.0 / (zi_prior_sd * zi_prior_sd) : 0.0;
        if (bp.tau.empty()) bp.tau.assign(p, tulpa::DEFAULT_TAU_BETA);
        bp.tau.resize(p_total, tau_zi);
        if (!bp.mean.empty()) bp.mean.resize(p_total, 0.0);
    }

    // --- Per-obs likelihood weights (borrowed; the family adapter scales the
    //     score + Fisher weight). Stable storage outlives the solve. ---
    std::vector<double> w_store;
    const double* w_ptr = nullptr;
    if (weights.isNotNull()) {
        Rcpp::NumericVector wv = Rcpp::as<Rcpp::NumericVector>(weights);
        tulpa::check_arg_length(wv.size(), N, "length(weights)", "length(y)");
        w_store.assign(wv.begin(), wv.end());
        w_ptr = w_store.data();
    }

    // --- n_coefs per term (default 1 = intercept only). ---
    std::vector<int> ncoefs(K, 1);
    if (re_ncoefs.isNotNull()) {
        Rcpp::IntegerVector nc = Rcpp::as<Rcpp::IntegerVector>(re_ncoefs);
        tulpa::check_arg_length(nc.size(), K, "length(re_ncoefs)",
                                "length(re_ngroups)");
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
        tulpa::check_arg_length(ov.size(), N, "length(offset)", "length(y)");
        proc.offset.assign(ov.begin(), ov.end());
    }

    tulpa::ProcessData proc_zi;
    if (has_zi) {
        proc_zi.p = p_zi;
        proc_zi.X_flat.resize((size_t)N * p_zi);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < p_zi; j++)
                proc_zi.X_flat[(size_t)i * p_zi + j] = X_zi_mat(i, j);
    }

    tulpa::LikelihoodSpec spec = tulpa::builtin_family_spec(family, has_zi);
    std::vector<int> n_trials(n.begin(), n.end());
    tulpa::BuiltinFamilyResponse resp;
    resp.y        = y.begin();
    resp.n_trials = n_trials.data();
    resp.N        = N;
    resp.family   = family;
    resp.phi      = phi;
    resp.phi2     = phi2;   // NA_REAL is a NaN => family default (e.g. t df = 4)
    resp.weights  = w_ptr;
    resp.prepare();

    tulpa::ModelData data;
    data.n_processes         = has_zi ? 2 : 1;
    data.processes.push_back(proc);
    if (has_zi) data.processes.push_back(proc_zi);
    data.N                   = N;
    data.sigma_beta          = 100.0;   // default ridge (tau = 1e-4); overridden by bp
    data.likelihood_spec     = &spec;
    data.model_response_data = &resp;
    data.sharing.init(data.n_processes);
    // Random effects enter the count predictor only. A ZI predictor with its
    // own random effects is a separate feature; sharing the count RE into it
    // would silently impose equal effects on both, which is not the model.
    if (has_zi) data.sharing.re[1] = false;

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
        const int q = ncoefs[t];
        const R_xlen_t need = data.re_correlated[t]
            ? (R_xlen_t)q * (q + 1) / 2
            : (R_xlen_t)q;
        if (sig.size() != need) {
            Rcpp::stop("re_sigma_list[[%d]] must have length %d for a %s "
                       "term with %d coefficient(s), got %d.",
                       t + 1, (int)need,
                       data.re_correlated[t] ? "correlated" : "diagonal",
                       q, (int)sig.size());
        }
        const std::string label =
            "re_sigma_list[[" + std::to_string(t + 1) + "]]";
        tulpa::pack_to_spec_re_params(sig.begin(), q, data.re_correlated[t],
                                      term_log_sigma[t], term_tanh_raw[t],
                                      label.c_str());
    }

    // --- ParamLayout: [beta | sigma slots | chol slots | RE effects], the
    //     schema build_latent_layout reads (mirrors hmc_param_layout). ---
    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    if (has_zi) {
        layout.process_beta_start.push_back(p);
        layout.process_beta_count.push_back(p_zi);
    }
    int next = p_total;
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
        // The warm start is [beta | per-term RE effects]; the RE half is only
        // known once populate_re_structure has resolved each term's groups and
        // coefficients, which is why the length is checked here.
        R_xlen_t need = p_total;
        for (int t = 0; t < K; t++)
            need += (R_xlen_t)data.re_n_groups_multi[t] * data.re_n_coefs[t];
        tulpa::check_arg_length(xi.size(), need, "length(x_init)",
                                "the latent dimension");
        int off = 0;
        for (int j = 0; j < p_total; j++) params[j] = xi[off++];
        for (int t = 0; t < K; t++) {
            const int sz = data.re_n_groups_multi[t] * data.re_n_coefs[t];
            for (int j = 0; j < sz; j++) params[layout.re_start_multi[t] + j] = xi[off++];
        }
    }

    std::vector<int> re_group_empty;   // groups come from re_group_multi_flat
    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);
    tulpa::SubspaceDebiasOptions db_opts;
    const tulpa::SubspaceDebiasOptions* db_ptr =
        tulpa::unwrap_debias(debias, db_opts);
    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        data, layout, params, re_group_empty, max_iter, tol, n_threads,
        /*blocks=*/nullptr, /*k_grid=*/0,
        (has_bp || has_zi) ? &bp : nullptr, return_re_cov,
        /*sparse_override=*/0, return_joint_hessian,
        compute_skew, skew_idx_ptr, db_ptr);
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
    //
    // The ridge goes on a clone. Rcpp binds a REALSXP argument without
    // duplicating it, so ridging `H` in place writes the ridge into the R
    // matrix the caller still holds, and a second call on the same matrix
    // samples from a precision carrying the ridge twice.
    Rcpp::NumericMatrix Hr = Rcpp::clone(H);
    for (int j = 0; j < n_x; j++) Hr(j, j) += tulpa::LAPLACE_UNIFORM_RIDGE;
    Rcpp::NumericMatrix L(n_x, n_x);
    double log_det;
    tulpa::dense_cholesky_factorize(Hr, n_x, L, log_det);

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

// Spatial / BYM2 / RSR mode finders and their R exports live in
// laplace_core_spatial.cpp, the GP / multiscale GP / multiscale temporal ones in
// laplace_core_gp.cpp, and nested Laplace / SPDE in nested_laplace.cpp and
// spde_laplace.cpp.
