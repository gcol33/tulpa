// tulpa_generic_sampler.cpp
// Proof-of-concept: generic model fitting via LikelihoodFn callback
// Demonstrates the tulpa multi-process interface end-to-end

#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <vector>
#include <cmath>
#include <random>
#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "linalg_fast.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"
#include "tulpa/nuts_api.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// ============================================================================
// Gaussian response data (model-specific, opaque to tulpa engine)
// ============================================================================
struct GaussianData {
    std::vector<double> y;  // Response vector
};

// ============================================================================
// Gaussian log-likelihood (templated for autodiff compatibility)
// ============================================================================
template<typename T>
T gaussian_likelihood(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& logit_oi,
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data
) {
    const auto* gd = static_cast<const GaussianData*>(model_data);
    T residual = T(gd->y[i]) - eta[0];
    T log_sigma = params[layout.extra_offset];  // log(residual SD)
    // Normal log-density on log scale (no exp needed):
    // log N(y|mu,sigma) = -log_sigma - 0.5*(y-mu)^2/sigma^2
    // = -log_sigma - 0.5*(y-mu)^2 * exp(-2*log_sigma)
    // Rewrite to avoid sigma^2: use -log_sigma - 0.5 * residual^2 * exp(-2*log_sigma)
    T neg_log_sigma = T(0.0) - log_sigma;
    // Precision = 1/sigma^2 = exp(-2*log_sigma). For AD types, log_post_impl.h
    // has safe_exp overloads. Use the two_log_sigma form directly:
    T half_resid_sq = T(0.5) * residual * residual;
    // log-density = -log_sigma - half_resid_sq * exp(-2*log_sigma)
    // We express exp(-2*log_sigma) as 1/sigma^2 without needing exp at all:
    // since the only thing that matters for MCMC is log-density up to constants,
    // we can parametrize as tau = log(precision) = -2*log_sigma:
    //   log p = 0.5*tau - 0.5*exp(tau)*residual^2
    // But this still needs exp. The simplest AD-compatible approach:
    // Write -log_sigma - 0.5*r^2/sigma^2 = -log_sigma - 0.5*r^2*exp(-2*log_sigma)
    // The arena::exp and fwd::exp are available via ADL when T is arena::Var / fwd::Dual
    T neg_two_ls = neg_log_sigma + neg_log_sigma;
    // For T=double: use std::exp; for T=arena::Var: uses tulpa::arena::exp via ADL
    using std::exp;
    T precision = exp(neg_two_ls);
    return neg_log_sigma - half_resid_sq * precision;
}

// ============================================================================
// Log-posterior for Gaussian model (wraps compute_log_post_generic)
// ============================================================================
static double compute_gaussian_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
    return tulpa::compute_log_post_generic<double>(
        params, data, layout,
        gaussian_likelihood<double>,
        data.model_response_data
    );
}

// ============================================================================
// Numerical gradient for generic models (central differences)
// ============================================================================
static void gradient_numerical_generic(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    double f0 = compute_gaussian_log_post(params, data, layout);
    if (log_post_out) *log_post_out = f0;

    const double eps = 1e-6;
    const int p = static_cast<int>(params.size());
    std::vector<double> params_work = params;

    for (int j = 0; j < p; j++) {
        params_work[j] = params[j] + eps;
        double f_plus = compute_gaussian_log_post(params_work, data, layout);
        params_work[j] = params[j] - eps;
        double f_minus = compute_gaussian_log_post(params_work, data, layout);
        params_work[j] = params[j];
        grad[j] = (f_plus - f_minus) / (2.0 * eps);
    }
}

// ============================================================================
// Simple HMC sampler (proof-of-concept, not production NUTS)
// Fixed step size, fixed number of leapfrog steps
// ============================================================================
static std::vector<std::vector<double>> run_hmc_simple(
    const std::vector<double>& init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    double step_size,
    int n_leapfrog,
    unsigned int seed
) {
    const int p = static_cast<int>(init.size());
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    std::vector<double> q = init;
    std::vector<double> grad(p, 0.0);
    double log_post = 0.0;
    gradient_numerical_generic(q, data, layout, grad, &log_post);

    std::vector<std::vector<double>> samples;
    samples.reserve(n_iter - n_warmup);

    int n_accept = 0;

    for (int iter = 0; iter < n_iter; iter++) {
        // Draw momentum
        std::vector<double> momentum(p);
        for (int j = 0; j < p; j++) {
            momentum[j] = normal(rng);
        }

        // Current state
        std::vector<double> q_prop = q;
        std::vector<double> m_prop = momentum;
        double current_H = -log_post;
        for (int j = 0; j < p; j++) {
            current_H += 0.5 * momentum[j] * momentum[j];
        }

        // Leapfrog integration
        std::vector<double> grad_prop(p, 0.0);
        double lp_prop = 0.0;
        gradient_numerical_generic(q_prop, data, layout, grad_prop, &lp_prop);

        // Half step for momentum
        for (int j = 0; j < p; j++) {
            m_prop[j] += 0.5 * step_size * grad_prop[j];
        }

        for (int l = 0; l < n_leapfrog; l++) {
            // Full step for position
            for (int j = 0; j < p; j++) {
                q_prop[j] += step_size * m_prop[j];
            }
            // Recompute gradient
            gradient_numerical_generic(q_prop, data, layout, grad_prop, &lp_prop);
            // Full step for momentum (except last)
            if (l < n_leapfrog - 1) {
                for (int j = 0; j < p; j++) {
                    m_prop[j] += step_size * grad_prop[j];
                }
            }
        }

        // Half step for momentum
        for (int j = 0; j < p; j++) {
            m_prop[j] += 0.5 * step_size * grad_prop[j];
        }

        // Compute proposed Hamiltonian
        double proposed_H = -lp_prop;
        for (int j = 0; j < p; j++) {
            proposed_H += 0.5 * m_prop[j] * m_prop[j];
        }

        // Metropolis accept/reject
        double log_alpha = current_H - proposed_H;
        if (std::log(uniform(rng)) < log_alpha) {
            q = q_prop;
            log_post = lp_prop;
            grad = grad_prop;
            n_accept++;
        }

        // Adaptive step size during warmup (very simple)
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            double accept_rate = static_cast<double>(n_accept) / (iter + 1);
            if (accept_rate < 0.5) step_size *= 0.8;
            else if (accept_rate > 0.8) step_size *= 1.2;
        }

        // Store post-warmup samples
        if (iter >= n_warmup) {
            samples.push_back(q);
        }
    }

    return samples;
}

// ============================================================================
// Rcpp entry point: fit a Gaussian GLM via generic tulpa interface
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_gaussian(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    double sigma_beta = 10.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    double step_size = 0.05,
    int n_leapfrog = 10,
    int seed = 42
) {
    const int N = y_r.size();
    const int p = X_r.ncol();

    // Set up model-specific response data
    GaussianData gd;
    gd.y.assign(y_r.begin(), y_r.end());

    // Set up generic ModelData
    ModelData data;
    data.N = N;
    data.n_processes = 1;
    data.sigma_beta = sigma_beta;

    // Process 0: the single linear predictor
    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize(N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[i * p + j] = X_r(i, j);
        }
    }
    data.processes.push_back(proc);
    data.model_response_data = &gd;

    // Sharing spec (trivial for single process)
    data.sharing.init(1);

    // ZI fields (not used)
    data.zi_type = tulpa_zi::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    // Set up ParamLayout
    ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);

    // Extra parameter: log(sigma) — the residual standard deviation
    layout.extra_offset = p;
    layout.n_extra_params = 1;
    layout.total_params = p + 1;

    // No RE, spatial, temporal, etc.
    layout.has_re = false;
    layout.has_zi = false;
    layout.has_oi = false;

    // Initial values: zeros for beta, log(1) = 0 for log_sigma
    std::vector<double> init(layout.total_params, 0.0);

    // Run HMC
    auto samples = run_hmc_simple(
        init, data, layout,
        n_iter, n_warmup,
        step_size, n_leapfrog,
        static_cast<unsigned int>(seed)
    );

    int n_samples = static_cast<int>(samples.size());

    // Convert to R matrix [n_samples x n_params]
    Rcpp::NumericMatrix draws(n_samples, layout.total_params);
    for (int s = 0; s < n_samples; s++) {
        for (int j = 0; j < layout.total_params; j++) {
            draws(s, j) = samples[s][j];
        }
    }

    // Name columns
    Rcpp::CharacterVector col_names(layout.total_params);
    for (int j = 0; j < p; j++) {
        col_names[j] = "beta[" + std::to_string(j + 1) + "]";
    }
    col_names[p] = "log_sigma";
    Rcpp::colnames(draws) = col_names;

    // Compute posterior means
    Rcpp::NumericVector means(layout.total_params, 0.0);
    for (int s = 0; s < n_samples; s++) {
        for (int j = 0; j < layout.total_params; j++) {
            means[j] += draws(s, j) / n_samples;
        }
    }
    means.names() = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("means") = means,
        Rcpp::Named("n_samples") = n_samples,
        Rcpp::Named("n_params") = layout.total_params,
        Rcpp::Named("accept_rate") = -1.0  // Not tracked in simple version
    );
}

// ============================================================================
// Generic NUTS sampler for multi-process models
// Uses tulpa's full NUTS backend with dual averaging and mass matrix adaptation.
// Model packages provide a LikelihoodFn<double> via LikelihoodSpec.
// ============================================================================

// Forward declare from hmc_sampler.cpp
// Default for inv_metric_init lives in hmc_sampler_funcs.h.
namespace tulpa_hmc {
    HMCResultCpp run_hmc_chain_cpp(
        const std::vector<double>& q_init,
        const ModelData& data,
        const ParamLayout& layout,
        int n_iter, int n_warmup, int L, int chain_id,
        unsigned int seed, bool verbose, int max_treedepth,
        MassMatrixType metric_type, double adapt_delta, int riemannian,
        const std::vector<double>& inv_metric_init);
}

// ============================================================================
// Shared Gaussian fixture: build the (GaussianData, LikelihoodSpec, ModelData,
// ParamLayout) bundle from y/X. The caller owns gd/spec/data/layout as locals
// so the pointers ModelData holds into gd/spec stay valid. Single source of
// truth for the single- and multi-chain generic entry points.
// ============================================================================
static void build_gaussian_model(
    const Rcpp::NumericVector& y_r,
    const Rcpp::NumericMatrix& X_r,
    double sigma_beta,
    GaussianData& gd,
    tulpa::LikelihoodSpec& spec,
    ModelData& data,
    ParamLayout& layout
) {
    const int N = y_r.size();
    const int p = X_r.ncol();

    gd.y.assign(y_r.begin(), y_r.end());

    spec.name = "gaussian";
    spec.n_processes = 1;
    spec.ll_double = gaussian_likelihood<double>;
    spec.ll_arena = gaussian_likelihood<tulpa::arena::Var>;
    spec.ll_fwd = gaussian_likelihood<::fwd::Dual>;
    spec.n_extra_params = 1;  // log_sigma

    data.N = N;
    data.n_processes = 1;
    data.sigma_beta = sigma_beta;

    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize(N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[i * p + j] = X_r(i, j);
        }
    }
    data.processes.push_back(proc);
    data.model_response_data = &gd;
    data.likelihood_spec = &spec;
    data.sharing.init(1);

    data.zi_type = tulpa::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    layout = tulpa_hmc::compute_param_layout(data);
}

// Column names for the Gaussian fixture: beta[1..p], log_sigma.
static Rcpp::CharacterVector gaussian_col_names(int p) {
    Rcpp::CharacterVector cn(p + 1);
    for (int j = 0; j < p; j++) {
        cn[j] = "beta[" + std::to_string(j + 1) + "]";
    }
    cn[p] = "log_sigma";
    return cn;
}

// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_generic(
    Rcpp::NumericVector y_r,           // Response vector
    Rcpp::NumericMatrix X_r,           // Design matrix (single process for now)
    double sigma_beta = 10.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = true,
    Rcpp::Nullable<Rcpp::NumericVector> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> inv_metric_init = R_NilValue
) {
    const int p = X_r.ncol();

    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, sigma_beta, gd, spec, data, layout);
    int n_params = layout.total_params;

    // Initial values — default origin, or caller-supplied for chain resume.
    std::vector<double> init_vec(n_params, 0.0);
    if (init.isNotNull()) {
        Rcpp::NumericVector iv(init);
        if ((int)iv.size() != n_params) {
            Rcpp::stop("init length %d != n_params %d", (int)iv.size(), n_params);
        }
        init_vec.assign(iv.begin(), iv.end());
    }

    // Optional warm-start inverse-mass diagonal (e.g. from a previous fit's
    // inv_metric output, gcol33/tulpa#29). Empty -> structural warm-start.
    std::vector<double> inv_metric_vec;
    if (inv_metric_init.isNotNull()) {
        Rcpp::NumericVector mv(inv_metric_init);
        if ((int)mv.size() != n_params) {
            Rcpp::stop("inv_metric_init length %d != n_params %d",
                       (int)mv.size(), n_params);
        }
        inv_metric_vec.assign(mv.begin(), mv.end());
    }

    // Run full NUTS (L=0 means NUTS)
    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init_vec, data, layout,
        n_iter, n_warmup,
        0,            // L=0 means NUTS
        1,            // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        0,            // riemannian=off
        inv_metric_vec
    );

    // Convert to R matrix
    int n_sample = result.n_sample;
    Rcpp::NumericMatrix draws(n_sample, n_params);
    for (int s = 0; s < n_sample; s++) {
        const double* row = result.sample_row(s);
        for (int j = 0; j < n_params; j++) {
            draws(s, j) = row[j];
        }
    }

    // Name columns
    Rcpp::CharacterVector col_names = gaussian_col_names(p);
    Rcpp::colnames(draws) = col_names;

    // Posterior means
    Rcpp::NumericVector means(n_params, 0.0);
    for (int s = 0; s < n_sample; s++) {
        for (int j = 0; j < n_params; j++) {
            means[j] += draws(s, j) / n_sample;
        }
    }
    means.names() = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("means") = means,
        Rcpp::Named("n_samples") = n_sample,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("log_prob") = Rcpp::wrap(result.log_prob),
        Rcpp::Named("accept_prob") = Rcpp::wrap(result.accept_prob),
        Rcpp::Named("divergent") = Rcpp::wrap(result.divergent),
        Rcpp::Named("treedepth") = Rcpp::wrap(result.treedepth),
        Rcpp::Named("sampler") = result.sampler.empty() ? "nuts" : result.sampler,
        Rcpp::Named("epsilon") = result.epsilon,
        // Warm-start / resume outputs (gcol33/tulpa#29)
        Rcpp::Named("inv_metric") = Rcpp::wrap(result.inv_metric_diag),
        Rcpp::Named("final_position") = Rcpp::wrap(result.final_position)
    );
}

// ============================================================================
// Multi-chain generic NUTS (gcol33/tulpa#30).
// Runs n_chains chains via tulpa's OpenMP across-chain core in one call and
// returns draws stacked chain-major with a chain_id vector — the layout
// tulpa::mcmc_diagnostics() consumes (#26) — plus per-chain epsilon, adapted
// inverse-mass diagonal, and final position (for resume).
//
// `init` / `inv_metric_init`, when supplied, are [n_chains x n_params]
// matrices (row c = chain c). `init` NULL -> origin for every chain;
// `inv_metric_init` NULL -> structural default for every chain. Passing the
// previous fit's `final_position` + `inv_metric` with n_warmup = 0 continues
// the chains.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_generic_chains(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    int n_chains = 4,
    double sigma_beta = 10.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> inv_metric_init = R_NilValue,
    std::string checkpoint_path = ""
) {
    if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");
    const int p = X_r.ncol();

    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, sigma_beta, gd, spec, data, layout);
    const int n_params = layout.total_params;

    // Per-chain initial position (rows of `init`, else origin).
    std::vector<std::vector<double>> q_init_per_chain(
        n_chains, std::vector<double>(n_params, 0.0));
    if (init.isNotNull()) {
        Rcpp::NumericMatrix im(init);
        if (im.nrow() != n_chains || im.ncol() != n_params) {
            Rcpp::stop("init must be [n_chains x n_params] = [%d x %d]",
                       n_chains, n_params);
        }
        for (int c = 0; c < n_chains; c++) {
            for (int j = 0; j < n_params; j++) q_init_per_chain[c][j] = im(c, j);
        }
    }

    // Per-chain inverse-mass diagonal (rows of `inv_metric_init`, else default).
    std::vector<std::vector<double>> inv_metric_per_chain;
    if (inv_metric_init.isNotNull()) {
        Rcpp::NumericMatrix mm(inv_metric_init);
        if (mm.nrow() != n_chains || mm.ncol() != n_params) {
            Rcpp::stop("inv_metric_init must be [n_chains x n_params] = [%d x %d]",
                       n_chains, n_params);
        }
        inv_metric_per_chain.resize(n_chains);
        for (int c = 0; c < n_chains; c++) {
            inv_metric_per_chain[c].resize(n_params);
            for (int j = 0; j < n_params; j++) inv_metric_per_chain[c][j] = mm(c, j);
        }
    }

    std::vector<tulpa_hmc::HMCResultCpp> chains = tulpa_hmc::run_hmc_parallel_chains_cpp(
        q_init_per_chain, inv_metric_per_chain, data,
        n_iter, n_warmup,
        0,                // L=0 means NUTS
        n_chains, static_cast<unsigned int>(seed), verbose,
        max_treedepth, tulpa::MassMatrixType::DIAG, adapt_delta,
        0,                // riemannian=off
        checkpoint_path
    );

    const int n_sample = chains[0].n_sample;
    const int n_total = n_sample * n_chains;

    // Draws stacked chain-major: chain 1's iterations, then chain 2's, ...
    Rcpp::NumericMatrix draws(n_total, n_params);
    Rcpp::IntegerVector chain_id(n_total);
    Rcpp::NumericVector log_prob(n_total);
    Rcpp::NumericVector accept_prob(n_total);
    Rcpp::IntegerVector divergent(n_total);
    Rcpp::IntegerVector treedepth(n_total);
    Rcpp::NumericVector epsilon(n_chains);
    Rcpp::NumericMatrix inv_metric(n_chains, n_params);
    Rcpp::NumericMatrix final_position(n_chains, n_params);

    int r = 0;
    for (int c = 0; c < n_chains; c++) {
        const tulpa_hmc::HMCResultCpp& ch = chains[c];
        for (int s = 0; s < ch.n_sample; s++) {
            const double* row = ch.sample_row(s);
            for (int j = 0; j < n_params; j++) draws(r, j) = row[j];
            chain_id[r] = c + 1;
            log_prob[r] = ch.log_prob[s];
            accept_prob[r] = ch.accept_prob[s];
            divergent[r] = ch.divergent[s];
            treedepth[r] = ch.treedepth[s];
            r++;
        }
        epsilon[c] = ch.epsilon;
        for (int j = 0; j < n_params; j++) {
            inv_metric(c, j) = (j < (int)ch.inv_metric_diag.size())
                                   ? ch.inv_metric_diag[j] : 1.0;
            final_position(c, j) = (j < (int)ch.final_position.size())
                                       ? ch.final_position[j] : 0.0;
        }
    }

    Rcpp::CharacterVector col_names = gaussian_col_names(p);
    Rcpp::colnames(draws) = col_names;
    Rcpp::colnames(inv_metric) = col_names;
    Rcpp::colnames(final_position) = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("chain_id") = chain_id,
        Rcpp::Named("n_chains") = n_chains,
        Rcpp::Named("n_samples") = n_sample,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("log_prob") = log_prob,
        Rcpp::Named("accept_prob") = accept_prob,
        Rcpp::Named("divergent") = divergent,
        Rcpp::Named("treedepth") = treedepth,
        Rcpp::Named("sampler") = chains[0].sampler.empty() ? "nuts" : chains[0].sampler,
        Rcpp::Named("epsilon") = epsilon,
        // Per-chain warm-start / resume outputs (gcol33/tulpa#29 + #30)
        Rcpp::Named("inv_metric") = inv_metric,
        Rcpp::Named("final_position") = final_position
    );
}

// ============================================================================
// C-ABI round-trip verifier for the resume outputs (gcol33/tulpa#29).
//
// The Rcpp-wrapper paths (cpp_tulpa_fit_generic / *_chains) read inv_metric
// and final_position directly off HMCResultCpp. Downstream packages instead
// reach the engine through the registered C ABI: tulpa::get_nuts_fn()
// -> R_GetCCallable("tulpa", "tulpa_run_nuts_generic") -> NUTSResult.
// fill_nuts_result_from_cpp() copies the resume fields into the C-ABI struct;
// this helper exercises that exact path so a regression to that copy (or to
// the ABI version / registration) fails a unit test rather than only showing
// up in a downstream consumer.
//
// Returns the two C-ABI resume fields (`inv_metric_out`, `final_position`)
// and the matrix of draws, so R-level tests can assert: (a) the C ABI
// populates the resume fields with the same values the Rcpp wrapper exposes,
// and (b) a continued chain warm-started from them samples the same posterior.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_test_c_abi_resume_roundtrip(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    int n_iter = 600,
    int n_warmup = 300,
    int max_treedepth = 7,
    double adapt_delta = 0.85,
    int seed = 9,
    Rcpp::Nullable<Rcpp::NumericVector> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> inv_metric_init = R_NilValue
) {
    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, /*sigma_beta=*/10.0, gd, spec, data, layout);
    const int n_params = layout.total_params;

    std::vector<double> init_vec(n_params, 0.0);
    if (init.isNotNull()) {
        Rcpp::NumericVector iv(init);
        if ((int)iv.size() != n_params) {
            Rcpp::stop("init length %d != n_params %d", (int)iv.size(), n_params);
        }
        init_vec.assign(iv.begin(), iv.end());
    }

    std::vector<double> inv_metric_vec;
    const double* inv_metric_ptr = nullptr;
    if (inv_metric_init.isNotNull()) {
        Rcpp::NumericVector mv(inv_metric_init);
        if ((int)mv.size() != n_params) {
            Rcpp::stop("inv_metric_init length %d != n_params %d",
                       (int)mv.size(), n_params);
        }
        inv_metric_vec.assign(mv.begin(), mv.end());
        inv_metric_ptr = inv_metric_vec.data();
    }

    // Reach the engine the way downstream packages do: through R_GetCCallable.
    // get_nuts_fn() caches the lookup and runs the ABI-version check.
    tulpa::NUTSFn run_nuts = tulpa::get_nuts_fn();
    if (run_nuts == nullptr) {
        Rcpp::stop("tulpa_run_nuts_generic is not registered");
    }

    tulpa::NUTSResult result = {};
    run_nuts(
        &data, &layout, init_vec.data(), n_params,
        n_iter, n_warmup, max_treedepth, adapt_delta,
        static_cast<unsigned int>(seed),
        /*verbose=*/0,
        inv_metric_ptr,
        &result
    );

    const int ns = result.n_sample;
    Rcpp::NumericMatrix draws(ns, n_params);
    for (int s = 0; s < ns; s++) {
        for (int j = 0; j < n_params; j++) {
            draws(s, j) = result.samples[s * n_params + j];
        }
    }
    Rcpp::CharacterVector col_names = gaussian_col_names(X_r.ncol());
    Rcpp::colnames(draws) = col_names;

    // The C-ABI resume fields — the thing this helper exists to verify.
    Rcpp::NumericVector inv_metric_out(n_params), final_position(n_params);
    for (int j = 0; j < n_params; j++) {
        inv_metric_out[j] = result.inv_metric_out[j];
        final_position[j] = result.final_position[j];
    }
    inv_metric_out.names() = col_names;
    final_position.names() = col_names;

    double epsilon = result.epsilon;
    result.free_buffers();

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("n_samples") = ns,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("epsilon") = epsilon,
        Rcpp::Named("inv_metric_out") = inv_metric_out,
        Rcpp::Named("final_position") = final_position
    );
}

// ============================================================================
// C-ABI round-trip verifier for the multi-chain runner (gcol33/tulpa#30).
//
// Mirrors cpp_test_c_abi_resume_roundtrip but for tulpa_run_nuts_chains:
// reaches the engine through tulpa::get_nuts_chains_fn() (R_GetCCallable),
// packs the chain-major [n_chains * n_params] init / inv_metric_diag buffers,
// allocates a NUTSResult[n_chains] for the output, and converts the per-chain
// results into R-side matrices.
//
// Returns draws stacked chain-major + a chain_id vector + n_chains — the
// exact (draws, chain_id, n_chains) layout tulpa::mcmc_diagnostics() reads
// — plus per-chain `inv_metric_out` / `final_position` for the resume
// contract verification.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_test_c_abi_chains_roundtrip(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    int n_chains = 3,
    int n_iter = 600,
    int n_warmup = 300,
    int max_treedepth = 7,
    double adapt_delta = 0.85,
    int seed = 30,
    Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> inv_metric_init = R_NilValue
) {
    if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");

    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, /*sigma_beta=*/10.0, gd, spec, data, layout);
    const int n_params = layout.total_params;
    const int p = X_r.ncol();

    // Per-chain init (chain-major rows), default origin.
    std::vector<double> init_flat((std::size_t)n_chains * n_params, 0.0);
    if (init.isNotNull()) {
        Rcpp::NumericMatrix im(init);
        if (im.nrow() != n_chains || im.ncol() != n_params) {
            Rcpp::stop("init must be [n_chains x n_params] = [%d x %d]",
                       n_chains, n_params);
        }
        for (int c = 0; c < n_chains; c++) {
            for (int j = 0; j < n_params; j++) {
                init_flat[(std::size_t)c * n_params + j] = im(c, j);
            }
        }
    }

    // Per-chain inverse-mass diagonal (chain-major rows), nullptr -> default.
    std::vector<double> inv_metric_flat;
    const double* inv_metric_ptr = nullptr;
    if (inv_metric_init.isNotNull()) {
        Rcpp::NumericMatrix mm(inv_metric_init);
        if (mm.nrow() != n_chains || mm.ncol() != n_params) {
            Rcpp::stop("inv_metric_init must be [n_chains x n_params] = [%d x %d]",
                       n_chains, n_params);
        }
        inv_metric_flat.assign((std::size_t)n_chains * n_params, 0.0);
        for (int c = 0; c < n_chains; c++) {
            for (int j = 0; j < n_params; j++) {
                inv_metric_flat[(std::size_t)c * n_params + j] = mm(c, j);
            }
        }
        inv_metric_ptr = inv_metric_flat.data();
    }

    // Reach the engine through R_GetCCallable. get_nuts_chains_fn() caches
    // the lookup and runs the ABI-version check on first use.
    tulpa::NUTSChainsFn run_chains = tulpa::get_nuts_chains_fn();
    if (run_chains == nullptr) {
        Rcpp::stop("tulpa_run_nuts_chains is not registered");
    }

    std::vector<tulpa::NUTSResult> results(n_chains, tulpa::NUTSResult{});
    run_chains(
        &data, &layout, init_flat.data(), n_params, n_chains,
        n_iter, n_warmup, max_treedepth, adapt_delta,
        static_cast<unsigned int>(seed),
        /*verbose=*/0,
        inv_metric_ptr,
        results.data()
    );

    const int n_sample = results[0].n_sample;
    const int n_total = n_sample * n_chains;

    Rcpp::NumericMatrix draws(n_total, n_params);
    Rcpp::IntegerVector chain_id(n_total);
    Rcpp::NumericVector epsilon(n_chains);
    Rcpp::NumericMatrix inv_metric_out(n_chains, n_params);
    Rcpp::NumericMatrix final_position(n_chains, n_params);

    int r = 0;
    for (int c = 0; c < n_chains; c++) {
        const tulpa::NUTSResult& res = results[c];
        for (int s = 0; s < res.n_sample; s++) {
            for (int j = 0; j < n_params; j++) {
                draws(r, j) = res.samples[s * n_params + j];
            }
            chain_id[r] = c + 1;
            r++;
        }
        epsilon[c] = res.epsilon;
        for (int j = 0; j < n_params; j++) {
            inv_metric_out(c, j) = res.inv_metric_out[j];
            final_position(c, j) = res.final_position[j];
        }
    }

    // Free the C-ABI-owned per-chain buffers. Mirrors what a downstream
    // consumer would do after copying results into its own R objects.
    for (int c = 0; c < n_chains; c++) results[c].free_buffers();

    Rcpp::CharacterVector col_names = gaussian_col_names(p);
    Rcpp::colnames(draws) = col_names;
    Rcpp::colnames(inv_metric_out) = col_names;
    Rcpp::colnames(final_position) = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("chain_id") = chain_id,
        Rcpp::Named("n_chains") = n_chains,
        Rcpp::Named("n_samples") = n_sample,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("epsilon") = epsilon,
        Rcpp::Named("inv_metric_out") = inv_metric_out,
        Rcpp::Named("final_position") = final_position
    );
}
