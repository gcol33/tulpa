// tulpa_generic_sampler.cpp
// Proof-of-concept: generic model fitting via LikelihoodFn callback
// Demonstrates the tulpa multi-process interface end-to-end

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "linalg_fast.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"

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
    bool verbose = true
) {
    const int N = y_r.size();
    const int p = X_r.ncol();

    // Gaussian model data
    GaussianData gd;
    gd.y.assign(y_r.begin(), y_r.end());

    // LikelihoodSpec (with arena AD for automatic gradients)
    tulpa::LikelihoodSpec spec;
    spec.name = "gaussian";
    spec.n_processes = 1;
    spec.ll_double = gaussian_likelihood<double>;
    spec.ll_arena = gaussian_likelihood<tulpa::arena::Var>;
    spec.ll_fwd = gaussian_likelihood<::fwd::Dual>;
    spec.n_extra_params = 1;  // log_sigma

    // ModelData
    ModelData data;
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

    // Unused fields
    data.zi_type = tulpa::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    // ParamLayout (computed via the generic branch in compute_param_layout)
    ParamLayout layout = tulpa_hmc::compute_param_layout(data);
    int n_params = layout.total_params;

    // Initial values
    std::vector<double> init(n_params, 0.0);

    // Run full NUTS (L=0 means NUTS)
    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init, data, layout,
        n_iter, n_warmup,
        0,            // L=0 means NUTS
        1,            // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        0             // riemannian=off
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
    Rcpp::CharacterVector col_names(n_params);
    for (int j = 0; j < p; j++) {
        col_names[j] = "beta[" + std::to_string(j + 1) + "]";
    }
    col_names[p] = "log_sigma";
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
        Rcpp::Named("epsilon") = result.epsilon
    );
}
