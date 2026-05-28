// tulpa_beta_sampler.cpp
//
// Built-in Beta-regression NUTS sampler. Demonstrates that the
// LikelihoodSpec → compute_log_post_generic → run_hmc_chain_cpp path
// supports families beyond the Gaussian proof-of-concept in
// tulpa_generic_sampler.cpp.
//
// Phi (the Beta precision in the mean–precision parametrisation,
// y ~ Beta(mu*phi, (1-mu)*phi)) is sampled jointly with beta as a single
// extra parameter on the log scale: NUTS performs exact marginalisation
// over phi, replacing the Brent outer-opt in tulpa_laplace_beta() and
// the would-be nested-Laplace grid over phi with a single MCMC pass.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include "hmc_sampler.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// ============================================================================
// Beta response data (model-specific, opaque to tulpa engine)
//
// log_y_i and log_1my_i are precomputed in the constructor so the per-obs
// likelihood loop avoids std::log on every NUTS iteration.
// ============================================================================
struct BetaData {
    std::vector<double> y;          // raw response in (0, 1)
    std::vector<double> log_y;      // log(y_i)
    std::vector<double> log_1my;    // log(1 - y_i)
    double log_phi_prior_sd;        // SD of N(0, sd) prior on log_phi
};

// ============================================================================
// Beta log-likelihood (templated for autodiff compatibility)
//
// Density (Ferrari & Cribari-Neto 2004), with mu = logit^{-1}(eta) and phi
// extracted from params at layout.extra_offset on the log scale:
//
//   log f = lgamma(phi) - lgamma(a) - lgamma(b)
//           + (a - 1) log(y) + (b - 1) log(1 - y),
//   a = mu * phi,  b = (1 - mu) * phi.
//
// The lgamma overloads on arena::Var / fwd::Dual carry the digamma chain
// rule, so reverse / forward AD recover the score in beta and log_phi
// without bespoke handcoding.
// ============================================================================
template<typename T>
T beta_likelihood(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& params,
    const ModelData& /*data*/,
    const ParamLayout& layout,
    const void* model_data
) {
    using std::exp;
    using std::lgamma;

    const auto* bd = static_cast<const BetaData*>(model_data);

    // mu = sigmoid(eta) — clamp via the standard -log1p(exp(-x)) trick to
    // stay numerically sound at large |eta|. arena::Var / fwd::Dual both
    // expose exp via ADL; we let the linear solve in NUTS guard against
    // pathological eta drifts rather than clamping mu directly.
    T eta_i = eta[0];
    T mu    = T(1.0) / (T(1.0) + exp(T(0.0) - eta_i));

    T log_phi = params[layout.extra_offset];
    T phi     = exp(log_phi);
    T a       = mu * phi;
    T b       = (T(1.0) - mu) * phi;

    return lgamma(phi) - lgamma(a) - lgamma(b)
         + (a - T(1.0)) * T(bd->log_y[i])
         + (b - T(1.0)) * T(bd->log_1my[i]);
}

// ============================================================================
// log_phi prior: Normal(0, sd_log_phi). Returned up to additive constants
// (the -log(sd) - 0.5*log(2*pi) term cancels in NUTS).
// ============================================================================
static double beta_extra_prior_double(
    const std::vector<double>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    const auto* bd = static_cast<const BetaData*>(model_data);
    double log_phi = params[layout.extra_offset];
    double sd      = bd->log_phi_prior_sd;
    return -0.5 * (log_phi / sd) * (log_phi / sd);
}

static tulpa::arena::Var beta_extra_prior_arena(
    const std::vector<tulpa::arena::Var>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    using tulpa::arena::Var;
    const auto* bd = static_cast<const BetaData*>(model_data);
    Var log_phi = params[layout.extra_offset];
    double sd   = bd->log_phi_prior_sd;
    Var z       = log_phi * Var(1.0 / sd);
    return Var(-0.5) * z * z;
}

// Forward declare from hmc_sampler.cpp (mirrors tulpa_generic_sampler.cpp).
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
// Rcpp entry: Fit a Beta GLM via tulpa NUTS, sampling beta + log_phi jointly.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_beta_nuts(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    double sigma_beta = 10.0,
    double log_phi_prior_sd = 3.0,
    double log_phi_init = 0.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = false
) {
    const int N = y_r.size();
    const int p = X_r.ncol();

    if (N == 0 || p == 0) {
        Rcpp::stop("y and X must be non-empty");
    }
    for (int i = 0; i < N; i++) {
        if (!R_finite(y_r[i]) || y_r[i] <= 0.0 || y_r[i] >= 1.0) {
            Rcpp::stop("y must be strictly in (0, 1) for the beta likelihood");
        }
    }

    // BetaData with precomputed log(y), log(1-y).
    BetaData bd;
    bd.y.assign(y_r.begin(), y_r.end());
    bd.log_y.resize(N);
    bd.log_1my.resize(N);
    for (int i = 0; i < N; i++) {
        bd.log_y[i]   = std::log(bd.y[i]);
        bd.log_1my[i] = std::log(1.0 - bd.y[i]);
    }
    bd.log_phi_prior_sd = log_phi_prior_sd;

    // LikelihoodSpec (arena AD path picks up the gradient automatically).
    tulpa::LikelihoodSpec spec;
    spec.name              = "beta";
    spec.n_processes       = 1;
    spec.ll_double         = beta_likelihood<double>;
    spec.ll_arena          = beta_likelihood<tulpa::arena::Var>;
    spec.ll_fwd            = beta_likelihood<::fwd::Dual>;
    spec.n_extra_params    = 1;  // log_phi
    spec.extra_prior       = beta_extra_prior_double;
    spec.extra_prior_arena = beta_extra_prior_arena;

    // ModelData
    ModelData data;
    data.N           = N;
    data.n_processes = 1;
    data.sigma_beta  = sigma_beta;

    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize(static_cast<size_t>(N) * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[i * p + j] = X_r(i, j);
        }
    }
    data.processes.push_back(proc);
    data.model_response_data = &bd;
    data.likelihood_spec     = &spec;
    data.sharing.init(1);

    // Unused channels
    data.zi_type     = tulpa::ZIType::NONE;
    data.p_zi        = 0;
    data.p_oi        = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    // ParamLayout via the generic branch (places log_phi at extra_offset).
    ParamLayout layout = tulpa_hmc::compute_param_layout(data);
    int n_params = layout.total_params;

    // Initial values: zeros for beta, user-supplied log_phi (default 0 -> phi=1).
    std::vector<double> init(n_params, 0.0);
    init[layout.extra_offset] = log_phi_init;

    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init, data, layout,
        n_iter, n_warmup,
        0,          // L=0 -> NUTS
        1,          // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        0,          // riemannian off
        std::vector<double>{}  // inv_metric_init: default
    );

    int n_sample = result.n_sample;
    Rcpp::NumericMatrix draws(n_sample, n_params);
    for (int s = 0; s < n_sample; s++) {
        const double* row = result.sample_row(s);
        for (int j = 0; j < n_params; j++) {
            draws(s, j) = row[j];
        }
    }

    Rcpp::CharacterVector col_names(n_params);
    for (int j = 0; j < p; j++) {
        col_names[j] = "beta[" + std::to_string(j + 1) + "]";
    }
    col_names[layout.extra_offset] = "log_phi";
    Rcpp::colnames(draws) = col_names;

    Rcpp::NumericVector means(n_params, 0.0);
    for (int s = 0; s < n_sample; s++) {
        for (int j = 0; j < n_params; j++) {
            means[j] += draws(s, j) / n_sample;
        }
    }
    means.names() = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws")       = draws,
        Rcpp::Named("means")       = means,
        Rcpp::Named("n_samples")   = n_sample,
        Rcpp::Named("n_params")    = n_params,
        Rcpp::Named("log_prob")    = Rcpp::wrap(result.log_prob),
        Rcpp::Named("accept_prob") = Rcpp::wrap(result.accept_prob),
        Rcpp::Named("divergent")   = Rcpp::wrap(result.divergent),
        Rcpp::Named("treedepth")   = Rcpp::wrap(result.treedepth),
        Rcpp::Named("sampler")     = result.sampler.empty() ? "nuts" : result.sampler,
        Rcpp::Named("epsilon")     = result.epsilon
    );
}
