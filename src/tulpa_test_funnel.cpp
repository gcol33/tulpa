// tulpa_test_funnel.cpp
// Test-only entry point: fit Neal's funnel through the production NUTS engine
// with the SoftAbs divergence-retry kernel toggled on or off.
//
// The post-warmup SoftAbs retry (hmc_nuts_chain_iter_nuts.h) is a
// state-dependent kernel mixture: when the primary NUTS trajectory diverges it
// re-runs a fresh trajectory under a frozen Hessian-based metric. Choosing the
// transition kernel conditional on the first kernel's divergence can in
// principle fail to leave the target invariant. Neal's funnel is the canonical
// divergence-generating target with a known marginal (v ~ N(0, gamma^2)), so a
// recovery / equivalence test on it is the arbiter of whether the retry
// preserves the posterior.
//
// The funnel is encoded as an extra-parameter-only model: one process with zero
// fixed-effect columns (so the engine adds no N(0, sigma_beta) prior) and
// (K + 1) extra parameters carrying the whole target through the likelihood.
// This drives the exact production run_hmc_chain_cpp path, retry code included,
// selected by the `riemannian` flag (1 = force retry on, 0 = off).

#include <Rcpp.h>
#include <string>
#include <vector>
#include <cmath>

#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// ----------------------------------------------------------------------------
// Neal's funnel:  v ~ N(0, gamma^2),  x_i | v ~ N(0, exp(v/2)^2),  i = 1..K.
// Model-specific response data: K and the v-prior precision 1/gamma^2.
// ----------------------------------------------------------------------------
struct FunnelData {
    int K = 0;
    double inv_gamma2 = 0.0;  // 1 / gamma^2 (v-prior precision)
};

// Per-"observation" funnel log-density (templated for the N/A/A_r AD modes).
// Observation i contributes log N(x_i | 0, exp(v/2)); the single v-prior term
// is folded onto observation 0 so the assembled sum is the exact funnel
// log-density. Additive constants are dropped (irrelevant to MCMC).
template<typename T>
static T funnel_likelihood(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& logit_oi,
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data
) {
    (void)eta; (void)logit_zi; (void)logit_oi; (void)data;
    const auto* fd = static_cast<const FunnelData*>(model_data);
    const int off = layout.extra_offset;
    const T v = params[off];
    const T xi = params[off + 1 + i];

    using std::exp;  // ADL picks arena::exp / fwd::exp for the AD types
    const T half = T(0.5);
    // log N(x_i | 0, exp(v/2)) = -v/2 - 0.5 * x_i^2 * exp(-v)
    T contrib = (T(0.0) - half * v) - half * xi * xi * exp(T(0.0) - v);
    if (i == 0) {
        // log N(v | 0, gamma) = -0.5 * v^2 / gamma^2
        contrib = contrib - half * v * v * T(fd->inv_gamma2);
    }
    return contrib;
}

// Assemble the (FunnelData, LikelihoodSpec, ModelData, ParamLayout) bundle.
// Caller owns fd/spec so the pointers ModelData holds stay valid.
static void build_funnel_model(
    int K,
    double gamma,
    FunnelData& fd,
    tulpa::LikelihoodSpec& spec,
    ModelData& data,
    ParamLayout& layout
) {
    fd.K = K;
    fd.inv_gamma2 = 1.0 / (gamma * gamma);

    spec.name = "funnel";
    spec.n_processes = 1;
    spec.ll_double = funnel_likelihood<double>;
    spec.ll_arena = funnel_likelihood<tulpa::arena::Var>;
    spec.ll_fwd = funnel_likelihood<::fwd::Dual>;
    spec.n_extra_params = K + 1;  // v, x_1..x_K

    data.N = K;
    data.n_processes = 1;
    data.sigma_beta = 10.0;  // unused: no fixed-effect columns

    tulpa::ProcessData proc;
    proc.p = 0;  // zero fixed effects -> no N(0, sigma_beta) prior on the target
    data.processes.push_back(proc);
    data.model_response_data = &fd;
    data.likelihood_spec = &spec;
    data.sharing.init(1);

    data.zi_type = tulpa::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    layout = tulpa_hmc::compute_param_layout(data);
}

// Forward declaration (definition in hmc_sampler.cpp; default args in
// hmc_sampler_funcs.h). Same declaration the generic sampler uses.
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
Rcpp::List cpp_test_funnel_nuts(
    int K = 9,
    double gamma = 3.0,
    int n_iter = 3000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 1,
    int riemannian = 0,   // 1 = force SoftAbs divergence retry on, 0 = off
    bool verbose = false
) {
    if (K < 1) Rcpp::stop("K must be >= 1");
    if (gamma <= 0.0) Rcpp::stop("gamma must be > 0");

    FunnelData fd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_funnel_model(K, gamma, fd, spec, data, layout);
    const int n_params = layout.total_params;  // K + 1

    std::vector<double> init(n_params, 0.0);
    std::vector<double> inv_metric_vec;  // empty -> structural warm-start

    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init, data, layout,
        n_iter, n_warmup,
        0,            // L = 0 -> NUTS
        1,            // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        riemannian,
        inv_metric_vec
    );

    const int n_sample = result.n_sample;
    Rcpp::NumericMatrix draws(n_sample, n_params);
    Rcpp::IntegerVector divergent(n_sample);
    int n_div = 0;
    for (int s = 0; s < n_sample; s++) {
        const double* row = result.sample_row(s);
        for (int j = 0; j < n_params; j++) draws(s, j) = row[j];
        divergent[s] = result.divergent[s];
        n_div += result.divergent[s];
    }

    Rcpp::CharacterVector col_names(n_params);
    col_names[0] = "v";
    for (int j = 1; j <= K; j++) col_names[j] = "x[" + std::to_string(j) + "]";
    Rcpp::colnames(draws) = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("divergent") = divergent,
        Rcpp::Named("n_divergent") = n_div,
        Rcpp::Named("n_samples") = n_sample,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("riemannian") = riemannian
    );
}
