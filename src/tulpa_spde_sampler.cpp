// tulpa_spde_sampler.cpp
//
// SPDE-via-NUTS sampler. Conditions on a fixed Matern parameter pair
// (kappa, tau_spde) and samples the (beta, w_mesh, log_phi) latent block
// jointly via tulpa's full NUTS backend.
//
// Architecturally orthogonal to the existing SPDE Laplace path
// (cpp_laplace_fit_spde / cpp_nested_laplace_spde): instead of running
// PIRLS to a mode and Gaussian-approximating the posterior, this uses
// the LikelihoodSpec extension point — the same mechanism downstream
// model packages already use — to feed the SPDE prior + likelihood
// through compute_log_post_generic and the production NUTS chain.
//
// Layout (using the LikelihoodSpec extra-params slot):
//   params[0 .. p)                  beta (process 0)
//   params[extra_offset .. +n_mesh) w_mesh (mesh-node effects)
//   params[extra_offset + n_mesh)   log_phi (sampled jointly)
//
// Hyperparameter integration over (kappa, tau_spde) is left to an outer
// loop (the existing nested-Laplace grid in cpp_nested_laplace_spde, or
// a future NUTS-over-hypers variant). Within a single call, (kappa, tau)
// are fixed.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include "hmc_sampler.h"
#include "spde_qbuilder.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

// Family enum local to this TU — selects the per-obs log-likelihood and
// the role of `log_phi`. Kept C++-local because the user-facing R wrapper
// translates a string into this code.
enum class SpdeFamily : int {
    GAUSSIAN = 0,    // log_phi = log(sigma);    y ~ N(mu, sigma^2)
    POISSON  = 1,    // log_phi unused;          y ~ Poisson(exp(eta))
    BINOMIAL = 2     // log_phi unused;          y ~ Binomial(n_trials, sigmoid(eta))
};

// =====================================================================
// SPDEData: response + spatial structure (Q, A) at fixed hypers, plus
// precomputed log(n_trials choose y) for the binomial path.
// =====================================================================
struct SPDEData {
    SpdeFamily family;
    std::vector<double> y;
    std::vector<int>    n_trials;     // binomial only
    std::vector<double> log_y_fact;   // log(y!) for poisson constant
    std::vector<double> log_choose;   // lchoose(n, y) for binomial constant

    int p;
    int n_mesh;
    tulpa::ARows         a_rows;
    std::vector<int>     Q_p;
    std::vector<int>     Q_i;
    std::vector<double>  Q_x;

    // log_phi prior: only used by Gaussian (log_sigma). Wider default
    // than the Beta sampler because residual SD is on the y-scale; the
    // R wrapper picks a sensible default per-family.
    double log_phi_prior_sd;
};

// =====================================================================
// Per-observation log-likelihood (templated for AD).
//
// The LikelihoodSpec contract gives us eta[0] = X_i @ beta from the
// generic eta-precompute. We add the mesh contribution sum_j A_ij * w_j
// here; w lives at params[extra_offset .. extra_offset + n_mesh).
// =====================================================================
template<typename T>
T spde_likelihood(
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
    using std::log;
    using std::lgamma;

    const auto* sd = static_cast<const SPDEData*>(model_data);

    // eta_i = X_i @ beta + sum_j A_ij * w_j
    T eta_i = eta[0];
    for (const auto& ae : sd->a_rows[i]) {
        eta_i = eta_i + T(ae.weight) * params[layout.extra_offset + ae.mesh_idx];
    }

    switch (sd->family) {
        case SpdeFamily::GAUSSIAN: {
            // log_sigma = params[extra_offset + n_mesh]
            T log_sigma = params[layout.extra_offset + sd->n_mesh];
            T resid     = T(sd->y[i]) - eta_i;
            // -log(sigma) - 0.5 * (resid / sigma)^2; constant -0.5*log(2 pi) dropped.
            T inv_sigma = exp(T(0.0) - log_sigma);
            return T(0.0) - log_sigma - T(0.5) * resid * resid * inv_sigma * inv_sigma;
        }
        case SpdeFamily::POISSON: {
            // log p(y | lambda = exp(eta)) = y*eta - exp(eta) - log(y!)
            T lambda = exp(eta_i);
            return T(sd->y[i]) * eta_i - lambda - T(sd->log_y_fact[i]);
        }
        case SpdeFamily::BINOMIAL: {
            // log p(y | n, p = sigmoid(eta)) = lchoose(n, y) + y*eta - n*log(1+exp(eta))
            // Numerically stable softplus: log(1+exp(eta)) = max(eta,0) + log(1+exp(-|eta|))
            // We avoid the abs branch in AD-land by using the standard form below.
            T one_plus_exp = T(1.0) + exp(eta_i);
            return T(sd->log_choose[i]) + T(sd->y[i]) * eta_i
                 - T(static_cast<double>(sd->n_trials[i])) * log(one_plus_exp);
        }
    }
    return T(0.0);
}

// =====================================================================
// SPDE prior contribution: -0.5 * w^T Q w + Normal(0, sd) prior on log_phi
// (gaussian only; for poisson/binomial log_phi is unused but still in
// extra_params with a tight standard-normal pin).
// =====================================================================
static double spde_extra_prior_double(
    const std::vector<double>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    const auto* sd = static_cast<const SPDEData*>(model_data);
    int wo = layout.extra_offset;

    // -0.5 * w^T Q w via CSC iteration
    double qf = 0.0;
    for (int col = 0; col < sd->n_mesh; col++) {
        for (int idx = sd->Q_p[col]; idx < sd->Q_p[col + 1]; idx++) {
            int row = sd->Q_i[idx];
            qf += params[wo + row] * sd->Q_x[idx] * params[wo + col];
        }
    }
    double lp = -0.5 * qf;

    // log_phi prior (always present in the layout; for non-Gaussian it
    // is pinned tightly so it does not float as a free parameter).
    double log_phi = params[wo + sd->n_mesh];
    double sd_lp   = sd->log_phi_prior_sd;
    lp += -0.5 * (log_phi / sd_lp) * (log_phi / sd_lp);

    return lp;
}

static tulpa::arena::Var spde_extra_prior_arena(
    const std::vector<tulpa::arena::Var>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    using tulpa::arena::Var;
    const auto* sd = static_cast<const SPDEData*>(model_data);
    int wo = layout.extra_offset;

    Var lp = Var(0.0);
    for (int col = 0; col < sd->n_mesh; col++) {
        for (int idx = sd->Q_p[col]; idx < sd->Q_p[col + 1]; idx++) {
            int row = sd->Q_i[idx];
            lp = lp + params[wo + row] * Var(sd->Q_x[idx]) * params[wo + col];
        }
    }
    Var minus_half_qf = Var(-0.5) * lp;

    Var log_phi = params[wo + sd->n_mesh];
    double sd_lp = sd->log_phi_prior_sd;
    Var z = log_phi * Var(1.0 / sd_lp);
    return minus_half_qf + Var(-0.5) * z * z;
}

}  // anonymous namespace

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

// =====================================================================
// Rcpp entry point.
// =====================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_spde_nuts(
    Rcpp::NumericVector y_r,
    Rcpp::IntegerVector n_trials_r,
    Rcpp::NumericMatrix X_r,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    double kappa, double tau_spde,
    std::string family,
    int alpha = 2,
    double sigma_beta = 10.0,
    double log_phi_prior_sd = 3.0,
    double log_phi_init = 0.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = false,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable = R_NilValue
) {
    const int N = y_r.size();
    const int p = X_r.ncol();
    if (N != n_obs) Rcpp::stop("length(y) must equal n_obs");
    if (N == 0 || p == 0) Rcpp::stop("y and X must be non-empty");
    if (n_mesh <= 0) Rcpp::stop("n_mesh must be positive");

    SpdeFamily fam;
    if      (family == "gaussian") fam = SpdeFamily::GAUSSIAN;
    else if (family == "poisson")  fam = SpdeFamily::POISSON;
    else if (family == "binomial") fam = SpdeFamily::BINOMIAL;
    else Rcpp::stop("Unsupported family for tulpa_nuts_spde: '%s'. "
                    "Supported: gaussian, poisson, binomial.", family.c_str());

    // Family-specific input validation.
    if (fam == SpdeFamily::POISSON) {
        for (int i = 0; i < N; i++) {
            if (!R_finite(y_r[i]) || y_r[i] < 0.0 ||
                std::abs(y_r[i] - std::round(y_r[i])) > 1e-9) {
                Rcpp::stop("poisson family requires non-negative integer y");
            }
        }
    } else if (fam == SpdeFamily::BINOMIAL) {
        if (n_trials_r.size() != N) {
            Rcpp::stop("binomial family requires n_trials of length n_obs");
        }
        for (int i = 0; i < N; i++) {
            if (!R_finite(y_r[i]) || y_r[i] < 0.0 || y_r[i] > n_trials_r[i] ||
                std::abs(y_r[i] - std::round(y_r[i])) > 1e-9) {
                Rcpp::stop("binomial family requires y in [0, n_trials]");
            }
        }
    }

    // Build SPDE Q at the supplied (kappa, tau_spde).
    tulpa::SpdeQBuilder qb;
    qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);
    if (rational_poles_nullable.isNotNull() && rational_weights_nullable.isNotNull()) {
        std::vector<double> poles   = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        std::vector<double> weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
        qb.rebuild_rational(kappa, tau_spde, poles, weights);
    } else {
        qb.rebuild(kappa, tau_spde, alpha);
    }

    // Per-row A storage.
    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    // Marshal SPDEData.
    SPDEData sd;
    sd.family = fam;
    sd.y.assign(y_r.begin(), y_r.end());
    if (fam == SpdeFamily::BINOMIAL) {
        sd.n_trials.assign(n_trials_r.begin(), n_trials_r.end());
        sd.log_choose.resize(N);
        for (int i = 0; i < N; i++) {
            sd.log_choose[i] = R::lchoose(static_cast<double>(sd.n_trials[i]), sd.y[i]);
        }
    }
    if (fam == SpdeFamily::POISSON) {
        sd.log_y_fact.resize(N);
        for (int i = 0; i < N; i++) {
            sd.log_y_fact[i] = R::lgammafn(sd.y[i] + 1.0);
        }
    }
    sd.p      = p;
    sd.n_mesh = n_mesh;
    sd.a_rows = a_rows;
    sd.Q_p.assign(qb.Q_p.begin(), qb.Q_p.end());
    sd.Q_i.assign(qb.Q_i.begin(), qb.Q_i.end());
    sd.Q_x.assign(qb.Q_x.begin(), qb.Q_x.end());
    sd.log_phi_prior_sd = log_phi_prior_sd;

    // LikelihoodSpec wiring extra params = w_mesh (n_mesh) + log_phi (1).
    tulpa::LikelihoodSpec spec;
    spec.name              = "spde_" + family;
    spec.n_processes       = 1;
    spec.ll_double         = spde_likelihood<double>;
    spec.ll_arena          = spde_likelihood<tulpa::arena::Var>;
    spec.ll_fwd            = spde_likelihood<::fwd::Dual>;
    spec.n_extra_params    = n_mesh + 1;
    spec.extra_prior       = spde_extra_prior_double;
    spec.extra_prior_arena = spde_extra_prior_arena;

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
    data.model_response_data = &sd;
    data.likelihood_spec     = &spec;
    data.sharing.init(1);

    data.zi_type     = tulpa::ZIType::NONE;
    data.p_zi        = 0;
    data.p_oi        = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    ParamLayout layout = tulpa_hmc::compute_param_layout(data);
    int n_params = layout.total_params;

    std::vector<double> init(n_params, 0.0);
    init[layout.extra_offset + n_mesh] = log_phi_init;

    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init, data, layout,
        n_iter, n_warmup,
        0,                                  // L=0 -> NUTS
        1,                                  // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        0,                                  // riemannian off
        std::vector<double>{}
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
    for (int m = 0; m < n_mesh; m++) {
        col_names[layout.extra_offset + m] = "w[" + std::to_string(m + 1) + "]";
    }
    col_names[layout.extra_offset + n_mesh] = "log_phi";
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
        Rcpp::Named("p")           = p,
        Rcpp::Named("n_mesh")      = n_mesh,
        Rcpp::Named("kappa")       = kappa,
        Rcpp::Named("tau_spde")    = tau_spde,
        Rcpp::Named("family")      = family,
        Rcpp::Named("log_prob")    = Rcpp::wrap(result.log_prob),
        Rcpp::Named("accept_prob") = Rcpp::wrap(result.accept_prob),
        Rcpp::Named("divergent")   = Rcpp::wrap(result.divergent),
        Rcpp::Named("treedepth")   = Rcpp::wrap(result.treedepth),
        Rcpp::Named("sampler")     = result.sampler.empty() ? "nuts" : result.sampler,
        Rcpp::Named("epsilon")     = result.epsilon
    );
}
