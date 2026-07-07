// tulpa_sample_glmm.cpp
// ----------------------------------------------------------------------------
// Generic GLMM front-door entry for the model-agnostic sampler kernels that
// drive a tulpa ModelData (gcol33/tulpa#54, #55): NUTS, elliptical slice
// sampling (ESS), SGHMC, SGLD, MCLMC, SMC, and VI. Every one consumes a tulpa
// ModelData + ParamLayout, so a single entry builds that model once and
// dispatches to the chosen kernel by name. No likelihood / link / layout logic
// is duplicated.
//
// Latent structure (gcol33/tulpa#75): random effects (intercept / slopes /
// correlated / multi-term), areal spatial fields (ICAR / BYM2), and temporal
// fields (RW1 / RW2 / AR1) are threaded through build_sampler_model_inputs(),
// which populates the ModelData and derives the layout via compute_param_layout.
// The kernels then sample the full parameter vector -- fixed effects, latent
// effects, AND the variance-component hyperparameters jointly (full Bayes over
// the variance components, not conditioning on them like the Laplace / logpost
// backends). The four default-link AD-covered families (gaussian / poisson /
// binomial / neg_binomial_2) get analytic gradients; any other family falls back
// to the numerical gradient, which still scores the full latent log-posterior.
// ----------------------------------------------------------------------------

#include <Rcpp.h>
#include <string>
#include <vector>
#include <cmath>

#include "laplace_spec_fit.h"      // as_offset_vec
#include "sampler_model_data.h"    // build_sampler_model_inputs / sampler_param_names
#include "hmc_sampler.h"           // tulpa_hmc::run_hmc_parallel_chains_cpp, HMCResultCpp
#include "ess_sampler.h"           // tulpa_ess::run_ess_sampler
#include "sghmc_sampler.h"         // tulpa_sghmc::run_sghmc_sampler / run_sgld_sampler
#include "mclmc_modeldata.h"       // tulpa_mclmc::run_mclmc_sampler
#include "smc_modeldata.h"         // tulpa::run_smc_sampler
#include "vi_types.h"              // tulpa::vi::VIConfig / VIResult / VIVariant

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// fit_vi is defined in vi_sampler.cpp without a public header declaration; the
// VI shim reaches it the same way. Forward-declare the exact signature.
namespace tulpa { namespace vi {
VIResult fit_vi(const ModelData& data, const ParamLayout& layout, int D,
                const VIConfig& config, const Eigen::VectorXd* init_mu);
} }

namespace {

// Build an [n_save x D] R matrix of draws from an Eigen sample matrix, taking
// the first D columns and naming them with `cn`.
Rcpp::NumericMatrix eigen_draws_to_r(const Eigen::MatrixXd& S, int D,
                                     const Rcpp::CharacterVector& cn) {
    const int n = (int)S.rows();
    Rcpp::NumericMatrix draws(n, D);
    for (int s = 0; s < n; s++)
        for (int j = 0; j < D; j++) draws(s, j) = S(s, j);
    if (n > 0) Rcpp::colnames(draws) = cn;
    return draws;
}

Rcpp::NumericVector col_means(const Rcpp::NumericMatrix& draws) {
    const int n = draws.nrow(), p = draws.ncol();
    Rcpp::NumericVector m(p, 0.0);
    for (int s = 0; s < n; s++)
        for (int j = 0; j < p; j++) m[j] += draws(s, j) / (n > 0 ? n : 1);
    if (n > 0) m.names() = Rcpp::colnames(draws);
    return m;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cpp_tulpa_sample_glmm(
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    std::string backend,
    double phi = 1.0,
    double sigma_beta = 10.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    int seed = 42,
    bool verbose = false,
    int n_chains = 4,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    double epsilon = 0.0,
    int L = 10,
    int batch_size = 0,
    double alpha = 0.1,
    int mclmc_adjusted = 0,
    int n_particles = 1000,
    int n_mcmc_steps = 5,
    double ess_threshold = 0.5,
    int vi_variant = 3,
    int vi_mc_samples = 10,
    int vi_max_iter = 10000,
    int vi_n_draws = 2000,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    Rcpp::Nullable<Rcpp::CharacterVector> fixed_names = R_NilValue,
    double phi2 = NA_REAL
) {
    // Argument groups (kept out of the signature so Rcpp::compileAttributes does
    // not fold the comments into the generated wrapper):
    //   NUTS:  n_chains, max_treedepth, adapt_delta
    //   SGHMC/SGLD/MCLMC: epsilon (0 => kernel default/adaptation), L,
    //       batch_size (0 => full-data gradient), alpha (SGHMC friction),
    //       mclmc_adjusted
    //   SMC:   n_particles, n_mcmc_steps, ess_threshold
    //   VI:    vi_variant (0=meanfield,1=lowrank,2=fullrank,3=auto),
    //       vi_mc_samples, vi_max_iter, vi_n_draws
    //   Structure: re_spec / spatial_spec / temporal_spec (null => absent),
    //       sigma_re_scale (half-Cauchy scale on the RE / BYM2 SDs).
    const int N = y.size();

    // Build the full ModelData + ParamLayout (fixed effects + any RE / spatial /
    // temporal structure) once; the kernels sample the full vector it lays out.
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, sigma_beta, offset, sigma_re_scale,
        re_spec, spatial_spec, temporal_spec);
    in.resp.phi2 = phi2;   // NA_REAL is a NaN => family default (e.g. t df = 4)
    const int D = in.layout.total_params;
    std::vector<double> init(D, 0.0);
    Rcpp::CharacterVector cn = tulpa::sampler_param_names(in.data, in.layout, fixed_names);

    Rcpp::List out;

    if (backend == "nuts" || backend == "hmc") {
        if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");
        std::vector<std::vector<double>> q_init(n_chains, std::vector<double>(D, 0.0));
        std::vector<std::vector<double>> inv_metric;   // empty -> structural default
        std::vector<tulpa_hmc::HMCResultCpp> chains = tulpa_hmc::run_hmc_parallel_chains_cpp(
            q_init, inv_metric, in.data, n_iter, n_warmup, /*L=*/0, n_chains,
            (unsigned int)seed, verbose, max_treedepth,
            tulpa::MassMatrixType::DIAG, adapt_delta, /*riemannian=*/0, "");
        const int n_sample = chains[0].n_sample;
        const int n_total = n_sample * n_chains;
        Rcpp::NumericMatrix draws(n_total, D);
        Rcpp::IntegerVector chain_id(n_total);
        Rcpp::NumericVector log_prob(n_total), accept_prob(n_total);
        Rcpp::IntegerVector divergent(n_total), treedepth(n_total);
        Rcpp::NumericVector epsilon_out(n_chains);
        int r = 0;
        for (int c = 0; c < n_chains; c++) {
            const tulpa_hmc::HMCResultCpp& ch = chains[c];
            for (int s = 0; s < ch.n_sample; s++) {
                const double* row = ch.sample_row(s);
                for (int j = 0; j < D; j++) draws(r, j) = row[j];
                chain_id[r] = c + 1;
                log_prob[r] = ch.log_prob[s];
                accept_prob[r] = ch.accept_prob[s];
                divergent[r] = ch.divergent[s];
                treedepth[r] = ch.treedepth[s];
                r++;
            }
            epsilon_out[c] = ch.epsilon;
        }
        if (n_total > 0) Rcpp::colnames(draws) = cn;
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws,
            Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("chain_id") = chain_id,
            Rcpp::Named("n_chains") = n_chains,
            Rcpp::Named("n_samples") = n_sample,
            Rcpp::Named("n_params") = D,
            Rcpp::Named("log_prob") = log_prob,
            Rcpp::Named("accept_prob") = accept_prob,
            Rcpp::Named("divergent") = divergent,
            Rcpp::Named("treedepth") = treedepth,
            Rcpp::Named("epsilon") = epsilon_out,
            Rcpp::Named("sampler") = "nuts");
        return out;
    }

    if (backend == "ess") {
        tulpa_ess::ESSConfig cfg;
        cfg.n_iter = n_iter; cfg.n_warmup = n_warmup; cfg.n_thin = 1;
        cfg.verbose = verbose; cfg.print_every = 500; cfg.seed = (unsigned int)seed;
        cfg.use_cholesky = true; cfg.adapt_during_warmup = false;
        cfg.adapt_interval = 50;
        // Joint (log_sigma_re, re) rescaling move: needed when an RE term is
        // present so log_sigma_re and the RE block mix (they are strongly
        // anti-correlated under the prior).
        cfg.joint_sigma_re = in.layout.has_re;
        cfg.joint_sigma_proposal_sd = 0.1;
        tulpa_ess::ESSResult res = tulpa_ess::run_ess_sampler(init, in.data, in.layout, cfg);
        if (!res.success) Rcpp::stop("ess sampler failed: %s", res.error_msg);
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, D, cn);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws,
            Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(),
            Rcpp::Named("n_params") = D,
            Rcpp::Named("avg_slice_evals") = res.avg_slice_evals,
            Rcpp::Named("sampler") = "ess");
        return out;
    }

    if (backend == "sghmc") {
        tulpa_sghmc::SGHMCConfig cfg;
        cfg.n_iter = n_iter; cfg.n_warmup = n_warmup; cfg.n_thin = 1;
        cfg.batch_size = (batch_size > 0) ? batch_size : N;
        cfg.epsilon = (epsilon > 0) ? epsilon : 0.01;
        cfg.alpha = alpha; cfg.L = L; cfg.verbose = verbose;
        cfg.print_every = 500; cfg.seed = (unsigned int)seed;
        cfg.adapt_epsilon = true;
        tulpa_sghmc::SGHMCResult res = tulpa_sghmc::run_sghmc_sampler(init, in.data, in.layout, cfg);
        if (!res.success) Rcpp::stop("sghmc sampler failed: %s", res.error_msg);
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, D, cn);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = D,
            Rcpp::Named("final_epsilon") = res.epsilon_history.empty() ? 0.0
                                            : res.epsilon_history.back(),
            Rcpp::Named("sampler") = "sghmc");
        return out;
    }

    if (backend == "sgld") {
        tulpa_sghmc::SGLDConfig cfg;
        cfg.n_iter = n_iter; cfg.n_warmup = n_warmup; cfg.n_thin = 1;
        cfg.batch_size = (batch_size > 0) ? batch_size : N;
        cfg.epsilon = (epsilon > 0) ? epsilon : 0.001;
        cfg.verbose = verbose; cfg.print_every = 500; cfg.seed = (unsigned int)seed;
        tulpa_sghmc::SGLDResult res = tulpa_sghmc::run_sgld_sampler(init, in.data, in.layout, cfg);
        if (!res.success) Rcpp::stop("sgld sampler failed: %s", res.error_msg);
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, D, cn);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = D,
            Rcpp::Named("sampler") = "sgld");
        return out;
    }

    if (backend == "mclmc") {
        tulpa_mclmc::MCLMCConfig cfg;
        cfg.n_iter = n_iter; cfg.n_warmup = n_warmup;
        cfg.step_size = epsilon;          // <= 0 -> internal adaptation
        cfg.L = (L > 0) ? L : 0;          // <= 0 -> auto
        cfg.adjusted = mclmc_adjusted;
        cfg.seed = (unsigned int)seed; cfg.verbose = verbose;
        tulpa_mclmc::MCLMCFitResult res = tulpa_mclmc::run_mclmc_sampler(init, in.data, in.layout, cfg);
        if (!res.success) Rcpp::stop("mclmc sampler failed: %s", res.error_msg);
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, D, cn);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = D,
            Rcpp::Named("sampler") = mclmc_adjusted ? "mamclmc" : "mclmc");
        return out;
    }

    if (backend == "smc") {
        tulpa::SMCConfig cfg;
        cfg.n_particles = n_particles; cfg.n_mcmc_steps = n_mcmc_steps;
        cfg.ess_threshold = ess_threshold; cfg.prior_sigma = sigma_beta;
        cfg.seed = (unsigned int)seed; cfg.verbose = verbose;
        tulpa::SMCDriverResult res = tulpa::run_smc_sampler(
            init, in.data, in.layout, cfg, nullptr, nullptr);
        if (!res.success) Rcpp::stop("smc sampler failed: %s", res.error_msg);
        const int M = (int)res.particles.size();
        Rcpp::NumericMatrix draws(M, D);
        for (int s = 0; s < M; s++)
            for (int j = 0; j < D; j++) draws(s, j) = res.particles[s][j];
        if (M > 0) Rcpp::colnames(draws) = cn;
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = M, Rcpp::Named("n_params") = D,
            Rcpp::Named("log_weights") = Rcpp::wrap(res.log_weights),
            Rcpp::Named("log_evidence") = res.log_evidence,
            Rcpp::Named("sampler") = "smc");
        return out;
    }

    if (backend == "vi") {
        tulpa::vi::VIConfig cfg;
        cfg.variant = static_cast<tulpa::vi::VIVariant>(vi_variant);
        cfg.max_iter = vi_max_iter; cfg.mc_samples = vi_mc_samples;
        cfg.seed = (unsigned int)seed; cfg.verbose = verbose;
        tulpa::vi::VIResult res = tulpa::vi::fit_vi(in.data, in.layout, D, cfg, nullptr);
        // Posterior draws for downstream summaries: prefer the kernel's own
        // sample matrix; fall back to sampling the fitted Gaussian when empty.
        Eigen::MatrixXd S = res.samples;
        if (S.rows() == 0 && res.mu.size() == D) {
            // Diagonal-Gaussian draws from (mu, diag(Sigma)) as a minimal fallback.
            Rcpp::RNGScope scope;
            S.resize(vi_n_draws, D);
            for (int s = 0; s < vi_n_draws; s++)
                for (int j = 0; j < D; j++) {
                    double sd = (res.Sigma.rows() == D) ? std::sqrt(std::max(1e-12, res.Sigma(j, j))) : 1.0;
                    S(s, j) = res.mu(j) + sd * R::norm_rand();
                }
        }
        Rcpp::NumericMatrix draws = eigen_draws_to_r(S, D, cn);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = D,
            Rcpp::Named("elbo") = res.final_elbo,
            Rcpp::Named("pareto_k") = res.psis_k,
            Rcpp::Named("converged") = res.converged,
            Rcpp::Named("sampler") = "vi");
        return out;
    }

    Rcpp::stop("Unknown backend '%s' for cpp_tulpa_sample_glmm.", backend);
}
