// tulpa_sample_glmm.cpp
// ----------------------------------------------------------------------------
// Generic GLMM front-door entry for the model-agnostic sampler kernels that
// previously had a C-ABI kernel but no R fitter (gcol33/tulpa#54, #55): NUTS,
// elliptical slice sampling (ESS), SGHMC, SGLD, MCLMC, SMC, and VI. Every one
// of these kernels consumes a tulpa ModelData + ParamLayout, so a single entry
// builds that model once -- through the SAME built-in-family spec scaffold the
// single-point Laplace fit uses (build_spec_family_inputs, the single source of
// truth for the per-observation likelihood + design) -- and dispatches to the
// chosen kernel by name. No likelihood / link / layout logic is duplicated.
//
// Scope: fixed-effect GLMs (binomial / poisson / gaussian / neg_binomial_2 /
// the rest of the built-in family set). Random-effect terms are NOT handled
// here: the ModelData-kernel path would have to either sample log(sigma_re)
// jointly (which needs a hyperprior the conditional-Laplace ModelData does not
// carry) or condition on it (which needs the kernels to hold a parameter fixed,
// which they cannot). Random-intercept models therefore stay on the conditional
// R-closure logpost backends (mode = "mala" / "pathfinder" / "imh_laplace"),
// which condition on sigma_re cleanly. The R fitter enforces this split.
// ----------------------------------------------------------------------------

#include <Rcpp.h>
#include <string>
#include <vector>
#include <cmath>

#include "laplace_spec_fit.h"   // build_spec_family_inputs / SpecFamilyInputs
#include "hmc_sampler.h"        // tulpa_hmc::run_hmc_parallel_chains_cpp, HMCResultCpp
#include "ess_sampler.h"        // tulpa_ess::run_ess_sampler
#include "sghmc_sampler.h"      // tulpa_sghmc::run_sghmc_sampler / run_sgld_sampler
#include "mclmc_modeldata.h"    // tulpa_mclmc::run_mclmc_sampler
#include "smc_modeldata.h"      // tulpa::run_smc_sampler
#include "vi_types.h"           // tulpa::vi::VIConfig / VIResult / VIVariant

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// fit_vi is defined in vi_sampler.cpp without a public header declaration; the
// VI shim reaches it the same way. Forward-declare the exact signature.
namespace tulpa { namespace vi {
VIResult fit_vi(const ModelData& data, const ParamLayout& layout, int D,
                const VIConfig& config, const Eigen::VectorXd* init_mu);
} }

namespace {

// Build an [n_save x p] R matrix of draws from an Eigen sample matrix, naming
// the columns beta[1..p]. The kernels report fixed-effect draws only here.
Rcpp::NumericMatrix eigen_draws_to_r(const Eigen::MatrixXd& S, int p) {
    const int n = (int)S.rows();
    Rcpp::NumericMatrix draws(n, p);
    for (int s = 0; s < n; s++)
        for (int j = 0; j < p; j++) draws(s, j) = S(s, j);
    Rcpp::CharacterVector cn(p);
    for (int j = 0; j < p; j++) cn[j] = "beta[" + std::to_string(j + 1) + "]";
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
    int vi_n_draws = 2000
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
    const int p = X.ncol();
    const int N = y.size();

    // Build the ModelData + ParamLayout once via the shared built-in-family
    // scaffold (no RE: empty re_group, n_re_groups = 0).
    tulpa::SpecFamilyInputs in;
    std::vector<int> no_re;
    tulpa::build_spec_family_inputs(
        in, y, n_trials, X, no_re, /*n_re_groups=*/0, /*sigma_re=*/1.0,
        family, phi, sigma_beta, /*n_block_latent=*/0);
    const int D = in.layout.total_params;   // == p for the no-RE fixed-effect path
    std::vector<double> init(D, 0.0);

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
        Rcpp::NumericMatrix draws(n_total, p);
        Rcpp::IntegerVector chain_id(n_total);
        Rcpp::NumericVector log_prob(n_total), accept_prob(n_total);
        Rcpp::IntegerVector divergent(n_total), treedepth(n_total);
        Rcpp::NumericVector epsilon_out(n_chains);
        int r = 0;
        for (int c = 0; c < n_chains; c++) {
            const tulpa_hmc::HMCResultCpp& ch = chains[c];
            for (int s = 0; s < ch.n_sample; s++) {
                const double* row = ch.sample_row(s);
                for (int j = 0; j < p; j++) draws(r, j) = row[j];
                chain_id[r] = c + 1;
                log_prob[r] = ch.log_prob[s];
                accept_prob[r] = ch.accept_prob[s];
                divergent[r] = ch.divergent[s];
                treedepth[r] = ch.treedepth[s];
                r++;
            }
            epsilon_out[c] = ch.epsilon;
        }
        Rcpp::CharacterVector cn(p);
        for (int j = 0; j < p; j++) cn[j] = "beta[" + std::to_string(j + 1) + "]";
        if (n_total > 0) Rcpp::colnames(draws) = cn;
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws,
            Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("chain_id") = chain_id,
            Rcpp::Named("n_chains") = n_chains,
            Rcpp::Named("n_samples") = n_sample,
            Rcpp::Named("n_params") = p,
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
        cfg.adapt_interval = 50; cfg.joint_sigma_re = false;
        cfg.joint_sigma_proposal_sd = 0.1;
        tulpa_ess::ESSResult res = tulpa_ess::run_ess_sampler(init, in.data, in.layout, cfg);
        if (!res.success) Rcpp::stop("ess sampler failed: %s", res.error_msg);
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, p);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws,
            Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(),
            Rcpp::Named("n_params") = p,
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
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, p);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = p,
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
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, p);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = p,
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
        Rcpp::NumericMatrix draws = eigen_draws_to_r(res.samples, p);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = p,
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
        Rcpp::NumericMatrix draws(M, p);
        for (int s = 0; s < M; s++)
            for (int j = 0; j < p; j++) draws(s, j) = res.particles[s][j];
        Rcpp::CharacterVector cn(p);
        for (int j = 0; j < p; j++) cn[j] = "beta[" + std::to_string(j + 1) + "]";
        if (M > 0) Rcpp::colnames(draws) = cn;
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = M, Rcpp::Named("n_params") = p,
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
        Rcpp::NumericMatrix draws = eigen_draws_to_r(S, p);
        out = Rcpp::List::create(
            Rcpp::Named("draws") = draws, Rcpp::Named("means") = col_means(draws),
            Rcpp::Named("n_samples") = draws.nrow(), Rcpp::Named("n_params") = p,
            Rcpp::Named("elbo") = res.final_elbo,
            Rcpp::Named("pareto_k") = res.psis_k,
            Rcpp::Named("converged") = res.converged,
            Rcpp::Named("sampler") = "vi");
        return out;
    }

    Rcpp::stop("Unknown backend '%s' for cpp_tulpa_sample_glmm.", backend);
}
