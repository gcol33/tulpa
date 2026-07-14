
// ============================================================================
// SGHMC / SGLD shims
// ============================================================================
//
// Both samplers run on the same NUTS-style ModelData + ParamLayout interface
// (see nuts_api.h). The shim forwards raw scalar config, builds the sampler
// config struct, calls the underlying tulpa::run_sghmc_sampler /
// run_sgld_sampler, and copies the result into a flat
// SGSamplerShimResult buffer.

#include "shim_guard.h"

namespace {

inline void copy_sg_sampler_result(
    const Eigen::MatrixXd& samples,
    const std::vector<double>& log_lik,
    const std::vector<double>& epsilon_history,
    bool success,
    const std::string& error_msg,
    tulpa::SGSamplerShimResult* out
) {
    int n_save   = (int)samples.rows();
    int n_params = (int)samples.cols();
    out->n_sample = n_save;
    out->n_params = n_params;
    out->success  = success ? 1 : 0;
    std::strncpy(out->error_msg, error_msg.c_str(), sizeof(out->error_msg) - 1);
    out->error_msg[sizeof(out->error_msg) - 1] = '\0';

    out->samples = new double[(size_t)n_save * (size_t)n_params];
    for (int i = 0; i < n_save; i++) {
        for (int j = 0; j < n_params; j++) {
            out->samples[(size_t)i * n_params + j] = samples(i, j);
        }
    }

    out->log_lik = new double[n_save];
    for (int i = 0; i < n_save; i++) out->log_lik[i] = log_lik[i];

    int n_eps = (int)epsilon_history.size();
    out->n_eps_history  = n_eps;
    out->epsilon_history = new double[n_eps > 0 ? n_eps : 1];
    for (int i = 0; i < n_eps; i++) out->epsilon_history[i] = epsilon_history[i];
    out->final_epsilon = (n_eps > 0) ? epsilon_history[n_eps - 1] : 0.0;
}

} // namespace

extern "C" void tulpa_sghmc_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double alpha,
    int L,
    unsigned int seed,
    int adapt_eps,
    double grad_clip,
    int verbose,
    tulpa::SGSamplerShimResult* result_out
) {
    TULPA_SHIM_GUARD_BEGIN
    std::vector<double> q_init(init, init + n_params);

    tulpa_sghmc::SGHMCConfig cfg;
    cfg.n_iter        = n_iter;
    cfg.n_warmup      = n_warmup;
    cfg.batch_size    = batch_size;
    cfg.epsilon       = epsilon;
    cfg.alpha         = alpha;
    cfg.L             = L;
    cfg.seed          = seed;
    cfg.adapt_epsilon = (adapt_eps != 0);
    cfg.grad_clip     = grad_clip;
    cfg.verbose       = (verbose != 0);

    tulpa_sghmc::SGHMCResult res =
        tulpa_sghmc::run_sghmc_sampler(q_init, *data, *layout, cfg);

    copy_sg_sampler_result(res.samples, res.log_lik, res.epsilon_history,
                           res.success, res.error_msg, result_out);
    TULPA_SHIM_GUARD_END("tulpa_sghmc_fit")
}

extern "C" void tulpa_sgld_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double schedule_a,
    double schedule_b,
    double schedule_gamma,
    int use_schedule,
    double grad_clip,
    unsigned int seed,
    int verbose,
    tulpa::SGSamplerShimResult* result_out
) {
    TULPA_SHIM_GUARD_BEGIN
    std::vector<double> q_init(init, init + n_params);

    tulpa_sghmc::SGLDConfig cfg;
    cfg.n_iter         = n_iter;
    cfg.n_warmup       = n_warmup;
    cfg.batch_size     = batch_size;
    cfg.epsilon        = epsilon;
    cfg.schedule_a     = schedule_a;
    cfg.schedule_b     = schedule_b;
    cfg.schedule_gamma = schedule_gamma;
    cfg.use_schedule   = (use_schedule != 0);
    cfg.grad_clip      = grad_clip;
    cfg.seed           = seed;
    cfg.verbose        = (verbose != 0);

    tulpa_sghmc::SGLDResult res =
        tulpa_sghmc::run_sgld_sampler(q_init, *data, *layout, cfg);

    copy_sg_sampler_result(res.samples, res.log_lik, res.epsilon_history,
                           res.success, res.error_msg, result_out);
    TULPA_SHIM_GUARD_END("tulpa_sgld_fit")
}

// ============================================================================
// MCLMC / MAMCLMC shim
// ============================================================================
//
// Both samplers internally take a std::function log_prob_grad callback that
// cannot cross a DLL boundary. The shim builds it inside tulpa from the
// existing compute_log_post + compute_gradient (see mclmc_modeldata.h),
// then dispatches to mclmc_sample / mamclmc_sample based on the `adjusted`
// flag. Result reuses SGSamplerShimResult — the flat shape fits.

extern "C" void tulpa_mclmc_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    double step_size,
    int L,
    unsigned int seed,
    int adjusted,
    int verbose,
    tulpa::SGSamplerShimResult* result_out
) {
    TULPA_SHIM_GUARD_BEGIN
    std::vector<double> q_init(init, init + n_params);

    tulpa_mclmc::MCLMCConfig cfg;
    cfg.n_iter   = n_iter;
    cfg.n_warmup = n_warmup;
    cfg.step_size = step_size;
    cfg.L        = L;
    cfg.adjusted = adjusted;
    cfg.seed     = seed;
    cfg.verbose  = (verbose != 0);
    // mass_diag stays empty (identity); extending the shim signature would
    // require an ABI bump.

    tulpa_mclmc::MCLMCFitResult res =
        tulpa_mclmc::run_mclmc_sampler(q_init, *data, *layout, cfg);

    copy_sg_sampler_result(res.samples, res.log_lik, res.epsilon_history,
                           res.success, res.error_msg, result_out);
    TULPA_SHIM_GUARD_END("tulpa_mclmc_fit")
}

// ============================================================================
// SMC shim
// ============================================================================
//
// Drives tulpa::run_smc_sampler. The mutation kernel is pluggable: if
// the model package passes nullptr, tulpa uses its built-in random-walk
// Metropolis kernel scaled by 1 / sqrt(beta) targeting
// log_prior + beta * log_lik.

extern "C" void tulpa_smc_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_particles,
    int n_mcmc_steps,
    double ess_threshold,
    double prior_sigma,
    tulpa::SmcMutationFn mutation,
    void* user_data,
    unsigned int seed,
    int verbose,
    tulpa::SMCShimResult* result_out
) {
    TULPA_SHIM_GUARD_BEGIN
    std::vector<double> init_vec(init, init + n_params);

    tulpa::SMCConfig cfg;
    cfg.n_particles    = n_particles;
    cfg.n_mcmc_steps   = n_mcmc_steps;
    cfg.ess_threshold  = ess_threshold;
    cfg.prior_sigma    = prior_sigma;
    cfg.seed           = seed;
    cfg.verbose        = (verbose != 0);

    tulpa::SMCDriverResult res = tulpa::run_smc_sampler(
        init_vec, *data, *layout, cfg, mutation, user_data);

    int N = (int)res.particles.size();
    int P = (N > 0) ? (int)res.particles[0].size() : n_params;

    result_out->n_particles  = N;
    result_out->n_params     = P;
    result_out->log_evidence = res.log_evidence;
    result_out->success      = res.success ? 1 : 0;

    std::strncpy(result_out->error_msg, res.error_msg.c_str(),
                 sizeof(result_out->error_msg) - 1);
    result_out->error_msg[sizeof(result_out->error_msg) - 1] = '\0';

    if (N > 0 && P > 0) {
        result_out->particles = new double[(size_t)N * (size_t)P];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                result_out->particles[(size_t)i * P + j] = res.particles[i][j];
            }
        }
    } else {
        result_out->particles = new double[1];
    }

    {
        int W = (int)res.log_weights.size();
        result_out->log_weights = new double[W > 0 ? W : 1];
        for (int i = 0; i < W; i++) result_out->log_weights[i] = res.log_weights[i];
    }
    TULPA_SHIM_GUARD_END("tulpa_smc_fit")
}
