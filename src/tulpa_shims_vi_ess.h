// ============================================================================
// VI shim
// ============================================================================

extern "C" void tulpa_fit_vi_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    int D,
    const tulpa::VIShimConfig* shim_config,
    const double* init_mu,
    int n_init_mu,
    tulpa::VIShimResult* result_out
) {
    using tulpa::vi::VIConfig;
    using tulpa::vi::VIVariant;

    VIConfig cfg;
    cfg.variant = static_cast<VIVariant>(shim_config->variant);
    cfg.max_iter = shim_config->max_iter;
    cfg.mc_samples = shim_config->mc_samples;
    cfg.tol_grad = shim_config->tol_grad;
    cfg.tol_rel_elbo = shim_config->tol_rel_elbo;
    cfg.patience = shim_config->patience;
    cfg.adam_alpha = shim_config->adam_alpha;
    cfg.adam_beta1 = shim_config->adam_beta1;
    cfg.adam_beta2 = shim_config->adam_beta2;
    cfg.adam_eps = shim_config->adam_eps;
    cfg.rank = shim_config->rank;
    cfg.use_laplace_init = shim_config->use_laplace_init != 0;
    cfg.fullrank_threshold = shim_config->fullrank_threshold;
    cfg.lowrank_threshold = shim_config->lowrank_threshold;
    cfg.verbose = shim_config->verbose != 0;
    cfg.print_every = shim_config->print_every;
    cfg.seed = shim_config->seed;

    Eigen::VectorXd init_vec;
    const Eigen::VectorXd* init_ptr = nullptr;
    if (init_mu && n_init_mu == D) {
        init_vec = Eigen::Map<const Eigen::VectorXd>(init_mu, D);
        init_ptr = &init_vec;
    }

    tulpa::vi::VIResult res = tulpa::vi::fit_vi(*data, *layout, D, cfg, init_ptr);

    result_out->variant_used = static_cast<int>(res.variant_used);
    result_out->D = D;
    result_out->rank_used = res.rank_used;
    result_out->iterations = res.iterations;
    result_out->converged = res.converged ? 1 : 0;
    result_out->final_elbo = res.final_elbo;
    result_out->psis_k = res.psis_k;

    result_out->mu       = nullptr;
    result_out->Sigma    = nullptr;
    result_out->L_factor = nullptr;
    result_out->d_diag   = nullptr;
    result_out->elbo_history = nullptr;

    if (res.mu.size() > 0) {
        int d = res.mu.size();
        result_out->mu = new double[d];
        for (int i = 0; i < d; i++) result_out->mu[i] = res.mu(i);
    }
    if (res.Sigma.rows() > 0 && res.Sigma.cols() > 0) {
        int rows = res.Sigma.rows(), cols = res.Sigma.cols();
        result_out->Sigma = new double[(size_t)rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_out->Sigma[(size_t)i * cols + j] = res.Sigma(i, j);
            }
        }
    }
    if (res.L_factor.rows() > 0 && res.L_factor.cols() > 0) {
        int rows = res.L_factor.rows(), cols = res.L_factor.cols();
        result_out->L_factor = new double[(size_t)rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_out->L_factor[(size_t)i * cols + j] = res.L_factor(i, j);
            }
        }
    }
    if (res.d_diag.size() > 0) {
        int d = res.d_diag.size();
        result_out->d_diag = new double[d];
        for (int i = 0; i < d; i++) result_out->d_diag[i] = res.d_diag(i);
    }
    if (!res.elbo_history.empty()) {
        size_t n = res.elbo_history.size();
        result_out->elbo_history = new double[n];
        for (size_t i = 0; i < n; i++) result_out->elbo_history[i] = res.elbo_history[i];
    }
}

// ============================================================================
// ESS shim
// ============================================================================

extern "C" void tulpa_run_ess_sampler_impl(
    const double* init_params,
    int n_params,
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const tulpa::ESSShimConfig* shim_config,
    tulpa::ESSShimResult* result_out
) {
    std::vector<double> init(init_params, init_params + n_params);

    tulpa_ess::ESSConfig cfg;
    cfg.n_iter = shim_config->n_iter;
    cfg.n_warmup = shim_config->n_warmup;
    cfg.n_thin = shim_config->n_thin;
    cfg.verbose = shim_config->verbose != 0;
    cfg.print_every = shim_config->print_every;
    cfg.seed = shim_config->seed;
    cfg.use_cholesky = shim_config->use_cholesky != 0;
    cfg.adapt_during_warmup = shim_config->adapt_during_warmup != 0;
    cfg.adapt_interval = shim_config->adapt_interval;
    cfg.joint_sigma_re = shim_config->joint_sigma_re != 0;

    tulpa_ess::ESSResult res = tulpa_ess::run_ess_sampler(init, *data, *layout, cfg);

    int n_save = res.samples.rows();
    int n_p    = res.samples.cols();
    result_out->n_save = n_save;
    result_out->n_params = n_p;
    result_out->n_slice_evals = res.n_slice_evals;
    result_out->avg_slice_evals = res.avg_slice_evals;
    result_out->success = res.success ? 1 : 0;
    result_out->samples = nullptr;
    result_out->log_lik = nullptr;
    std::strncpy(result_out->error_msg,
                 res.error_msg.c_str(),
                 sizeof(result_out->error_msg) - 1);
    result_out->error_msg[sizeof(result_out->error_msg) - 1] = '\0';

    if (n_save > 0 && n_p > 0) {
        result_out->samples = new double[(size_t)n_save * n_p];
        for (int i = 0; i < n_save; i++) {
            for (int j = 0; j < n_p; j++) {
                result_out->samples[(size_t)i * n_p + j] = res.samples(i, j);
            }
        }
    }
    if (!res.log_lik.empty()) {
        size_t n = res.log_lik.size();
        result_out->log_lik = new double[n];
        for (size_t i = 0; i < n; i++) result_out->log_lik[i] = res.log_lik[i];
    }
}

