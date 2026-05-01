// =====================================================================
// GP gradient via autodiff (O(N*nn^3) - much faster than numerical O(N^2))
// Uses templated NNGP likelihood from hmc_gp_autodiff.h
// =====================================================================

// =====================================================================
// Common autodiff prior setup (beta, RE, phi priors)
// Shared by gp_autodiff, msgp_autodiff, gp_temporal_autodiff.
// Returns log_post, sigma_re, phi_num, phi_denom via output parameters.
// =====================================================================
struct AutodiffCommonResult {
    tulpa::ad::Var log_post;
    tulpa::ad::Var sigma_re;
    tulpa::ad::Var phi_num;
    tulpa::ad::Var phi_denom;
};

static inline AutodiffCommonResult add_common_priors_ad(
    tulpa::ad::Tape* tape,
    const std::vector<tulpa::ad::Var>& params_ad,
    const ModelData& data,
    const ParamLayout& layout
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    Var log_post(tape, 0.0);

    // Fixed effects priors: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.legacy.p_num; j++) {
        Var beta = params_ad[layout.legacy.beta_num_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        Var beta = params_ad[layout.legacy.beta_denom_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }

    // Random effects priors (if present)
    Var sigma_re(tape, 1.0);
    if (layout.has_re && data.n_re_groups > 0) {
        Var log_sigma_re = params_ad[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        Var ratio = sigma_re / data.sigma_re_scale;
        log_post = log_post - safe_log(1.0 + ratio * ratio);
        log_post = log_post + log_sigma_re;  // Jacobian

        if (data.re_parameterization == 1) {
            for (int g = 0; g < data.n_re_groups; g++) {
                Var re_g = params_ad[layout.re_start + g];
                log_post = log_post - 0.5 * re_g * re_g;
            }
        } else {
            Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
            for (int g = 0; g < data.n_re_groups; g++) {
                Var re_g = params_ad[layout.re_start + g];
                log_post = log_post - 0.5 * tau_re * re_g * re_g;
                log_post = log_post + 0.5 * safe_log(tau_re);
            }
        }
    }

    // Overdispersion priors (Gamma)
    Var phi_num(tape, 1.0);
    Var phi_denom(tape, 1.0);
    if (layout.legacy.has_phi_num) {
        Var log_phi = params_ad[layout.legacy.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_num + log_phi;
    }
    if (layout.legacy.has_phi_denom) {
        Var log_phi = params_ad[layout.legacy.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_denom + log_phi;
    }

    return {log_post, sigma_re, phi_num, phi_denom};
}

void compute_gradient_gp_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    std::vector<Var> params_ad = make_vars(tape, params);

    auto [log_post, sigma_re, phi_num, phi_denom] = add_common_priors_ad(tape, params_ad, data, layout);

    // =========================================================================
    // GP priors and NNGP likelihood
    // =========================================================================
    std::vector<Var> gp_w_ad;
    Var sigma2_gp(tape, 1.0);
    Var phi_gp(tape, 0.1);

    if (layout.is_gp && data.has_gp) {
        Var log_sigma2_gp = params_ad[layout.log_sigma2_gp_idx];
        Var log_phi_gp = params_ad[layout.log_phi_gp_idx];
        sigma2_gp = safe_exp(log_sigma2_gp);
        phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 (penalizes large variance)
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects
        int N_gp = data.gp_data.n_obs;
        gp_w_ad.resize(N_gp);
        for (int i = 0; i < N_gp; i++) {
            gp_w_ad[i] = params_ad[layout.gp_w_start + i];
        }

        // NNGP log-likelihood using templated function
        Var gp_ll = tulpa_gp::gp_nngp_log_lik_t(gp_w_ad, sigma2_gp, phi_gp, data.gp_data);
        log_post = log_post + gp_ll;
    }

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.legacy.p_num);
    std::vector<Var> beta_denom_ad(data.legacy.p_denom);
    for (int j = 0; j < data.legacy.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.legacy.beta_num_start + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.legacy.beta_denom_start + j];
    }

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            Var re_eff = (data.re_parameterization == 1) ? sigma_re * re_g : re_g;
            eta_num = eta_num + re_eff;
            eta_denom = eta_denom + re_eff;
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && data.has_gp && !gp_w_ad.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            Var gp_effect = gp_w_ad[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = tulpa::math::log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], p);
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num) +
                   tulpa::math::log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_poisson(data.legacy.y_num[i], mu_num) +
                   tulpa::math::log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// Multi-scale GP gradient (hand-coded, ~2-3x faster than autodiff)
// Uses analytical gradients for w and numerical for sigma2/phi
// =====================================================================

void compute_gradient_msgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Multi-scale GP parameters ---
    int N_gp = data.multiscale_gp_data.n_obs;

    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_phi_local = params[layout.log_phi_gp_local_idx];
    double sigma2_local = std::exp(log_sigma2_local);
    double phi_local = std::exp(log_phi_local);

    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_phi_regional = params[layout.log_phi_gp_regional_idx];
    double sigma2_regional = std::exp(log_sigma2_regional);
    double phi_regional = std::exp(log_phi_regional);

    // Extract spatial effects
    std::vector<double> w_local(N_gp), w_regional(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local[i] = params[layout.gp_local_start + i];
        w_regional[i] = params[layout.gp_regional_start + i];
    }

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return; // Out of bounds - return zero gradient
    }

    // --- Shared base priors + MSGP-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // PC priors on GP variances
    grad[layout.log_sigma2_gp_local_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    grad[layout.log_sigma2_gp_regional_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);

    // Jacobians for log-transforms
    grad[layout.log_phi_gp_local_idx] = 1.0;
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // Compute NNGP gradients w.r.t. spatial effects (analytical)
    auto [gp_local, gp_regional] = make_msgp_gp_views(data.multiscale_gp_data);

    tulpa_gp::NNGPGradients nngp_grads_local, nngp_grads_regional;
    tulpa_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local, gp_local, nngp_grads_local);
    tulpa_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional, gp_regional, nngp_grads_regional);

    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_local_start + i] += nngp_grads_local.grad_w[i];
        grad[layout.gp_regional_start + i] += nngp_grads_regional.grad_w[i];
    }

    grad[layout.log_sigma2_gp_local_idx] += nngp_grads_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += nngp_grads_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += nngp_grads_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += nngp_grads_regional.grad_log_phi;

    // =========================================================================
    // Data likelihood loop (scalar per-obs - MSGP scatter requires it)
    // =========================================================================
    for (int i = 0; i < pre.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num += data.legacy.X_num_flat[i * data.legacy.p_num + j] * pre.cp.beta_num[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * pre.cp.beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(pre.cp.re, g, pre.cp.sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        // Multi-scale GP spatial effect
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num += ms_spatial;
            eta_denom += ms_spatial;
        } else {
            eta_num += ms_spatial;
        }

        if (pre.fuse_lp) pre.obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom);

        double dLL_deta_num = 0.0, dLL_deta_denom = 0.0;
        compute_obs_residuals(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, dLL_deta_num, dLL_deta_denom);

        scatter_beta_gradients(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());
        scatter_re_gradient(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());

        // Gradients for GP spatial effects
        double dLL_dspatial = data.multiscale_gp_data.shared ?
                              (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        grad[layout.gp_local_start + loc_i] += dLL_dspatial;
        grad[layout.gp_regional_start + loc_i] += dLL_dspatial;

        accumulate_phi_likelihood_grad(data, layout, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, grad.data());
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Multi-scale GP gradient (autodiff, ~3x faster than numerical)
// =====================================================================

void compute_gradient_msgp_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    std::vector<Var> params_ad = make_vars(tape, params);

    auto [log_post, sigma_re, phi_num, phi_denom] = add_common_priors_ad(tape, params_ad, data, layout);

    // =========================================================================
    // Multi-scale GP priors and NNGP likelihoods
    // =========================================================================
    int N_gp = data.multiscale_gp_data.n_obs;

    // Local scale parameters
    Var log_sigma2_local = params_ad[layout.log_sigma2_gp_local_idx];
    Var log_phi_local = params_ad[layout.log_phi_gp_local_idx];
    Var sigma2_local = safe_exp(log_sigma2_local);
    Var phi_local = safe_exp(log_phi_local);

    // Regional scale parameters
    Var log_sigma2_regional = params_ad[layout.log_sigma2_gp_regional_idx];
    Var log_phi_regional = params_ad[layout.log_phi_gp_regional_idx];
    Var sigma2_regional = safe_exp(log_sigma2_regional);
    Var phi_regional = safe_exp(log_phi_regional);

    // PC priors on variances
    log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    log_post = log_post + log_sigma2_local;  // Jacobian

    log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
    log_post = log_post + log_sigma2_regional;  // Jacobian

    // Range priors (uniform within bounds) - check bounds, return -inf if violated
    double phi_local_val = get_value(phi_local);
    double phi_regional_val = get_value(phi_regional);
    if (phi_local_val < data.multiscale_gp_data.range_local_lower ||
        phi_local_val > data.multiscale_gp_data.range_local_upper ||
        phi_regional_val < data.multiscale_gp_data.range_regional_lower ||
        phi_regional_val > data.multiscale_gp_data.range_regional_upper) {
        // Out of bounds - return zero gradients (log_post = -inf)
        // TapeScope destructor handles cleanup
        return;
    }
    log_post = log_post + log_phi_local;    // Jacobian
    log_post = log_post + log_phi_regional; // Jacobian

    // Extract GP spatial effects
    std::vector<Var> w_local_ad(N_gp);
    std::vector<Var> w_regional_ad(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local_ad[i] = params_ad[layout.gp_local_start + i];
        w_regional_ad[i] = params_ad[layout.gp_regional_start + i];
    }

    // NNGP log-likelihood for each scale using templated function
    Var msgp_ll = tulpa_gp::multiscale_gp_log_lik_t(
        w_local_ad, w_regional_ad,
        sigma2_local, phi_local,
        sigma2_regional, phi_regional,
        data.multiscale_gp_data);
    log_post = log_post + msgp_ll;

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.legacy.p_num);
    std::vector<Var> beta_denom_ad(data.legacy.p_denom);
    for (int j = 0; j < data.legacy.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.legacy.beta_num_start + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.legacy.beta_denom_start + j];
    }

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            Var re_eff = (data.re_parameterization == 1) ? sigma_re * re_g : re_g;
            eta_num = eta_num + re_eff;
            eta_denom = eta_denom + re_eff;
        }

        // Add multi-scale GP spatial effect (map observation to unique location)
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        Var ms_spatial = w_local_ad[loc_i] + w_regional_ad[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num = eta_num + ms_spatial;
            eta_denom = eta_denom + ms_spatial;
        } else {
            eta_num = eta_num + ms_spatial;
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = tulpa::math::log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], p);
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num) +
                   tulpa::math::log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_poisson(data.legacy.y_num[i], mu_num) +
                   tulpa::math::log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// GP + Temporal gradient (autodiff, combines GP and temporal effects)
// =====================================================================

void compute_gradient_gp_plus_temporal_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    std::vector<Var> params_ad = make_vars(tape, params);

    auto [log_post, sigma_re, phi_num, phi_denom] = add_common_priors_ad(tape, params_ad, data, layout);

    // =========================================================================
    // Temporal priors
    // =========================================================================
    Var tau_temporal(tape, 1.0);
    Var rho_ar1(tape, 0.0);
    std::vector<std::vector<Var>> phi_temporal_ad;  // [group][time]

    if (layout.has_temporal) {
        Var log_tau_temporal = params_ad[layout.log_tau_temporal_idx];
        tau_temporal = safe_exp(log_tau_temporal);

        // tau ~ Gamma(shape, rate) with Jacobian
        log_post = log_post + (data.tau_temporal_shape - 1.0) * log_tau_temporal
                            - data.tau_temporal_rate * tau_temporal
                            + log_tau_temporal;

        // AR1: also estimate rho
        if (layout.is_ar1) {
            Var logit_rho = params_ad[layout.logit_rho_ar1_idx];
            rho_ar1 = 1.0 / (1.0 + safe_exp(-logit_rho));

            // rho ~ Uniform(0,1) prior with logit Jacobian
            log_post = log_post + safe_log(rho_ar1) + safe_log(1.0 - rho_ar1);
        }

        // Extract temporal effects for each group
        int T = data.n_times;
        phi_temporal_ad.resize(data.n_temporal_groups);
        for (int g = 0; g < data.n_temporal_groups; g++) {
            phi_temporal_ad[g].resize(T);
            for (int t = 0; t < T; t++) {
                phi_temporal_ad[g][t] = params_ad[layout.temporal_start + g * T + t];
            }

            // Temporal prior
            Var temporal_prior = tulpa_temporal::temporal_log_prior_t(
                phi_temporal_ad[g], data.temporal_type, tau_temporal, rho_ar1, data.temporal_cyclic);
            log_post = log_post + temporal_prior;
        }
    }

    // =========================================================================
    // GP priors and NNGP likelihood
    // =========================================================================
    std::vector<Var> gp_w_ad;
    Var sigma2_gp(tape, 1.0);
    Var phi_gp(tape, 0.1);

    if (layout.is_gp && data.has_gp) {
        Var log_sigma2_gp = params_ad[layout.log_sigma2_gp_idx];
        Var log_phi_gp = params_ad[layout.log_phi_gp_idx];
        sigma2_gp = safe_exp(log_sigma2_gp);
        phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 (penalizes large variance)
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects
        int N_gp = data.gp_data.n_obs;
        gp_w_ad.resize(N_gp);
        for (int i = 0; i < N_gp; i++) {
            gp_w_ad[i] = params_ad[layout.gp_w_start + i];
        }

        // NNGP log-likelihood using templated function
        Var gp_ll = tulpa_gp::gp_nngp_log_lik_t(gp_w_ad, sigma2_gp, phi_gp, data.gp_data);
        log_post = log_post + gp_ll;
    }

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.legacy.p_num);
    std::vector<Var> beta_denom_ad(data.legacy.p_denom);
    for (int j = 0; j < data.legacy.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.legacy.beta_num_start + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.legacy.beta_denom_start + j];
    }

    int T = data.n_times;

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            Var re_eff = (data.re_parameterization == 1) ? sigma_re * re_g : re_g;
            eta_num = eta_num + re_eff;
            eta_denom = eta_denom + re_eff;
        }

        // Add temporal effect
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;  // 0-based
            int g = data.temporal_group_idx[i] - 1; // 0-based
            if (g >= 0 && g < (int)phi_temporal_ad.size() &&
                t >= 0 && t < (int)phi_temporal_ad[g].size()) {
                Var temporal_effect = phi_temporal_ad[g][t];
                if (data.temporal_shared) {
                    eta_num = eta_num + temporal_effect;
                    eta_denom = eta_denom + temporal_effect;
                } else {
                    eta_num = eta_num + temporal_effect;
                }
            }
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && data.has_gp && !gp_w_ad.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            Var gp_effect = gp_w_ad[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = tulpa::math::log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], p);
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num) +
                   tulpa::math::log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_poisson(data.legacy.y_num[i], mu_num) +
                   tulpa::math::log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}
