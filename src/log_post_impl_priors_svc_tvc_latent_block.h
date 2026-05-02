// log_post_impl_priors_svc_tvc_latent_block.h
// Function-body fragment of compute_log_post_impl<T> in log_post_impl.h.
// Included via #include directly inside the function body — relies on the
// surrounding lexical scope (params, data, layout, log_post, T, beta_*,
// phi_*, re_vals, re_term_offsets, etc.). NOT standalone-compilable.
// No header guard: each fragment is included exactly once per umbrella.
// SVC (spatially-varying coefficients), TVC (temporally-varying
// coefficients), and latent-factor priors.

    // SVC (Spatially-Varying Coefficients) parameters and priors
    std::vector<T> svc_sigma2;
    std::vector<T> svc_phi;
    std::vector<T> svc_w_flat;
    std::vector<T> svc_eta;

    if (layout.has_svc && data.svc_data.n_svc > 0) {
        int n_svc = data.svc_data.n_svc;
        int n_obs = data.svc_data.n_obs;

        if (data.svc_is_hsgp) {
            // HSGP-based SVC: basis function approximation
            int m_total = data.svc_hsgp_data.m_total;

            // Per-term: sigma2, lengthscale, beta[m_total]
            svc_eta.resize(n_obs, T(0.0));
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                T sigma2_j = safe_exp(log_sigma2);

                // PC prior on sigma
                T sigma_j = safe_sqrt(sigma2_j);
                double rate_sigma = 4.6;
                log_post = log_post - rate_sigma * sigma_j + T(0.5) * log_sigma2;

                T log_ls = params[layout.log_phi_svc_start + j];
                T ls_j = safe_exp(log_ls);

                // LogNormal(0,1) on lengthscale
                log_post = log_post - T(0.5) * log_ls * log_ls;

                // Extract beta_j and compute f_j = Phi * (sqrt(S_j) * beta_j)
                // N(0, I) prior on beta
                for (int k = 0; k < m_total; k++) {
                    T beta_jk = params[layout.svc_w_start + j * m_total + k];
                    log_post = log_post - T(0.5) * beta_jk * beta_jk;

                    // Compute scaled beta: sqrt(S(eigenvalue_k, sigma2_j, ls_j)) * beta_jk
                    double omega_sq = data.svc_hsgp_data.eigenvalues[k];
                    T S_k = sigma2_j * T(std::sqrt(2.0 * M_PI)) * ls_j *
                            safe_exp(T(-0.5) * ls_j * ls_j * T(omega_sq));
                    T sqrt_S_k = safe_sqrt(S_k);

                    // Accumulate f_j[i] = sum_k phi[i,k] * sqrt_S_k * beta_jk
                    for (int i = 0; i < n_obs; i++) {
                        double phi_ik = data.svc_hsgp_data.phi_flat[i * m_total + k];
                        svc_eta[i] = svc_eta[i] + T(phi_ik) * sqrt_S_k * beta_jk *
                                     T(data.svc_data.X_svc[i * n_svc + j]);
                    }
                }
            }
        } else {
            // NNGP-based SVC (original path)

            // Extract sigma2 (spatial variance) parameters
            svc_sigma2.resize(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                svc_sigma2[j] = safe_exp(log_sigma2);

                T sigma = safe_sqrt(svc_sigma2[j]);
                double scale = data.svc_sigma2_prior_scale;
                log_post = log_post - safe_log(T(1.0) + sigma * sigma / T(scale * scale));
                log_post = log_post + log_sigma2;
            }

            // Extract phi (spatial range) parameters
            svc_phi.resize(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_phi = params[layout.log_phi_svc_start + j];
                svc_phi[j] = safe_exp(log_phi);

                double phi_val = get_value(svc_phi[j]);
                if (phi_val < data.svc_phi_prior_lower || phi_val > data.svc_phi_prior_upper) {
                    return T(-INFINITY);
                }
                log_post = log_post + log_phi;
            }

            // Extract SVC values
            int n_svc_params = n_svc * n_obs;
            svc_w_flat.resize(n_svc_params);
            for (int k = 0; k < n_svc_params; k++) {
                svc_w_flat[k] = params[layout.svc_w_start + k];
            }

            // NNGP prior on each SVC term
            for (int j = 0; j < n_svc; j++) {
                std::vector<T> w_j(n_obs);
                for (int k = 0; k < n_obs; k++) {
                    w_j[k] = svc_w_flat[j * n_obs + k];
                }
                log_post = log_post + tulpa_svc_ad::nngp_log_lik(w_j, svc_sigma2[j], svc_phi[j], data.svc_data);
            }

            // Soft sum-to-zero constraint
            log_post = log_post + tulpa_svc_ad::svc_sum_to_zero_penalty(svc_w_flat, data.svc_data, 1.0);

            // Precompute SVC contribution to linear predictor
            svc_eta.resize(n_obs, T(0.0));
            tulpa_svc_ad::compute_svc_eta(svc_w_flat, data.svc_data, svc_eta);
        }
    }

    // TVC (Temporally-Varying Coefficients) parameters and priors
    std::vector<T> tvc_tau;
    std::vector<T> tvc_rho;
    std::vector<T> tvc_w_flat;
    std::vector<T> tvc_eta;

    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        int n_tvc = data.tvc_data.n_tvc;
        int n_groups = data.tvc_data.n_groups;
        int n_times = data.tvc_data.n_times;
        int n_obs = data.tvc_data.n_obs;

        // Extract tau (precision) parameters
        tvc_tau.resize(n_tvc);
        for (int j = 0; j < n_tvc; j++) {
            T log_tau = params[layout.log_tau_tvc_start + j];
            tvc_tau[j] = safe_exp(log_tau);

            // PC prior on tau (exponential prior on sigma = 1/sqrt(tau))
            // P(sigma > U) = alpha  =>  rate = -log(alpha) / U
            T sigma_tvc = T(1.0) / safe_sqrt(tvc_tau[j]);
            T rate_tvc = T(-std::log(0.01) / 1.0);  // P(sigma > 1) = 0.01
            log_post = log_post + safe_log(rate_tvc) - rate_tvc * sigma_tvc
                     - safe_log(T(2.0) * sigma_tvc);
            log_post = log_post + log_tau;  // Jacobian for log transform
        }

        // Extract rho (AR1 correlation) parameters if AR1 structure
        tvc_rho.resize(n_tvc, T(0.0));
        if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
            for (int j = 0; j < n_tvc; j++) {
                T logit_rho = params[layout.logit_rho_tvc_start + j];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho);
                tvc_rho[j] = T(2.0) * u - T(1.0);

                // Uniform(-1, 1) prior on rho
                // Jacobian for logit((rho+1)/2) transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        tvc_w_flat.resize(n_tvc_params);
        for (int k = 0; k < n_tvc_params; k++) {
            tvc_w_flat[k] = params[layout.tvc_w_start + k];
        }

        // TVC temporal prior (RW1, RW2, or AR1)
        log_post = log_post + tulpa_tvc_ad::tvc_log_prior(
            tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho
        );

        // Soft sum-to-zero constraint for identifiability
        log_post = log_post + tulpa_tvc_ad::tvc_sum_to_zero_penalty(
            tvc_w_flat, data.tvc_data, 0.001
        );

        // Precompute TVC contribution to linear predictor
        tvc_eta.resize(n_obs, T(0.0));
        tulpa_tvc_ad::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
    }

    // Latent factors parameters and priors
    std::vector<T> latent_sigma;
    std::vector<T> latent_factors;
    std::vector<T> latent_eta;

    if (layout.has_latent && data.latent_n_factors > 0) {
        int K = data.latent_n_factors;
        int N = data.N;

        // Extract log_sigma parameters
        std::vector<T> log_sigma_latent(K);
        for (int k = 0; k < K; k++) {
            log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        }

        // Compute sigma from log_sigma
        latent_sigma.resize(K);
        tulpa_latent_ad::extract_sigma(latent_sigma, log_sigma_latent);

        // Extract factors and apply constraint
        int n_factor_params = N * K;
        latent_factors.resize(n_factor_params);
        for (int j = 0; j < n_factor_params; j++) {
            latent_factors[j] = params[layout.latent_factor_start + j];
        }

        // Apply identifiability constraint
        if (data.latent_constraint == 0) {  // SUM_TO_ZERO
            tulpa_latent_ad::apply_sum_to_zero(latent_factors, N, K);
        } else {  // FIRST_ZERO
            tulpa_latent_ad::apply_first_zero(latent_factors, N, K);
        }

        // Sigma prior: Exponential on sigma (PC prior)
        log_post = log_post + tulpa_latent_ad::latent_sigma_log_prior(
            log_sigma_latent, data.latent_sigma_prior_rate
        );

        // Factor prior: N(0, 1) on each factor score
        tulpa_latent::LatentConstraint constraint =
            (data.latent_constraint == 0) ? tulpa_latent::LatentConstraint::SUM_TO_ZERO
                                          : tulpa_latent::LatentConstraint::FIRST_ZERO;
        log_post = log_post + tulpa_latent_ad::latent_factor_log_prior(
            latent_factors, N, K, constraint
        );

        // Precompute latent factor contribution to linear predictor
        latent_eta.resize(N, T(0.0));
        tulpa_latent_ad::latent_contributions_all(latent_eta, latent_factors, latent_sigma, N, K);
    }


