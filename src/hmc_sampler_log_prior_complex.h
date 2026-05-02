  // Latent factor priors
  // latent_sigma / latent_factors_vec live in state (aliased above).
  if (layout.has_latent && data.latent_n_factors > 0) {
    int K = data.latent_n_factors;
    int N = data.N;

    // Extract log_sigma parameters
    std::vector<double> log_sigma_latent(K);
    for (int k = 0; k < K; k++) {
      log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
    }

    // Extract sigma (exponentiated)
    latent_sigma.resize(K);
    for (int k = 0; k < K; k++) {
      latent_sigma[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factor scores (N x K, row-major)
    int n_factor_params = N * K;
    latent_factors_vec.resize(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
      latent_factors_vec[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint (sum-to-zero or first-zero)
    if (data.latent_constraint == 0) {  // sum_to_zero
      tulpa_latent::apply_sum_to_zero(latent_factors_vec, N, K);
    } else {  // first_zero
      tulpa_latent::apply_first_zero(latent_factors_vec, N, K);
    }

    // PC prior on sigma (exponential prior with Jacobian)
    log_post += tulpa_latent::latent_sigma_log_prior(log_sigma_latent,
                                                       data.latent_sigma_prior_rate);

    // Standard normal prior on factor scores
    tulpa_latent::LatentConstraint constraint =
      (data.latent_constraint == 0) ? tulpa_latent::LatentConstraint::SUM_TO_ZERO
                                    : tulpa_latent::LatentConstraint::FIRST_ZERO;
    log_post += tulpa_latent::latent_factor_log_prior(latent_factors_vec, N, K, constraint);
  }

  // Spatiotemporal interaction priors
  double tau_st = 1.0, tau_st2 = 1.0, rho_st = 0.0;
  double phi_st_space = 1.0, phi_st_time = 1.0;
  const double* st_delta = nullptr;

  if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
    // Extract precision parameter
    double log_tau_st = params[layout.log_tau_st_idx];
    tau_st = std::exp(log_tau_st);

    // PC prior on tau (exponential on sigma = 1/sqrt(tau))
    double sigma_st = 1.0 / std::sqrt(tau_st);
    double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
    log_post += std::log(lambda) - lambda * sigma_st - std::log(2.0 * sigma_st);
    log_post += log_tau_st;  // Jacobian for log transform

    // Single precision for all ST types (including Type IV)
    // tau_st2 always 1.0 ? removed non-identifiable second precision

    // AR1 rho parameter
    if (layout.logit_rho_st_idx >= 0) {
      double logit_rho_st = params[layout.logit_rho_st_idx];
      rho_st = 2.0 / (1.0 + std::exp(-logit_rho_st)) - 1.0;  // Map to (-1, 1)

      // Uniform(-1, 1) prior on rho
      // Jacobian for logit((rho+1)/2) transform
      double x = (rho_st + 1.0) / 2.0;
      log_post += std::log(x) + std::log(1.0 - x);
    }

    // GP range parameters
    if (layout.is_st_gp) {
      double log_phi_space = params[layout.log_phi_st_space_idx];
      double log_phi_time = params[layout.log_phi_st_time_idx];
      phi_st_space = std::exp(log_phi_space);
      phi_st_time = std::exp(log_phi_time);

      // Uniform prior within bounds
      if (phi_st_space < data.st_phi_space_prior_lower ||
          phi_st_space > data.st_phi_space_prior_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      if (phi_st_time < data.st_phi_time_prior_lower ||
          phi_st_time > data.st_phi_time_prior_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      log_post += log_phi_space + log_phi_time;  // Jacobians
    }

    // Spatiotemporal interaction effects
    // When precomputed_st_log_prior is provided (from gradient function), use it
    // to avoid recomputing the expensive Kronecker quadratic form (O(S*T^2) with
    // heap allocations). The precomputed value includes:
    //   - GMRF quadratic form (spatiotemporal_log_prior)
    //   - Rank/Jacobian terms
    //   - Sum-to-zero penalty
    if (precomputed_st_log_prior) {
      log_post += *precomputed_st_log_prior;
    } else {
      const double* z_or_delta = &params[layout.st_delta_start];
      int S = data.spatiotemporal_data.n_spatial;
      int T = data.spatiotemporal_data.n_times;
      int ST = S * T;

      // NC reparameterization for Type IV: store z, reconstruct delta
      const bool st_use_nc = (data.st_parameterization == 1 &&
                              data.spatiotemporal_data.type == STType::TYPE_IV);

      static thread_local std::vector<double> st_delta_nc;
      if (st_use_nc) {
        // Forward transform: delta = z / sqrt(tau_st)
        double inv_scale = 1.0 / std::sqrt(tau_st);
        st_delta_nc.resize(ST);
        for (int k = 0; k < ST; k++) {
          st_delta_nc[k] = z_or_delta[k] * inv_scale;
        }
        st_delta = st_delta_nc.data();

        // NC prior: -0.5 * z^T (Q_s ? Q_t) z  (tau-free GMRF)
        log_post += tulpa_spatiotemporal::spatiotemporal_log_prior(
          z_or_delta, 1.0, 1.0, rho_st, phi_st_space, phi_st_time,
          data.spatiotemporal_data
        );

        // Add the rank term with actual tau and Jacobian correction
        int rank_space = S - 1;
        int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T - 1) : (T - 2);
        if (data.spatiotemporal_data.temporal_cyclic) rank_time = T;
        int total_rank = rank_space * rank_time;
        // GMRF normalization: 0.5 * rank * log(tau)
        // NC Jacobian: -ST/2 * log(tau)
        // Combined: 0.5 * (rank - ST) * log(tau)
        log_post += 0.5 * (total_rank - ST) * std::log(tau_st);

        // Sum-to-zero on reconstructed delta
        log_post += tulpa_spatiotemporal::st_sum_to_zero_penalty(
          st_delta, S, T, 0.001, true, true
        );
      } else if (data.st_is_hsgp) {
        // HSGP-ST: spectral basis interaction (centered)
        // Each basis function j gets independent temporal GMRF with precision tau_st / S(lambda_j)
        st_delta = z_or_delta;
        int M = data.st_hsgp_data.m_total;

        // HSGP-ST hyperparameters
        double sigma2_st_hsgp = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        double lengthscale_st_hsgp = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);

        // PC prior on sigma_st_hsgp: rate=4.6
        double sigma_st = std::sqrt(sigma2_st_hsgp);
        log_post += -4.6 * sigma_st + 0.5 * params[layout.log_sigma2_st_hsgp_idx];

        // LogNormal(0,1) on lengthscale
        double log_ls_st = params[layout.log_lengthscale_st_hsgp_idx];
        log_post += -0.5 * log_ls_st * log_ls_st;

        // Per-basis-function temporal GMRF prior
        int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T - 1) :
                     (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        if (data.spatiotemporal_data.temporal_cyclic) rank_t = T;

        for (int j = 0; j < M; j++) {
          double S_j = tulpa_hsgp::spectral_density_se(
            data.st_hsgp_data.eigenvalues[j], sigma2_st_hsgp, lengthscale_st_hsgp);
          double prec_j = tau_st / std::max(S_j, 1e-10);

          // GMRF quadratic form: -0.5 * prec_j * delta_j' Q_t delta_j
          const double* dj = &st_delta[j * T];
          double qf = 0.0;
          if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
            for (int t = 1; t < T; t++) qf += (dj[t] - dj[t-1]) * (dj[t] - dj[t-1]);
          } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
            for (int t = 2; t < T; t++) {
              double d2 = dj[t] - 2.0*dj[t-1] + dj[t-2];
              qf += d2 * d2;
            }
          }
          log_post += 0.5 * rank_t * std::log(prec_j) - 0.5 * prec_j * qf;

          // Soft sum-to-zero per basis function
          double sum_j = 0.0;
          for (int t = 0; t < T; t++) sum_j += dj[t];
          log_post += -0.5 * 0.001 * sum_j * sum_j;
        }
      } else {
        // Centered parameterization (ICAR/BYM2 spatial)
        st_delta = z_or_delta;

        // Apply spatiotemporal log-prior from hmc_spatiotemporal.h
        log_post += tulpa_spatiotemporal::spatiotemporal_log_prior(
          st_delta, tau_st, tau_st2, rho_st, phi_st_space, phi_st_time,
          data.spatiotemporal_data
        );

        // Soft sum-to-zero constraint for identifiability
        log_post += tulpa_spatiotemporal::st_sum_to_zero_penalty(
          st_delta, S, T, 0.001, true, true
        );
      }
    }
  }

  // TVC (Temporally-Varying Coefficients) priors
  std::vector<double> tvc_tau;
  std::vector<double> tvc_rho;
  std::vector<double> tvc_w_flat;
  // tvc_eta lives in state (aliased above).

  if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;

    // Extract tau (precision) parameters
    tvc_tau.resize(n_tvc);
    for (int j = 0; j < n_tvc; j++) {
      double log_tau = params[layout.log_tau_tvc_start + j];
      tvc_tau[j] = std::exp(log_tau);

      // PC prior on tau (exponential prior on sigma = 1/sqrt(tau))
      // P(sigma > U) = alpha  =>  rate = -log(alpha) / U
      double sigma = 1.0 / std::sqrt(tvc_tau[j]);
      double rate = -std::log(0.01) / 1.0;  // P(sigma > 1) = 0.01
      log_post += std::log(rate) - rate * sigma - std::log(2.0 * sigma);
      log_post += log_tau;  // Jacobian for log transform
    }

    // Extract rho parameters for AR1 structure
    tvc_rho.resize(n_tvc, 0.0);
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1 &&
        layout.logit_rho_tvc_start >= 0) {
      for (int j = 0; j < n_tvc; j++) {
        double logit_rho = params[layout.logit_rho_tvc_start + j];
        tvc_rho[j] = 2.0 / (1.0 + std::exp(-logit_rho)) - 1.0;  // Map to (-1, 1)

        // Uniform(-1, 1) prior on rho
        // Jacobian for logit((rho+1)/2) transform
        double x = (tvc_rho[j] + 1.0) / 2.0;
        log_post += std::log(x) + std::log(1.0 - x);
      }
    }

    // Extract TVC values
    int n_tvc_params = n_groups * n_tvc * n_times;
    tvc_w_flat.resize(n_tvc_params);
    for (int k = 0; k < n_tvc_params; k++) {
      tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // TVC temporal prior
    log_post += tulpa_tvc::tvc_log_prior(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho);

    // Soft sum-to-zero constraint for identifiability
    log_post += tulpa_tvc::tvc_sum_to_zero_penalty(tvc_w_flat, data.tvc_data, 0.001);

    // Precompute TVC contribution to linear predictor
    tvc_eta.resize(data.N, 0.0);
    tulpa_tvc::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
  }

  // ============ SVC (Spatially-Varying Coefficients) ============
  double* svc_eta_ptr = nullptr;

  if (layout.has_svc && data.svc_data.n_svc > 0) {
    int n_svc = data.svc_data.n_svc;
    int n_obs = data.svc_data.n_obs;

    if (data.svc_is_hsgp) {
      // HSGP-based SVC: basis function approximation
      int m_total = data.svc_hsgp_data.m_total;

      svc_eta_ptr = data.svc_data.eta_ws.data();
      std::fill(svc_eta_ptr, svc_eta_ptr + data.N, 0.0);

      static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws_lp;
      hsgp_ws_lp.init(data.N, m_total);

      for (int j = 0; j < n_svc; j++) {
        double log_sigma2 = params[layout.log_sigma2_svc_start + j];
        double sigma2_j = std::exp(log_sigma2);
        double sigma_j = std::sqrt(sigma2_j);
        double rate_sigma = 4.6;
        log_post += -rate_sigma * sigma_j + 0.5 * log_sigma2;

        double log_ls = params[layout.log_phi_svc_start + j];
        log_post += -0.5 * log_ls * log_ls;  // LogNormal(0,1)

        double ls_j = std::exp(log_ls);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        // N(0,I) prior on beta
        for (int k = 0; k < m_total; k++) {
          log_post += -0.5 * beta_j[k] * beta_j[k];
        }

        // Evaluate f_j = Phi * (sqrt(S_j) * beta_j)
        tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, ls_j, data.svc_hsgp_data, hsgp_ws_lp);

        // Accumulate into svc_eta
        for (int i = 0; i < n_obs; i++) {
          svc_eta_ptr[i] += data.svc_data.X_svc[i * n_svc + j] * hsgp_ws_lp.hsgp_f[i];
        }
      }
    } else {
      // NNGP-based SVC (original path)
      double* svc_sigma2 = data.svc_data.sigma2_ws.data();
      double* svc_phi = data.svc_data.phi_ws.data();
      double* svc_w_flat = data.svc_data.w_flat_ws.data();

      for (int j = 0; j < n_svc; j++) {
        double log_sigma2 = params[layout.log_sigma2_svc_start + j];
        svc_sigma2[j] = std::exp(log_sigma2);
        double sigma = std::sqrt(svc_sigma2[j]);
        double scale = data.svc_sigma2_prior_scale;
        log_post += -std::log(1.0 + (sigma * sigma) / (scale * scale));
        log_post += log_sigma2;
      }

      for (int j = 0; j < n_svc; j++) {
        double log_phi = params[layout.log_phi_svc_start + j];
        svc_phi[j] = std::exp(log_phi);
        if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
          return -INFINITY;
        }
        log_post += log_phi;
      }

      int n_svc_params = n_svc * n_obs;
      for (int k = 0; k < n_svc_params; k++) {
        svc_w_flat[k] = params[layout.svc_w_start + k];
      }

      double* w_j = data.svc_data.w_j_ws.data();
      for (int j = 0; j < n_svc; j++) {
        for (int i = 0; i < n_obs; i++) {
          w_j[i] = svc_w_flat[j * n_obs + i];
        }
        std::vector<double> w_j_vec(w_j, w_j + n_obs);
        log_post += tulpa_svc::nngp_log_lik(w_j_vec, svc_sigma2[j], svc_phi[j], data.svc_data);
      }

      std::vector<double> svc_w_flat_vec(svc_w_flat, svc_w_flat + n_svc_params);
      log_post += tulpa_svc::svc_sum_to_zero_penalty(svc_w_flat_vec, data.svc_data, 1.0);

      svc_eta_ptr = data.svc_data.eta_ws.data();
      std::fill(svc_eta_ptr, svc_eta_ptr + data.N, 0.0);
      tulpa_svc::compute_svc_eta(svc_w_flat_vec, data.svc_data, data.svc_data.eta_ws);
    }
  }
