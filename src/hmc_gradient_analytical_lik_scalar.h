// hmc_gradient_analytical_lik_scalar.h
// Function-body fragment of compute_gradient_analytical: scalar
// fallback likelihood path (`if (!used_vectorized) { ... }`). NOT
// standalone-compilable; relies on the surrounding scope.

  if (!used_vectorized) {

  // Accumulators for beta gradients (will be added via X' * residual)
  std::vector<double> grad_beta_num(data.legacy.p_num, 0.0);
  std::vector<double> grad_beta_denom(data.legacy.p_denom, 0.0);
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  // Pre-compute per-term sigmas once (constant across i) for crossed
  // non-centered RE; const-shared across OpenMP threads.
  std::vector<double> sigma_re_terms_buf;
  const double* sigma_re_terms_ptr = nullptr;
  if (layout.has_re && data.n_re_terms > 1 && data.re_parameterization == 1) {
    sigma_re_terms_buf.resize(data.n_re_terms);
    for (int t = 0; t < data.n_re_terms; t++) {
      sigma_re_terms_buf[t] = std::exp(params[layout.log_sigma_re_multi[t]]);
    }
    sigma_re_terms_ptr = sigma_re_terms_buf.data();
  }

  #ifdef _OPENMP
  #pragma omp parallel
  {
    std::vector<double> local_grad_beta_num(data.legacy.p_num, 0.0);
    std::vector<double> local_grad_beta_denom(data.legacy.p_denom, 0.0);
    double local_grad_phi_num = 0.0;
    double local_grad_phi_denom = 0.0;
    std::vector<double> local_grad_re(layout.has_re ? data.n_re_groups : 0, 0.0);
    // Thread-local buffer for crossed RE gradients (all terms combined)
    std::vector<double> local_grad_re_crossed(
        (layout.has_re && data.n_re_terms > 1) ? data.total_re_groups : 0, 0.0);
    double local_obs_ll = 0.0;  // Fused log-likelihood accumulator
    // Pre-allocated per-obs group index buffer for crossed RE (reused across iterations)
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);

    #pragma omp for schedule(static)
    for (int i = 0; i < data.N; i++) {
  #else
    // Pre-allocated per-obs group index buffer for crossed RE
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);
    for (int i = 0; i < data.N; i++) {
  #endif
      // Compute linear predictors
      double eta_num = tulpa_linalg::dot_product(
          &data.legacy.X_num_flat[i * data.legacy.p_num], beta_num, data.legacy.p_num);
      double eta_denom = (data.legacy.p_denom > 0) ?
          tulpa_linalg::dot_product(
              &data.legacy.X_denom_flat[i * data.legacy.p_denom], beta_denom, data.legacy.p_denom) :
          0.0;

      // Add RE if present
      int re_idx = -1;
      int n_crossed_terms = 0;
      if (layout.has_re) {
        if (layout.has_re_slopes && n_re_terms_slopes > 0) {
          // Random slopes case: loop over ALL terms (supports crossed+slopes)
          // Each term can have intercept-only (n_coefs=1) or intercept+slopes (n_coefs>1)
          for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
            int group_idx = data.re_group_multi_flat[i * data.n_re_terms + t_re];
            if (group_idx <= 0) continue;
            int g = group_idx - 1;
            int n_coefs = layout.re_n_coefs_multi[t_re];
            int re_base = layout.re_start_multi[t_re] + g * n_coefs;

            bool is_corr_t = !re_nc_flat.empty() &&
                             t_re < (int)layout.re_correlated_multi.size() &&
                             layout.re_correlated_multi[t_re] && n_coefs > 1;
            bool is_uncorr_nc = !is_corr_t && slopes_nc;

            // Intercept contribution
            double re_contrib;
            if (is_corr_t) {
              re_contrib = re_nc_flat[re_base];
            } else if (is_uncorr_nc) {
              double sigma_int = std::exp(params[layout.log_sigma_re_slopes[t_re][0]]);
              re_contrib = sigma_int * params[re_base];
            } else {
              re_contrib = params[re_base];
            }

            // Slope contributions (only for terms with slopes)
            int n_slopes = n_coefs - 1;
            if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() &&
                !data.re_slope_matrices[t_re].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[t_re][i * n_slopes + s];
                double re_slope;
                if (is_corr_t) {
                  re_slope = re_nc_flat[re_base + 1 + s];
                } else if (is_uncorr_nc) {
                  double sigma_s = std::exp(params[layout.log_sigma_re_slopes[t_re][1 + s]]);
                  re_slope = sigma_s * params[re_base + 1 + s];
                } else {
                  re_slope = params[re_base + 1 + s];
                }
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            if (data.legacy.model_type != ModelType::BINOMIAL) {
              eta_denom += re_contrib;
            }
          }
        } else if (data.n_re_terms > 1) {
          // Crossed RE: shared per-i kernel (also used by expand_re_crossed).
          // Per-term sigmas were pre-computed once at function scope.
          n_crossed_terms = data.n_re_terms;
          double re_total = re_crossed_contribution(
              i, data, layout, params.data(),
              sigma_re_terms_ptr, re_idx_multi_buf.data());
          eta_num += re_total;
          if (data.legacy.model_type != ModelType::BINOMIAL) {
            eta_denom += re_total;
          }
        } else if (data.re_group[i] > 0) {
          // Single intercept-only RE: shared per-i kernel.
          re_idx = data.re_group[i] - 1;
          double sigma_re_nc = (data.re_parameterization == 1) ? sigma_re : 0.0;
          double re_val = re_single_contribution(i, data, layout, params.data(), sigma_re_nc);
          eta_num += re_val;
          if (data.legacy.model_type != ModelType::BINOMIAL) {
            eta_denom += re_val;
          }
        }
      }
      // Add temporal effect if present
      int t_idx = -1;
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        double temporal_effect = temporal_contribution(i, data, phi_temporal, &t_idx);
        eta_num += temporal_effect;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          eta_denom += temporal_effect;
        }
      }

      // Add spatial effect if present
      int s_idx = -1;
      double d_spatial_d_phi = 0.0;  // Derivative of spatial_effect wrt phi_spatial
      double d_spatial_d_theta = 0.0;  // Derivative of spatial_effect wrt theta_bym2
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        double spatial_effect;
        if (data.spatial_type == SpatialType::BYM2) {
          spatial_effect = spatial_bym2_contribution(
              i, data, phi_spatial, theta_bym2,
              sigma_s_bym2, sigma_u_bym2,
              &s_idx, &d_spatial_d_phi, &d_spatial_d_theta);
        } else {
          spatial_effect = spatial_icar_contribution(i, data, phi_spatial, &s_idx);
          d_spatial_d_phi = 1.0;
        }
        eta_num += spatial_effect;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          eta_denom += spatial_effect;
        }
      }

      double resid_num = 0.0;
      double resid_denom = 0.0;
      double grad_phi_num_i = 0.0;
      double grad_phi_denom_i = 0.0;
      double grad_logit_zi_i = 0.0;
      double grad_logit_oi_i = 0.0;

      // Compute ZI linear predictor if applicable
      double logit_zi = 0.0;
      double zi_prob = 0.0;
      if (layout.has_zi && data.p_zi > 0) {
        logit_zi = tulpa_linalg::dot_product(
            &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
        zi_prob = 1.0 / (1.0 + std::exp(-logit_zi));
      }

      // Compute OI linear predictor if applicable
      double logit_oi = 0.0;
      double oi_prob = 0.0;
      if (layout.has_oi && data.p_oi > 0) {
        logit_oi = tulpa_linalg::dot_product(
            &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
        oi_prob = 1.0 / (1.0 + std::exp(-logit_oi));
      }

      if (data.legacy.model_type == ModelType::BINOMIAL) {
        // ---- BINOMIAL ----
        // p = inv_logit(eta_num), LL = y*log(p) + (n-y)*log(1-p)
        // d(LL)/d(eta) = y - n*p
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int n_trials = data.legacy.y_denom[i];
        int y_num_i = data.legacy.y_num[i];

        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
          // ZI-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*(1-p)^n
            double p0_binom = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double p0 = zi_prob + (1.0 - zi_prob) * p0_binom;
            // d(LL)/d(eta) = (1-zi) * d((1-p)^n)/d(eta) / p0
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*p*(1-p)^n
            resid_num = -(1.0 - zi_prob) * n_trials * p * p0_binom / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_binom) / p0;
          } else {
            // P(Y=y) = (1-zi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
          }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
          // Hurdle-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta
            resid_num = 0.0;  // No p contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncBinomial(y|n,p)
            double p0_binom = std::pow(1.0 - p, n_trials);
            double normalizer = 1.0 - p0_binom;
            if (normalizer < 1e-12) normalizer = 1e-12;
            // Gradient from truncated binomial
            // d(log(1-(1-p)^n))/d(eta) = n*p*(1-p)^n / (1-(1-p)^n)
            double grad_normalizer = n_trials * p * p0_binom / normalizer;
            resid_num = (y_num_i - n_trials * p) - grad_normalizer;
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
          }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
          // OI-Binomial (One-inflation only)
          // P(Y=n) = oi + (1-oi) * p^n
          // P(Y=y, y<n) = (1-oi) * Binomial(y|n,p)
          if (y_num_i == n_trials) {
            // y = n (structural one or binomial one)
            double pn = std::pow(p, n_trials);  // p^n
            double P_yn = oi_prob + (1.0 - oi_prob) * pn;
            if (P_yn < 1e-12) P_yn = 1e-12;
            // d(log P)/d(eta) = (1-oi) * n * p^(n-1) * p*(1-p) / P
            //                 = (1-oi) * n * p^n * (1-p) / P
            resid_num = (1.0 - oi_prob) * n_trials * pn * (1.0 - p) / P_yn;
            // d(log P)/d(logit_oi) = oi*(1-oi)*(1 - p^n) / P
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / P_yn;
          } else {
            // y < n: P(Y=y) = (1-oi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::ZOIB) {
          // ZOIB (Zero-One Inflated Binomial) - MIXTURE MODEL
          // P(Y=0) = zi + (1-zi)*(1-oi)*(1-p)^n
          // P(Y=n) = (1-zi)*(oi + (1-oi)*p^n)
          // P(Y=y, 0<y<n) = (1-zi)*(1-oi)*Binomial(y|n,p)
          // Note: zi_prob = zi, oi_prob = oi
          if (y_num_i == 0) {
            // y = 0: P = zi + (1-zi)*(1-oi)*(1-p)^n = A + B
            double binom_zero = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double A = zi_prob;  // structural zero component
            double B = (1.0 - zi_prob) * (1.0 - oi_prob) * binom_zero;  // binomial zero
            double P = A + B;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d((1-p)^n)/d(eta) / P
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*(1-p)^n * p
            double d_binom_d_eta = -n_trials * binom_zero * p;
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_binom_d_eta / P;

            // d(log P)/d(logit_zi) = [dA - B] * zi*(1-zi) / P
            // dA/d(logit_zi) = zi*(1-zi), dB/d(logit_zi) = -(1-oi)*binom_zero * zi*(1-zi)
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - (1.0 - oi_prob) * binom_zero) / P;

            // d(log P)/d(logit_oi) = dB/d(logit_oi) / P
            // dB/d(logit_oi) = -(1-zi)*binom_zero * oi*(1-oi)
            grad_logit_oi_i = -(1.0 - zi_prob) * binom_zero * oi_prob * (1.0 - oi_prob) / P;

          } else if (y_num_i == n_trials) {
            // y = n: P = (1-zi)*(oi + (1-oi)*p^n) = (1-zi)*C
            double pn = std::pow(p, n_trials);  // p^n
            double C = oi_prob + (1.0 - oi_prob) * pn;  // oi + (1-oi)*p^n
            double P = (1.0 - zi_prob) * C;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d(p^n)/d(eta) / P
            // d(p^n)/d(eta) = n*p^(n-1) * p*(1-p) = n*p^n*(1-p)
            double d_pn_d_eta = n_trials * pn * (1.0 - p);
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_pn_d_eta / P;

            // d(log P)/d(logit_zi) = d(log(1-zi))/d(logit_zi) = -zi
            grad_logit_zi_i = -zi_prob;

            // d(log P)/d(logit_oi) = dC/d(logit_oi) / C
            // dC/d(logit_oi) = oi*(1-oi) - p^n * oi*(1-oi) = oi*(1-oi)*(1 - p^n)
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / C;

          } else {
            // 0 < y < n: P = (1-zi)*(1-oi)*Binomial
            // log P = log(1-zi) + log(1-oi) + log_binom
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else {
          // Standard binomial (no ZI)
          resid_num = y_num_i - n_trials * p;
        }
        // No denominator contribution for binomial

      } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        // ---- NEGBIN_NEGBIN ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];
        int y_denom_i = data.legacy.y_denom[i];

        // Denominator NegBin gradient (always standard, not ZI)
        double denom_d = mu_denom + phi_denom;
        resid_denom = y_denom_i - mu_denom * (y_denom_i + phi_denom) / denom_d;
        grad_phi_denom_i = tulpa::math::portable_digamma(y_denom_i + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                           + std::log(phi_denom / denom_d)
                           + (mu_denom - y_denom_i) / denom_d;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*p0_nb
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(LL)/d(mu) = (1-zi) * d(p0_nb)/d(mu) / p0
            // d(p0_nb)/d(mu) = phi * (phi/(phi+mu))^phi * (-1/(phi+mu)) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            // phi gradient for ZI-NegBin at y=0 (complex, using approximation)
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            // P(Y=y) = (1-zi) * NB(y|mu,phi)
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta, where theta = sigmoid(logit_zi) here represents P(Y>0)
            // Note: for hurdle, logit_zi parameterizes theta = P(Y>0), so zi_prob IS theta
            resid_num = 0.0;  // No mu contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
            grad_phi_num_i = 0.0;
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncNB(y|mu,phi)
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double log_normalizer = std::log(1.0 - p0_nb);
            double denom_num = mu_num + phi_num;
            // Gradient from truncated NB: NB residual MINUS normalizer correction
            // LL = log NB(y) - log(1-p0), so d(LL)/d(eta) = NB_resid - d(log(1-p0))/d(mu)*mu
            // d(log(1-p0))/d(mu) = phi*p0 / ((phi+mu)*(1-p0))
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
        } else {
          // Standard NegBin (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
        }

      } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        // ---- POISSON_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) - skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate = phi_denom / mu_denom;
          grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                  - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_POISSON) {
          // ZI-Poisson numerator
          double exp_neg_mu = std::exp(-mu_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*exp(-mu)
            double p0 = zi_prob + (1.0 - zi_prob) * exp_neg_mu;
            // d(LL)/d(eta) = d(LL)/d(mu) * mu = -(1-zi)*exp(-mu)*mu / p0
            resid_num = -(1.0 - zi_prob) * exp_neg_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - exp_neg_mu) / p0;
            grad_phi_denom_i = grad_phi_gamma;  // Only gamma part
          } else {
            // P(Y=y) = (1-zi) * Poisson(y|mu)
            resid_num = y_num_i - mu_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_denom_i = grad_phi_gamma;
          }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_POISSON) {
          // Hurdle-Poisson numerator
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // zi_prob is theta here
            grad_phi_denom_i = grad_phi_gamma;
          } else {
            // Truncated Poisson: d(LL)/d(eta) = y - mu - mu*exp(-mu)/(1-exp(-mu))
            // LL = log Poi(y) - log(1-exp(-mu)), correction is SUBTRACTED
            double exp_neg_mu = std::exp(-mu_num);
            resid_num = y_num_i - mu_num - mu_num * exp_neg_mu / (1.0 - exp_neg_mu);
            grad_logit_zi_i = 1.0 - zi_prob;
            grad_phi_denom_i = grad_phi_gamma;
          }
        } else {
          // Standard Poisson (no ZI)
          resid_num = y_num_i - mu_num;
          grad_phi_denom_i = grad_phi_gamma;
        }

      } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        // ---- NEGBIN_GAMMA ----
        // NegBin numerator + Gamma denominator
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) - skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate = phi_denom / mu_denom;
          grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                  - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling (same as NEGBIN_NEGBIN)
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator (same logic as NEGBIN_NEGBIN ZI case)
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(p0_nb)/d(mu) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
          grad_phi_denom_i = grad_phi_gamma;
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator (same as NEGBIN_NEGBIN hurdle)
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta)
            grad_phi_num_i = 0.0;
          } else {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta)
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
          grad_phi_denom_i = grad_phi_gamma;
        } else {
          // Standard NegBin numerator (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
          grad_phi_denom_i = grad_phi_gamma;
        }

      } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
        // ---- GAMMA_GAMMA ----
        // Gamma requires y > 0; skip contributions for y <= 0 (matches log_lik_gamma)
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];

        if (y_num_i > 0.0) {
          resid_num = phi_num * (y_num_i / mu_num - 1.0);
          double rate_num = phi_num / mu_num;
          grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                           - tulpa::math::portable_digamma(phi_num) - y_num_i / mu_num;
        }
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate_denom = phi_denom / mu_denom;
          grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                             - tulpa::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;
        }

      } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
        // ---- LOGNORMAL ----
        // Both numerator and denominator are Lognormal distributed
        // log(y) ~ Normal(mu, sigma^2), so y ~ Lognormal(mu, sigma^2)
        // LL = -log(y) - log(sigma) - 0.5*((log(y) - mu)/sigma)^2
        // d(LL)/d(eta) = d(LL)/d(mu) = (log(y) - mu) / sigma^2
        // (Note: eta IS mu for lognormal, so no chain rule needed)
        double mu_num = eta_num;  // mu is directly the linear predictor
        double mu_denom = eta_denom;
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];
        double log_y_num = std::log(y_num_i);
        double log_y_denom = std::log(y_denom_i);

        // phi_num, phi_denom are sigma (std dev on log scale)
        double sigma_num = phi_num;
        double sigma_denom = phi_denom;
        double sigma_num_sq = sigma_num * sigma_num;
        double sigma_denom_sq = sigma_denom * sigma_denom;

        // Residuals (gradient w.r.t. mu)
        resid_num = (log_y_num - mu_num) / sigma_num_sq;
        resid_denom = (log_y_denom - mu_denom) / sigma_denom_sq;

        // Sigma gradients: d(LL)/d(sigma), NOT d(LL)/d(log_sigma)
        // Because the accumulation code multiplies by phi to get d(LL)/d(log_phi)
        // d(LL)/d(sigma) = -1/sigma + z^2/sigma = (-1 + z^2) / sigma
        double z_num = (log_y_num - mu_num) / sigma_num;
        double z_denom = (log_y_denom - mu_denom) / sigma_denom;
        grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
        grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

      } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
        // ---- BETA_BINOMIAL ----
        // Overdispersed binomial: y ~ BetaBinom(n, alpha, beta)
        // where p = alpha/(alpha+beta), phi = alpha + beta (concentration)
        // We parameterize: logit(p) = eta, phi = overdispersion
        // alpha = p * phi, beta_param = (1-p) * phi
        // LL = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+phi)
        //      - lgamma(alpha) - lgamma(beta) + lgamma(phi) + lchoose(n,y)
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y_i = data.legacy.y_num[i];
        int n_i = data.legacy.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;

        // d(LL)/d(eta) = d(LL)/d(p) * d(p)/d(eta) where d(p)/d(eta) = p*(1-p)
        // d(LL)/d(p) = phi * (digamma(y+alpha) - digamma(n-y+beta) - digamma(alpha) + digamma(beta))
        double psi_y_alpha = tulpa::math::portable_digamma(y_i + alpha);
        double psi_nmy_beta = tulpa::math::portable_digamma(n_i - y_i + beta_param);
        double psi_alpha = tulpa::math::portable_digamma(alpha);
        double psi_beta = tulpa::math::portable_digamma(beta_param);
        double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        resid_num = dLL_dp * p * (1.0 - p);

        // d(LL)/d(phi) where phi = alpha + beta
        // d(LL)/d(phi) = p*digamma(y+alpha) + (1-p)*digamma(n-y+beta) - digamma(n+phi)
        //               - p*digamma(alpha) - (1-p)*digamma(beta) + digamma(phi)
        double psi_n_phi = tulpa::math::portable_digamma(n_i + phi_num);
        double psi_phi = tulpa::math::portable_digamma(phi_num);
        grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;

        // No denominator contribution for beta-binomial (like binomial)
      }

      // Accumulate ZI coefficient gradients
      if (layout.has_zi && data.p_zi > 0) {
        for (int j = 0; j < data.p_zi; j++) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad_beta_zi[j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
          #else
          grad_beta_zi[j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
          #endif
        }
      }

      // Accumulate OI coefficient gradients
      if (layout.has_oi && data.p_oi > 0) {
        for (int j = 0; j < data.p_oi; j++) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad_beta_oi[j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
          #else
          grad_beta_oi[j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
          #endif
        }
      }

      // Accumulate beta gradients: grad += X[i,:] * resid
      for (int j = 0; j < data.legacy.p_num; j++) {
        #ifdef _OPENMP
        local_grad_beta_num[j] += data.legacy.X_num_flat[i * data.legacy.p_num + j] * resid_num;
        #else
        grad_beta_num[j] += data.legacy.X_num_flat[i * data.legacy.p_num + j] * resid_num;
        #endif
      }
      // For BINOMIAL, beta_denom doesn't affect likelihood
      if (data.legacy.model_type != ModelType::BINOMIAL) {
        for (int j = 0; j < data.legacy.p_denom; j++) {
          #ifdef _OPENMP
          local_grad_beta_denom[j] += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * resid_denom;
          #else
          grad_beta_denom[j] += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * resid_denom;
          #endif
        }
      }

      // Accumulate RE gradient
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        // Random slopes case: scatter to ALL terms (supports crossed+slopes)
        double re_grad_base = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          re_grad_base += resid_denom;
        }
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
          int group_idx = data.re_group_multi_flat[i * data.n_re_terms + t_re];
          if (group_idx <= 0) continue;
          int g = group_idx - 1;
          int n_coefs = layout.re_n_coefs_multi[t_re];

          // Intercept gradient
          #ifdef _OPENMP
          #pragma omp atomic
          grad_re_slopes_lik[t_re][g * n_coefs] += re_grad_base;
          #else
          grad_re_slopes_lik[t_re][g * n_coefs] += re_grad_base;
          #endif

          // Slope gradients: multiply by slope design value
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() &&
              !data.re_slope_matrices[t_re].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[t_re][i * n_slopes + s];
              double slope_grad = re_grad_base * x_slope;
              #ifdef _OPENMP
              #pragma omp atomic
              grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += slope_grad;
              #else
              grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += slope_grad;
              #endif
            }
          }
        }
      } else if (n_crossed_terms > 0) {
        // Crossed RE (multiple intercept-only terms)
        double re_grad_i = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;
        }
        for (int t = 0; t < n_crossed_terms; t++) {
          if (re_idx_multi_buf[t] >= 0) {
            #ifdef _OPENMP
            // Thread-local accumulation (reduced at end of parallel block)
            local_grad_re_crossed[data.re_offsets[t] + re_idx_multi_buf[t]] += re_grad_i;
            #else
            grad[layout.re_start_multi[t] + re_idx_multi_buf[t]] += re_grad_i;
            #endif
          }
        }
      } else if (re_idx >= 0) {
        // Simple intercept-only RE (single term)
        double re_grad_i = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;  // Shared RE affects both processes
        }
        #ifdef _OPENMP
        local_grad_re[re_idx] += re_grad_i;
        #else
        grad[layout.re_start + re_idx] += re_grad_i;
        #endif
      }

      // Accumulate temporal gradient (from likelihood)
      if (t_idx >= 0) {
        double temp_grad_i = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          temp_grad_i += resid_denom;
        }
        #ifdef _OPENMP
        #pragma omp atomic
        grad_temporal_lik[t_idx] += temp_grad_i;
        #else
        grad_temporal_lik[t_idx] += temp_grad_i;
        #endif
      }

      // Accumulate spatial gradient (from likelihood)
      if (s_idx >= 0) {
        double lik_grad = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          lik_grad += resid_denom;
        }
        // For ICAR: grad_spatial[s] += lik_grad (d_spatial_d_phi = 1)
        // For BYM2: grad_phi[s] += lik_grad * d_spatial_d_phi, grad_theta[s] += lik_grad * d_spatial_d_theta
        #ifdef _OPENMP
        #pragma omp atomic
        grad_spatial_lik[s_idx] += lik_grad * d_spatial_d_phi;
        #else
        grad_spatial_lik[s_idx] += lik_grad * d_spatial_d_phi;
        #endif

        if (data.spatial_type == SpatialType::BYM2) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad[layout.theta_bym2_start + s_idx] += lik_grad * d_spatial_d_theta;
          #else
          grad[layout.theta_bym2_start + s_idx] += lik_grad * d_spatial_d_theta;
          #endif
        }
      }

      // Accumulate phi gradients
      #ifdef _OPENMP
      local_grad_phi_num += grad_phi_num_i;
      local_grad_phi_denom += grad_phi_denom_i;
      #else
      grad_phi_num_lik += grad_phi_num_i;
      grad_phi_denom_lik += grad_phi_denom_i;
      #endif

      // Fused log-likelihood: compute per-observation log_lik using already-computed
      // intermediates. This avoids a separate O(N) pass through compute_log_post.
      if (compute_lp) {
        double ll_i = 0.0;
        if (data.legacy.model_type == ModelType::BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int n_trials = data.legacy.y_denom[i];
          int y_i = data.legacy.y_num[i];
          if (data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
            ll_i = tulpa_zi::zi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
            ll_i = tulpa_zi::hurdle_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
            ll_i = tulpa_zi::oi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_oi);
          } else if (data.zi_type == tulpa_zi::ZIType::ZOIB) {
            ll_i = tulpa_zi::zoib_lpmf_logit(y_i, n_trials, p_i, logit_zi, logit_oi);
          } else {
            ll_i = log_lik_binomial(y_i, n_trials, eta_num);
          }
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_negbin(data.legacy.y_denom[i], mu_denom_i, phi_denom);
        } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_poisson(data.legacy.y_num[i], mu_num_i);
          }
          ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          ll_i = log_lik_gamma(data.legacy.y_num_cont[i], phi_num, mu_num_i);
          ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
          double log_y_num_i = std::log(data.legacy.y_num_cont[i]);
          double log_y_denom_i = std::log(data.legacy.y_denom_cont[i]);
          double z_num_i = (log_y_num_i - eta_num) / phi_num;
          double z_denom_i = (log_y_denom_i - eta_denom) / phi_denom;
          ll_i = -log_y_num_i - std::log(phi_num) - 0.5 * z_num_i * z_num_i;
          ll_i += -log_y_denom_i - std::log(phi_denom) - 0.5 * z_denom_i * z_denom_i;
        } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int y_i = data.legacy.y_num[i];
          int n_i = data.legacy.y_denom[i];
          double alpha_i = p_i * phi_num;
          double beta_i = (1.0 - p_i) * phi_num;
          ll_i = std::lgamma(y_i + alpha_i) + std::lgamma(n_i - y_i + beta_i) - std::lgamma(n_i + phi_num);
          ll_i += -std::lgamma(alpha_i) - std::lgamma(beta_i) + std::lgamma(phi_num);
          ll_i += tulpa::math::portable_lchoose(n_i, y_i);
        }
        #ifdef _OPENMP
        local_obs_ll += ll_i;
        #else
        obs_log_lik += ll_i;
        #endif
      }
    }

  #ifdef _OPENMP
    // Reduce local accumulators
    #pragma omp critical
    {
      for (int j = 0; j < data.legacy.p_num; j++) {
        grad_beta_num[j] += local_grad_beta_num[j];
      }
      for (int j = 0; j < data.legacy.p_denom; j++) {
        grad_beta_denom[j] += local_grad_beta_denom[j];
      }
      grad_phi_num_lik += local_grad_phi_num;
      grad_phi_denom_lik += local_grad_phi_denom;
      for (int g = 0; g < (int)local_grad_re.size(); g++) {
        grad[layout.re_start + g] += local_grad_re[g];
      }
      // Reduce crossed RE thread-local buffers to scattered grad positions
      // Only for crossed intercept-only RE (n_re_terms > 1, not slopes)
      if (!local_grad_re_crossed.empty()) {
        for (int t = 0; t < (int)data.re_n_groups_multi.size(); t++) {
          int re_start_t = layout.re_start_multi[t];
          int offset_t = data.re_offsets[t];
          for (int g = 0; g < data.re_n_groups_multi[t]; g++) {
            grad[re_start_t + g] += local_grad_re_crossed[offset_t + g];
          }
        }
      }
      obs_log_lik += local_obs_ll;
    }
  }  // end parallel
  #endif

  // Add likelihood gradients to total
  for (int j = 0; j < data.legacy.p_num; j++) {
    grad[layout.legacy.beta_num_start + j] += grad_beta_num[j];
  }
  for (int j = 0; j < data.legacy.p_denom; j++) {
    grad[layout.legacy.beta_denom_start + j] += grad_beta_denom[j];
  }

  // Phi gradients (with Jacobian for log transform)
  if (layout.legacy.has_phi_num) {
    grad[layout.legacy.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.legacy.has_phi_denom) {
    grad[layout.legacy.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // ZI coefficient gradients (likelihood contribution)
  if (layout.has_zi && data.p_zi > 0) {
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] += grad_beta_zi[j];
    }
  }

  // OI coefficient gradients (likelihood contribution)
  if (layout.has_oi && data.p_oi > 0) {
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] += grad_beta_oi[j];
    }
  }

  } // end if (!used_vectorized)

  // Random slopes likelihood gradients (MUST be outside used_vectorized check:
  // the hybrid slopes path sets used_vectorized=true but populates grad_re_slopes_lik
  // via its own scatter pass - the write-back chain rule runs for all paths)
  if (layout.has_re_slopes && n_re_terms_slopes > 0) {
    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_nc = (t < (int)nc_L_flats.size() && !nc_L_flats[t].empty());

      if (is_nc) {
        // Non-centered correlated slopes: chain rule from dLL/d(re_nc) back
        // to (z, log_sigma, raw_chol). Single helper covers all three pieces;
        // grad_z and grad_raw slots are contiguous, log_sigma is scattered.
        const auto& L_flat = nc_L_flats[t];
        const auto& sigmas = nc_sigmas_vec[t];
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> g_log_sigma(n_coefs, 0.0);

        tulpa::chol_nc_chain_rule_add(
            L_flat.data(), n_coefs, sigmas.data(),
            &params[re_start_t], &params[chol_start],
            &re_nc_flat[re_start_t], n_groups,
            grad_re_slopes_lik[t].data(),
            &grad[re_start_t],         // grad_z (contiguous)
            g_log_sigma.data(),         // grad_log_sigma (scattered into temp)
            &grad[chol_start]);         // grad_raw (contiguous)

        for (int c = 0; c < n_coefs; c++) {
          grad[layout.log_sigma_re_slopes[t][c]] += g_log_sigma[c];
        }

      } else if (data.re_parameterization == 1) {
        // Uncorrelated non-centered: apply chain rule re = sigma * z
        // grad_re_slopes_lik[t][g*nc+c] = dLL/d(re[g,c])
        // grad[z_gc] += dLL/d(re_gc) * sigma_c
        // grad[log_sigma_c] += dLL/d(re_gc) * z_gc * sigma_c
        for (int c = 0; c < n_coefs; c++) {
          double sigma_c = std::exp(params[layout.log_sigma_re_slopes[t][c]]);
          double sigma_lik_grad = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double lik_gc = grad_re_slopes_lik[t][g * n_coefs + c];
            double z_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] += lik_gc * sigma_c;
            sigma_lik_grad += lik_gc * z_gc * sigma_c;
          }
          grad[layout.log_sigma_re_slopes[t][c]] += sigma_lik_grad;
        }
      } else {
        // Uncorrelated centered: add grad_re_slopes_lik directly
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] += grad_re_slopes_lik[t][g * n_coefs + c];
          }
        }
      }
    }
  }
