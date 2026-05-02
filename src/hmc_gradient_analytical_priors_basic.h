// hmc_gradient_analytical_priors_basic.h
// Function-body fragment of compute_gradient_analytical:
// random-effect priors (intercepts + slopes, both correlated and
// uncorrelated, both centered and non-centered parameterizations).
// NOT standalone-compilable: relies on locals declared inside
// compute_gradient_analytical (params, data, layout, grad, beta_num,
// beta_denom, sigma_re, tau_re, re_prior_grad_sigma,
// grad_re_slopes_lik, n_re_terms_slopes, slopes_nc, re_nc_flat,
// nc_L_flats, nc_sigmas_vec, ...). Included exactly once per umbrella
// so no header guards.

  // ============ Prior gradients (cheap) ============

  // Beta priors: N(0, sigma_beta^2)
  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());

  // sigma_re: Half-Cauchy prior with scale = data.sigma_re_scale (via log transform)
  // log_post = -log(1 + (sigma/scale)^2) + log(sigma) (Jacobian)
  // d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
  if (layout.has_re) {
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
  }

  // phi priors: Gamma(shape, rate) via log transform
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // RE prior: N(0, sigma_re^2)
  // d/d(re[g]) = -tau_re * re[g]
  // Also accumulates contribution to sigma_re gradient
  double re_prior_grad_sigma = 0.0;
  std::vector<std::vector<double>> grad_re_slopes_lik;  // [term][g*n_coefs+c] likelihood contributions
  int n_re_terms_slopes = 0;
  bool slopes_nc = (layout.has_re_slopes && data.re_parameterization == 1);

  // Non-centered slopes: pre-computed RE values for observation loop
  std::vector<double> re_nc_flat;
  // Per-term storage for non-centered chain rule in write-back
  std::vector<std::vector<double>> nc_L_flats;   // [term] -> L_flat
  std::vector<std::vector<double>> nc_sigmas_vec; // [term] -> sigmas

  if (layout.has_re && layout.has_re_slopes && !layout.has_re_correlated_slopes) {
    // ============ Uncorrelated random slopes prior gradients ============
    n_re_terms_slopes = data.n_re_terms;
    grad_re_slopes_lik.resize(n_re_terms_slopes);

    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters and compute priors
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        double log_sigma_c = params[log_sigma_idx];
        double sigma_c = std::exp(log_sigma_c);

        // Half-Cauchy prior on sigma_c
        double ratio_c = sigma_c / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;

        if (slopes_nc) {
          // Non-centered: params store z ~ N(0,1). Prior: -0.5*z^2
          // No sigma contribution from z prior (chain rule applied after obs loop)
          for (int g = 0; g < n_groups; g++) {
            double z_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -z_gc;
          }
        } else {
          // Centered: params store re ~ N(0, sigma_c^2)
          double tau_c = 1.0 / (sigma_c * sigma_c + 1e-10);
          double sigma_grad_c = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double re_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
          }
          grad[log_sigma_idx] += sigma_grad_c;
        }
      }
    }
  } else if (layout.has_re && layout.has_re_slopes && layout.has_re_correlated_slopes) {
    // ============ Correlated random slopes prior gradients ============
    // Multivariate normal with Sigma = diag(sigma) * L * L' * diag(sigma)
    // where L is lower-triangular Cholesky factor with L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
    // LKJ(eta=2) prior on correlation matrix
    n_re_terms_slopes = data.n_re_terms;
    grad_re_slopes_lik.resize(n_re_terms_slopes);

    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_correlated = layout.re_correlated_multi[t];
      grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        sigmas[c] = std::exp(params[log_sigma_idx]);

        // Half-Cauchy prior on sigma_c: d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
        double ratio_c = sigmas[c] / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;
      }

      if (is_correlated && n_coefs > 1) {
        // Build Cholesky factor L with tanh parameterization (must match
        // compute_log_post). LKJ(eta=2) prior + L -> R + tanh-Jacobian
        // gradient is written directly into grad[chol_start..] - slots are
        // zero-initialised at function entry.
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> L_flat(n_coefs * n_coefs, 0.0);
        if (!tulpa::build_L_from_raw(&params[chol_start], n_coefs, L_flat.data())) {
          return;  // Safety guard (shouldn't trigger with tanh)
        }
        tulpa::lkj_log_prior_grad_add(&params[chol_start], L_flat.data(), n_coefs,
                                      /*eta=*/2.0, &grad[chol_start]);

        // ---- Non-centered parameterization ----
        // Params store z ~ N(0,1). Compute re = diag(sigma) * L * z.
        if (re_nc_flat.empty()) {
          re_nc_flat.assign(params.size(), 0.0);
        }
        tulpa::compute_u_eff(L_flat.data(), n_coefs, sigmas.data(),
                              &params[re_start_t], n_groups,
                              &re_nc_flat[re_start_t]);

        // Save term data for write-back chain rule
        nc_L_flats.resize(n_re_terms_slopes);
        nc_sigmas_vec.resize(n_re_terms_slopes);
        nc_L_flats[t] = L_flat;
        nc_sigmas_vec[t] = sigmas;

        // Prior on z: N(0, I) -> grad[z_idx] = -z[g,c]
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
          }
        }
        // Sigma: Half-Cauchy prior already written above. No centered contribution.
        // In non-centered, the log-det of Jacobian (re = diag(sigma)*L*z)
        // cancels with the |Sigma|^{-1/2} normalization, so no -n_groups term.
      } else {
        // Uncorrelated term within a mixed model (e.g., intercept-only term
        // alongside correlated slopes terms in crossed+slopes)
        if (data.re_parameterization == 1) {
          // Non-centered: z ~ N(0,1), prior grad = -z
          for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
              grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
            }
          }
        } else {
          // Centered: re ~ N(0, sigma^2), prior grad = -tau*re
          for (int c = 0; c < n_coefs; c++) {
            double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
            double sigma_grad_c = 0.0;
            for (int g = 0; g < n_groups; g++) {
              double re_gc = params[re_start_t + g * n_coefs + c];
              grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
              sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
            }
            int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
            grad[log_sigma_idx] += sigma_grad_c;
          }
        }
      }
    }
  } else if (layout.has_re && !layout.has_re_slopes) {
    // Intercept-only RE (single or crossed terms)
    int n_terms = (data.n_re_terms > 1) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      // Get term-specific parameters
      int log_sigma_idx = (n_terms > 1) ? layout.log_sigma_re_multi[t] : layout.log_sigma_re_idx;
      int re_start_t = (n_terms > 1) ? layout.re_start_multi[t] : layout.re_start;
      int n_groups_t = (n_terms > 1) ? data.re_n_groups_multi[t] : data.n_re_groups;

      double log_sigma_t = params[log_sigma_idx];
      double sigma_t = std::exp(log_sigma_t);

      // Half-Cauchy prior on sigma_t (same for both parameterizations)
      double ratio_t = sigma_t / data.sigma_re_scale;
      double ratio_t_sq = ratio_t * ratio_t;
      grad[log_sigma_idx] = -2.0 * ratio_t_sq / (1.0 + ratio_t_sq) + 1.0;

      if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0, 1), prior grad = -z
        // No sigma contribution from z prior (sigma gradient comes from likelihood chain rule)
        for (int g = 0; g < n_groups_t; g++) {
          double z_g = params[re_start_t + g];
          grad[re_start_t + g] = -z_g;
        }
      } else {
        // Centered: re ~ N(0, sigma_t^2)
        double tau_t = 1.0 / (sigma_t * sigma_t + 1e-10);
        double sigma_grad_t = 0.0;
        for (int g = 0; g < n_groups_t; g++) {
          double re_g = params[re_start_t + g];
          grad[re_start_t + g] = -tau_t * re_g;
          sigma_grad_t += tau_t * re_g * re_g - 1.0;
        }
        grad[log_sigma_idx] += sigma_grad_t;
      }
    }
  }

