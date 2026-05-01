// =====================================================================
// RE gradient helpers for specialized gradient functions
// Handles both centered and non-centered parameterizations correctly
// =====================================================================

// Initialize RE gradient with prior contribution
static inline void re_gradient_prior(
    const ModelData& data,
    const ParamLayout& layout,
    const double* re,   // re[g] = params[re_start + g]
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0) return;

    // Half-Cauchy prior on sigma_re (log-scale)
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

    if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0,1), prior grad = -z
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -re[g];
        }
    } else {
        // Centered: re ~ N(0, sigma^2)
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }
}

// Get RE value for observation (handles NC -> sigma*z transformation)
static inline double re_value_for_eta(
    const double* re,
    int g,
    double sigma_re,
    int re_parameterization
) {
    double val = re[g];
    if (re_parameterization == 1) val *= sigma_re;
    return val;
}

// Apply NC chain rule transformation after observation loop
// Must be called AFTER observation loop has accumulated likelihood gradients in grad[re+g]
static inline void re_gradient_nc_transform(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0 || data.re_parameterization != 1) return;

    double sigma_lik_grad = 0.0;
    for (int g = 0; g < data.n_re_groups; g++) {
        double z_g = params[layout.re_start + g];
        // Extract centered lik grad: total - prior = grad[re+g] - (-z_g) = grad[re+g] + z_g
        double centered_lik = grad[layout.re_start + g] + z_g;
        // z gradient = prior + chain rule through sigma*z
        grad[layout.re_start + g] = -z_g + sigma_re * centered_lik;
        // sigma gradient from likelihood: z_g * d_ll/d_re_g
        sigma_lik_grad += z_g * centered_lik;
    }
    // d_ll/d_log_sigma = sigma * sum(z_g * d_ll/d_re_g)
    grad[layout.log_sigma_re_idx] += sigma_re * sigma_lik_grad;
}

// =====================================================================
// Shared gradient building blocks for specialized H-mode functions
// These helpers extract the duplicated code from 11 specialized gradient
// functions into single-source-of-truth implementations.
// All are static inline - zero overhead, compiler inlines them.
// =====================================================================

// Common parameters extracted from the parameter vector
struct CommonGradParams {
    const double* beta_num;
    const double* beta_denom;
    double sigma_re;
    const double* re;
    double phi_num;
    double phi_denom;
};

// Extract common parameters from the HMC parameter vector
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params,
    const ParamLayout& layout
) {
    CommonGradParams cp;
    cp.beta_num = &params[layout.legacy.beta_num_start];
    cp.beta_denom = &params[layout.legacy.beta_denom_start];
    cp.sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    cp.re = layout.has_re ? &params[layout.re_start] : nullptr;
    cp.phi_num = layout.legacy.has_phi_num ? std::exp(params[layout.legacy.log_phi_num_idx]) : 1.0;
    cp.phi_denom = layout.legacy.has_phi_denom ? std::exp(params[layout.legacy.log_phi_denom_idx]) : 1.0;
    return cp;
}

// Beta N(0, sigma_beta^2) prior gradient
// d/d(beta) = -tau_beta * beta where tau_beta = 1/sigma_beta^2
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom,
    double* grad
) {
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.legacy.p_num; j++) {
        grad[layout.legacy.beta_num_start + j] = -tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        grad[layout.legacy.beta_denom_start + j] = -tau_beta * beta_denom[j];
    }
}

// Phi Gamma(shape, rate) prior gradient on log-scale
// d/d(log_phi) = shape - rate*phi
// (equivalently: (shape-1) - rate*phi + 1 with Jacobian expanded)
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom,
    double* grad
) {
    if (layout.legacy.has_phi_num) {
        grad[layout.legacy.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.legacy.has_phi_denom) {
        grad[layout.legacy.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }
}

// Per-observation residual computation (dLL/deta for each family)
// Handles all model types: BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA,
// NEGBIN_GAMMA, and catch-all (GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL)
static inline void compute_obs_residuals(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double& dLL_deta_num, double& dLL_deta_denom
) {
    dLL_deta_num = 0.0;
    dLL_deta_denom = 0.0;

    if (data.legacy.model_type == ModelType::BINOMIAL) {
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        dLL_deta_num = data.legacy.y_num[i] - data.legacy.y_denom[i] * p;
    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / (mu_num + phi_num);
        dLL_deta_denom = data.legacy.y_denom[i] - mu_denom * (data.legacy.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
    } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = data.legacy.y_num[i] - mu_num;
        // Gamma requires y > 0; skip if y_denom_cont <= 0 (matches log_lik_gamma)
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double denom_nb = mu_num + phi_num;
        dLL_deta_num = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / denom_nb;
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
        // Gamma: dLL/d(eta) = shape * (y/mu - 1) where mu = exp(eta)
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = (data.legacy.y_num_cont[i] > 0.0)
            ? phi_num * (data.legacy.y_num_cont[i] / mu_num - 1.0) : 0.0;
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
        // Lognormal: dLL/d(eta) = (log(y) - eta) / sigma^2
        double sigma_num = phi_num, sigma_denom = phi_denom;
        dLL_deta_num = (data.legacy.y_num_cont[i] > 0.0)
            ? (std::log(data.legacy.y_num_cont[i]) - eta_num) / (sigma_num * sigma_num) : 0.0;
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? (std::log(data.legacy.y_denom_cont[i]) - eta_denom) / (sigma_denom * sigma_denom) : 0.0;
    } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
        // Beta-binomial: mu = logit^{-1}(eta), phi = precision
        // dLL/d(eta) = d/d(eta) [lbeta(y + mu*phi, n-y + (1-mu)*phi) - lbeta(mu*phi, (1-mu)*phi)]
        // = mu*(1-mu)*phi * [digamma(y + mu*phi) - digamma(n-y + (1-mu)*phi)
        //                    - digamma(mu*phi) + digamma((1-mu)*phi)]
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        double a = p * phi_num;
        double b = (1.0 - p) * phi_num;
        double y = data.legacy.y_num[i];
        double n = data.legacy.y_denom[i];
        double dp_deta = p * (1.0 - p);  // sigmoid derivative
        double da_deta = dp_deta * phi_num;
        double db_deta = -dp_deta * phi_num;
        dLL_deta_num = da_deta * (tulpa::math::portable_digamma(y + a) - tulpa::math::portable_digamma(a))
                     + db_deta * (tulpa::math::portable_digamma(n - y + b) - tulpa::math::portable_digamma(b));
    }
}

// Per-observation ZI-aware residual computation
// Produces dLL/deta_num, dLL/deta_denom, plus phi and ZI/OI gradients per observation.
// When no ZI/OI is active, delegates to compute_obs_residuals and zeros ZI outputs.
static inline void compute_obs_residuals_zi(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    const double* beta_zi, const double* beta_oi,
    double& dLL_deta_num, double& dLL_deta_denom,
    double& grad_phi_num_i, double& grad_phi_denom_i,
    double& grad_logit_zi_i, double& grad_logit_oi_i
) {
    dLL_deta_num = 0.0;
    dLL_deta_denom = 0.0;
    grad_phi_num_i = 0.0;
    grad_phi_denom_i = 0.0;
    grad_logit_zi_i = 0.0;
    grad_logit_oi_i = 0.0;

    // Non-ZI fast path: delegate to compute_obs_residuals
    if (!layout.has_zi && !layout.has_oi) {
        compute_obs_residuals(data, i, eta_num, eta_denom, phi_num, phi_denom,
                              dLL_deta_num, dLL_deta_denom);
        return;
    }

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
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int n_trials = data.legacy.y_denom[i];
        int y_num_i = data.legacy.y_num[i];

        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
            if (y_num_i == 0) {
                double p0_binom = std::pow(1.0 - p, n_trials);
                double p0 = zi_prob + (1.0 - zi_prob) * p0_binom;
                dLL_deta_num = -(1.0 - zi_prob) * n_trials * p * p0_binom / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_binom) / p0;
            } else {
                dLL_deta_num = y_num_i - n_trials * p;
                grad_logit_zi_i = -zi_prob;
            }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
            } else {
                double p0_binom = std::pow(1.0 - p, n_trials);
                double normalizer = 1.0 - p0_binom;
                if (normalizer < 1e-12) normalizer = 1e-12;
                double grad_normalizer = n_trials * p * p0_binom / normalizer;
                dLL_deta_num = (y_num_i - n_trials * p) - grad_normalizer;
                grad_logit_zi_i = 1.0 - zi_prob;
            }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
            if (y_num_i == n_trials) {
                double pn = std::pow(p, n_trials);
                double P_yn = oi_prob + (1.0 - oi_prob) * pn;
                if (P_yn < 1e-12) P_yn = 1e-12;
                dLL_deta_num = (1.0 - oi_prob) * n_trials * pn * (1.0 - p) / P_yn;
                grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / P_yn;
            } else {
                dLL_deta_num = y_num_i - n_trials * p;
                grad_logit_oi_i = -oi_prob;
            }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::ZOIB) {
            if (y_num_i == 0) {
                double binom_zero = std::pow(1.0 - p, n_trials);
                double A = zi_prob;
                double B = (1.0 - zi_prob) * (1.0 - oi_prob) * binom_zero;
                double P = A + B;
                if (P < 1e-12) P = 1e-12;
                double d_binom_d_eta = -n_trials * binom_zero * p;
                dLL_deta_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_binom_d_eta / P;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - (1.0 - oi_prob) * binom_zero) / P;
                grad_logit_oi_i = -(1.0 - zi_prob) * binom_zero * oi_prob * (1.0 - oi_prob) / P;
            } else if (y_num_i == n_trials) {
                double pn = std::pow(p, n_trials);
                double C = oi_prob + (1.0 - oi_prob) * pn;
                double P = (1.0 - zi_prob) * C;
                if (P < 1e-12) P = 1e-12;
                double d_pn_d_eta = n_trials * pn * (1.0 - p);
                dLL_deta_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_pn_d_eta / P;
                grad_logit_zi_i = -zi_prob;
                grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / C;
            } else {
                dLL_deta_num = y_num_i - n_trials * p;
                grad_logit_zi_i = -zi_prob;
                grad_logit_oi_i = -oi_prob;
            }
        } else {
            dLL_deta_num = y_num_i - n_trials * p;
        }

    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        // ---- NEGBIN_NEGBIN ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];
        int y_denom_i = data.legacy.y_denom[i];

        // Denominator NegBin gradient (always standard, not ZI)
        double denom_d = mu_denom + phi_denom;
        dLL_deta_denom = y_denom_i - mu_denom * (y_denom_i + phi_denom) / denom_d;
        grad_phi_denom_i = tulpa::math::portable_digamma(y_denom_i + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                           + std::log(phi_denom / denom_d)
                           + (mu_denom - y_denom_i) / denom_d;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            if (y_num_i == 0) {
                double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
                double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
                dLL_deta_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
                grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
            } else {
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
            }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = 0.0;
            } else {
                double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                            - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
                grad_logit_zi_i = 1.0 - zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
                grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
            }
        } else {
            double denom_num = mu_num + phi_num;
            dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
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
            dLL_deta_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate = phi_denom / mu_denom;
            grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                    - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_POISSON) {
            double exp_neg_mu = std::exp(-mu_num);
            if (y_num_i == 0) {
                double p0 = zi_prob + (1.0 - zi_prob) * exp_neg_mu;
                dLL_deta_num = -(1.0 - zi_prob) * exp_neg_mu * mu_num / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - exp_neg_mu) / p0;
                grad_phi_denom_i = grad_phi_gamma;
            } else {
                dLL_deta_num = y_num_i - mu_num;
                grad_logit_zi_i = -zi_prob;
                grad_phi_denom_i = grad_phi_gamma;
            }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_POISSON) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
                grad_phi_denom_i = grad_phi_gamma;
            } else {
                double exp_neg_mu = std::exp(-mu_num);
                dLL_deta_num = y_num_i - mu_num - mu_num * exp_neg_mu / (1.0 - exp_neg_mu);
                grad_logit_zi_i = 1.0 - zi_prob;
                grad_phi_denom_i = grad_phi_gamma;
            }
        } else {
            dLL_deta_num = y_num_i - mu_num;
            grad_phi_denom_i = grad_phi_gamma;
        }

    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        // ---- NEGBIN_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) - skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
            dLL_deta_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate = phi_denom / mu_denom;
            grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                    - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling (same as NEGBIN_NEGBIN)
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            if (y_num_i == 0) {
                double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
                double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
                dLL_deta_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
                grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
            } else {
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
            }
            grad_phi_denom_i = grad_phi_gamma;
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = 0.0;
            } else {
                double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                            - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
                grad_logit_zi_i = 1.0 - zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
                grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
            }
            grad_phi_denom_i = grad_phi_gamma;
        } else {
            double denom_num = mu_num + phi_num;
            dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            grad_phi_denom_i = grad_phi_gamma;
        }

    } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
        // ---- GAMMA_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];

        if (y_num_i > 0.0) {
            dLL_deta_num = phi_num * (y_num_i / mu_num - 1.0);
            double rate_num = phi_num / mu_num;
            grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                             - tulpa::math::portable_digamma(phi_num) - y_num_i / mu_num;
        }
        if (y_denom_i > 0.0) {
            dLL_deta_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate_denom = phi_denom / mu_denom;
            grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                               - tulpa::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;
        }

    } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
        // ---- LOGNORMAL ----
        double mu_num = eta_num;
        double mu_denom = eta_denom;
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];
        double log_y_num = std::log(y_num_i);
        double log_y_denom = std::log(y_denom_i);
        double sigma_num = phi_num;
        double sigma_denom = phi_denom;
        double sigma_num_sq = sigma_num * sigma_num;
        double sigma_denom_sq = sigma_denom * sigma_denom;

        dLL_deta_num = (log_y_num - mu_num) / sigma_num_sq;
        dLL_deta_denom = (log_y_denom - mu_denom) / sigma_denom_sq;

        double z_num = (log_y_num - mu_num) / sigma_num;
        double z_denom = (log_y_denom - mu_denom) / sigma_denom;
        grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
        grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

    } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
        // ---- BETA_BINOMIAL ----
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y_i = data.legacy.y_num[i];
        int n_i = data.legacy.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;

        double psi_y_alpha = tulpa::math::portable_digamma(y_i + alpha);
        double psi_nmy_beta = tulpa::math::portable_digamma(n_i - y_i + beta_param);
        double psi_alpha = tulpa::math::portable_digamma(alpha);
        double psi_beta = tulpa::math::portable_digamma(beta_param);
        double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        dLL_deta_num = dLL_dp * p * (1.0 - p);

        double psi_n_phi = tulpa::math::portable_digamma(n_i + phi_num);
        double psi_phi = tulpa::math::portable_digamma(phi_num);
        grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;
    }
}

// Scatter residuals to beta gradient slots
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    for (int j = 0; j < data.legacy.p_num; j++) {
        grad[layout.legacy.beta_num_start + j] += dLL_deta_num * data.legacy.X_num_flat[i * data.legacy.p_num + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        grad[layout.legacy.beta_denom_start + j] += dLL_deta_denom * data.legacy.X_denom_flat[i * data.legacy.p_denom + j];
    }
}

// Scatter residuals to RE gradient slot
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    if (layout.has_re && data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
    }
}

// Per-observation phi likelihood gradient accumulation
// Handles NB phi_num, NB phi_denom, and Gamma phi_denom
static inline void accumulate_phi_likelihood_grad(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double* grad
) {
    // phi_num gradient
    if (layout.legacy.has_phi_num) {
        if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
            data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
            double mu_num = std::exp(eta_num);
            double y = data.legacy.y_num[i];
            double dLL_dphi = tulpa::math::portable_digamma(y + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;
        } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
            // Gamma shape_num: dLL/d(shape) = log(shape/mu) + 1 + log(y) - digamma(shape) - y/mu
            double y = data.legacy.y_num_cont[i];
            if (y > 0.0) {
                double mu_num = std::exp(eta_num);
                double dLL_dphi = std::log(phi_num / mu_num) + 1.0 + std::log(y)
                                 - tulpa::math::portable_digamma(phi_num) - y / mu_num;
                grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;
            }
        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
            // Lognormal sigma_num: dLL/d(sigma) = -1/sigma + z^2/sigma where z = (log(y)-eta)/sigma
            double y = data.legacy.y_num_cont[i];
            if (y > 0.0) {
                double z = (std::log(y) - eta_num) / phi_num;
                double dLL_dphi = (-1.0 + z * z) / phi_num;
                grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;  // chain rule: d/d(log_phi) = dphi * phi
            }
        } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
            // Beta-binomial precision: phi = alpha + beta, a = p*phi, b = (1-p)*phi
            // dLL/d(phi) = p*[digamma(y+a) - digamma(a)] + (1-p)*[digamma(n-y+b) - digamma(b)]
            //              - [digamma(n+phi) - digamma(phi)]
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            double a = p * phi_num;
            double b = (1.0 - p) * phi_num;
            double y = data.legacy.y_num[i];
            double n = data.legacy.y_denom[i];
            double dLL_dphi = p * (tulpa::math::portable_digamma(y + a) - tulpa::math::portable_digamma(a))
                            + (1.0 - p) * (tulpa::math::portable_digamma(n - y + b) - tulpa::math::portable_digamma(b))
                            - (tulpa::math::portable_digamma(n + phi_num) - tulpa::math::portable_digamma(phi_num));
            grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;
        }
    }

    // phi_denom gradient
    if (layout.legacy.has_phi_denom) {
        if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_denom = std::exp(eta_denom);
            double y = data.legacy.y_denom[i];
            double dLL_dphi = tulpa::math::portable_digamma(y + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                             + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                             - (y + phi_denom) / (mu_denom + phi_denom);
            grad[layout.legacy.log_phi_denom_idx] += dLL_dphi * phi_denom;
        } else if (data.legacy.model_type == ModelType::POISSON_GAMMA ||
                   data.legacy.model_type == ModelType::NEGBIN_GAMMA ||
                   data.legacy.model_type == ModelType::GAMMA_GAMMA) {
            // Gamma shape_denom: same formula for PG, NG, and GG
            double y = data.legacy.y_denom_cont[i];
            if (y > 0.0) {
                double mu_denom = std::exp(eta_denom);
                double rate = phi_denom / mu_denom;
                double dLL_dphi = std::log(rate) + 1.0 + std::log(y)
                                 - tulpa::math::portable_digamma(phi_denom) - y / mu_denom;
                grad[layout.legacy.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
            // Lognormal sigma_denom
            double y = data.legacy.y_denom_cont[i];
            if (y > 0.0) {
                double z = (std::log(y) - eta_denom) / phi_denom;
                double dLL_dphi = (-1.0 + z * z) / phi_denom;
                grad[layout.legacy.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }
}

// Temporal tau prior gradient on log-scale
// d/d(log_tau) = shape - rate*tau
// (equivalently: (shape-1) - rate*tau + 1 with Jacobian expanded)
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad
) {
    grad[layout.log_tau_temporal_idx] = data.tau_temporal_shape
                                        - data.tau_temporal_rate * tau_temporal;
}

// Temporal sum-to-zero penalty gradient for RW1/RW2
// d/d(phi[t]) [-0.5 * lambda * (sum phi)^2] = -lambda * sum(phi)
static inline void temporal_sum_to_zero_grad(
    const double* phi, int T, int base_idx, double lambda,
    double* grad
) {
    double sp = 0.0;
    for (int t = 0; t < T; t++) sp += phi[t];
    for (int t = 0; t < T; t++) grad[base_idx + t] -= lambda * sp;
}

// =====================================================================
// Temporal GMRF prior gradient helper (RW1/RW2/AR1)
// Shared by all gradient functions that include temporal effects.
// Writes phi gradients, tau gradient, and rho gradient (for AR1).
// Expects grad_temporal_lik[0..T_len-1] to hold likelihood contributions.
// =====================================================================
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const double* phi_temporal, int T_len,
    const double* grad_temporal_lik,
    double* grad
) {
    int T = data.n_times;
    int n_groups = data.n_temporal_groups;

    // Initialize temporal gradients with likelihood contribution
    for (int t = 0; t < T_len; t++) {
        grad[layout.temporal_start + t] = grad_temporal_lik[t];
    }

    if (data.temporal_type == TemporalType::RW1) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t > 0) {
                    g += tau_temporal * (phi_g[t - 1] - phi_g[t]);
                    qf += (phi_g[t] - phi_g[t - 1]) * (phi_g[t] - phi_g[t - 1]);
                }
                if (t < T - 1) g += tau_temporal * (phi_g[t + 1] - phi_g[t]);
                grad[base + t] += g;
            }
            if (data.temporal_cyclic) {
                double dc = phi_g[0] - phi_g[T - 1];
                qf += dc * dc;
                grad[base + 0] -= tau_temporal * dc;
                grad[base + T - 1] += tau_temporal * dc;
            }
            total_qf += qf;
            total_rank += data.temporal_cyclic ? T : T - 1;
            temporal_sum_to_zero_grad(phi_g, T, base, 0.001, grad);
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * total_rank - 0.5 * tau_temporal * total_qf;

    } else if (data.temporal_type == TemporalType::RW2) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t >= 2) g -= tau_temporal * (phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t]);
                if (t >= 1 && t < T - 1) g += 2.0 * tau_temporal * (phi_g[t - 1] - 2.0 * phi_g[t] + phi_g[t + 1]);
                if (t < T - 2) g -= tau_temporal * (phi_g[t] - 2.0 * phi_g[t + 1] + phi_g[t + 2]);
                grad[base + t] += g;
            }
            for (int t = 2; t < T; t++) {
                double d2 = phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t];
                qf += d2 * d2;
            }
            if (data.temporal_cyclic && T >= 3) {
                double d2_a = phi_g[T - 2] - 2.0 * phi_g[T - 1] + phi_g[0];
                double d2_b = phi_g[T - 1] - 2.0 * phi_g[0] + phi_g[1];
                qf += d2_a * d2_a + d2_b * d2_b;
                grad[base + T - 2] -= tau_temporal * d2_a;
                grad[base + T - 1] += 2.0 * tau_temporal * d2_a;
                grad[base + 0] -= tau_temporal * d2_a;
                grad[base + T - 1] -= tau_temporal * d2_b;
                grad[base + 0] += 2.0 * tau_temporal * d2_b;
                grad[base + 1] -= tau_temporal * d2_b;
            }
            total_qf += qf;
            total_rank += data.temporal_cyclic ? T : T - 2;
            temporal_sum_to_zero_grad(phi_g, T, base, 0.001, grad);
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * total_rank - 0.5 * tau_temporal * total_qf;

    } else if (data.temporal_type == TemporalType::AR1) {
        double omr2 = 1.0 - rho_ar1 * rho_ar1;
        double total_qf = 0.0, total_gr = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            grad[base] += -tau_temporal * omr2 * phi_g[0];
            if (T > 1) grad[base] += tau_temporal * rho_ar1 * (phi_g[1] - rho_ar1 * phi_g[0]);
            double qf = omr2 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                double r = phi_g[t] - rho_ar1 * phi_g[t - 1];
                qf += r * r;
                double g = -tau_temporal * r;
                if (t < T - 1) g += tau_temporal * rho_ar1 * (phi_g[t + 1] - rho_ar1 * phi_g[t]);
                grad[base + t] += g;
            }
            total_qf += qf;
            total_gr += tau_temporal * rho_ar1 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                total_gr += tau_temporal * (phi_g[t] - rho_ar1 * phi_g[t - 1]) * phi_g[t - 1];
            }
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * total_qf;
        if (layout.logit_rho_ar1_idx >= 0) {
            double gr = -n_groups * rho_ar1 / omr2 + total_gr;
            grad[layout.logit_rho_ar1_idx] += gr * rho_ar1 * (1.0 - rho_ar1);
        }
    }
}

// =====================================================================
// PC prior gradient on log(sigma2) for GP variances
// Returns d log_prior / d log_sigma2 INCLUDING Jacobian for exp transform.
// Formula: -0.5 * rate * sigma + 0.5  where rate = -log(alpha) / U
// =====================================================================
static inline double gp_pc_prior_grad_log_sigma2(
    double sigma2, double U, double alpha
) {
    double sigma = std::sqrt(sigma2 + 1e-10);
    double rate = -std::log(alpha + 1e-10) / (U + 1e-10);
    return -0.5 * rate * sigma + 0.5;
}

// =====================================================================
// Build GPData views from MultiscaleGPData for local and regional scales
// =====================================================================
static inline std::pair<GPData, GPData> make_msgp_gp_views(
    const MultiscaleGPData& msgp
) {
    GPData gp_local;
    gp_local.n_obs = msgp.n_obs;
    gp_local.nn = msgp.nn_local;
    gp_local.coords = msgp.coords;
    gp_local.nn_idx = msgp.nn_idx_local;
    gp_local.nn_dist = msgp.nn_dist_local;
    gp_local.nn_order = msgp.nn_order_local;
    gp_local.nn_order_inv = msgp.nn_order_inv_local;
    gp_local.cov_type = msgp.cov_type;

    GPData gp_regional;
    gp_regional.n_obs = msgp.n_obs;
    gp_regional.nn = msgp.nn_regional;
    gp_regional.coords = msgp.coords;
    gp_regional.nn_idx = msgp.nn_idx_regional;
    gp_regional.nn_dist = msgp.nn_dist_regional;
    gp_regional.nn_order = msgp.nn_order_regional;
    gp_regional.nn_order_inv = msgp.nn_order_inv_regional;
    gp_regional.cov_type = msgp.cov_type;

    return {gp_local, gp_regional};
}

// =====================================================================
// Spatial ICAR/BYM2 GMRF prior gradient helper
// Used by ms_temporal and spatiotemporal gradient functions.
// Writes phi_spatial gradients, theta_bym2 gradients (BYM2), tau gradient (ICAR).
// grad_spatial_lik and grad_theta_lik must hold accumulated likelihood contributions.
// =====================================================================
static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* spatial_phi,
    double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2,
    double rho_bym2,
    const double* theta_bym2,
    const double* grad_spatial_lik,
    const double* grad_theta_lik,
    double* grad
) {
    int S = data.n_spatial_units;
    if (layout.is_bym2) {
        // Soft sum-to-zero gradient: -0.01 * sum(phi) for each phi[s]
        double phi_sum = 0.0;
        for (int s = 0; s < S; s++) phi_sum += spatial_phi[s];
        for (int s = 0; s < S; s++) {
            double icar_grad = 0.0;
            for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                int j = data.adj_col_idx[idx];
                icar_grad += (spatial_phi[j] - spatial_phi[s]);
            }
            grad[layout.spatial_start + s] = grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor + icar_grad - 0.01 * phi_sum;
            grad[layout.theta_bym2_start + s] = grad_theta_lik[s] * sigma_u_bym2 - theta_bym2[s];
        }
        double grad_sigma_s_lik = 0.0, grad_sigma_u_lik = 0.0;
        for (int s = 0; s < S; s++) {
            grad_sigma_s_lik += grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s];
            grad_sigma_u_lik += grad_theta_lik[s] * sigma_u_bym2 * theta_bym2[s];
        }
        grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
        grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                    - rho_bym2 * grad_sigma_u_lik);
    } else {
        for (int s = 0; s < S; s++) {
            double icar_grad = 0.0;
            for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                int j = data.adj_col_idx[idx];
                icar_grad += tau_spatial * (spatial_phi[j] - spatial_phi[s]);
            }
            grad[layout.spatial_start + s] = grad_spatial_lik[s] + icar_grad;
        }
        double icar_qf = icar_quadratic_form_ptr(spatial_phi, S, data);
        grad[layout.log_tau_spatial_idx] += 0.5 * (S - 1) - 0.5 * tau_spatial * icar_qf;
    }
}
