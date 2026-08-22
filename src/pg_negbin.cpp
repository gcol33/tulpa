// pg_negbin.cpp
// Pólya-Gamma Gibbs sampler for Negative Binomial models
// PG augmentation for the logit component (omega ~ PG(y + r, eta), exact
// real shape); dispersion r updated by random-walk Metropolis-Hastings on
// log r against the marginal-of-omega NB likelihood.
// Reference: Zhou et al. (2012) "Lognormal and Gamma Mixed Negative Binomial Regression"

#include "pg_negbin.h"
#include "pg_shared.h"
#include "pg_rng.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>  // for std::pair

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

// ---------------------------------------------------------------------
// The PG normal-normal beta and random-effect updates are model-agnostic
// (the post-PG conjugate step is identical to the binomial sampler), so
// this file reuses the shared update_beta / update_re kernels declared in
// pg_shared.h. Only the NB-specific dispersion (r) update lives below.
// ---------------------------------------------------------------------

// Update sigma_re with proper half-Cauchy prior using auxiliary variable
// Reference: Gelman (2006) "Prior distributions for variance parameters"
// Half-Cauchy(0, scale): use auxiliary a ~ IG(1/2, 1/scale^2)
//                        and sigma^2 | a ~ IG(1/2, 1/a)
struct SigmaReState {
  double sigma_re;
  double aux;  // auxiliary variable for half-Cauchy
};

SigmaReState update_sigma_re_negbin_hc(
    const NumericVector& re,
    double scale,
    double aux_current
) {
  int J = re.size();
  SigmaReState result;

  double ss = 0.0;
  for (int j = 0; j < J; j++) {
    ss += re[j] * re[j];
  }

  // Update sigma^2 | b, a ~ IG((J + 1)/2, ss/2 + 1/a)
  double shape_sigma = (J + 1.0) / 2.0;
  double rate_sigma = ss / 2.0 + 1.0 / aux_current;

  double sigma_sq = 1.0 / R::rgamma(shape_sigma, 1.0 / rate_sigma);
  result.sigma_re = std::sqrt(sigma_sq);

  // Update a | sigma^2 ~ IG(1, 1/scale^2 + 1/sigma^2)
  double shape_aux = 1.0;
  double rate_aux = 1.0 / (scale * scale) + 1.0 / sigma_sq;
  result.aux = 1.0 / R::rgamma(shape_aux, 1.0 / rate_aux);

  return result;
}


// Center random effects (soft sum-to-zero constraint)
// This prevents RE from absorbing intercept and inflating sigma_re
void center_random_effects(NumericVector& re, NumericVector& beta) {
  int J = re.size();
  double re_mean = 0.0;
  for (int j = 0; j < J; j++) {
    re_mean += re[j];
  }
  re_mean /= J;

  // Absorb RE mean into intercept
  beta[0] += re_mean;

  // Center RE
  for (int j = 0; j < J; j++) {
    re[j] -= re_mean;
  }
}

// The PG-NB augmentation samples beta / eta on the Zhou (2012) log-odds scale,
// where the mean is mu = r * exp(eta). tulpa's neg_binomial_2 family is the NB2
// mean scale mu = exp(eta_nb2) (R/family_loglik.R, .mean_log), shared by the
// Laplace / NUTS backends. The two parameterizations differ only by a log(r)
// shift in the intercept: eta_nb2 = eta + log(r), so beta_nb2[0] = beta[0] +
// log(r) (slopes and random effects are identical). Report the user-facing
// draws on the NB2 mean scale so tulpa_gibbs() agrees with the other backends;
// the internal sampler stays on the Zhou scale, where the PG augmentation and
// the joint (r, beta0) update are derived. The transform is applied per draw
// (a derived quantity summarised after sampling), so the posterior is exact.
static inline void store_beta_nb2(Rcpp::NumericMatrix& beta_draws, int row,
                                  const Rcpp::NumericVector& beta, double r) {
  const int p = beta.size();
  beta_draws(row, 0) = beta[0] + std::log(r);   // intercept on the NB2 mean scale
  for (int j = 1; j < p; j++) beta_draws(row, j) = beta[j];
}

// Log-likelihood of the NB given eta (linear predictor) and r.
//
// In Zhou's logit parameterization mu = r exp(eta), so the failure probability
// r / (r + mu) is 1 / (1 + exp(eta)) with r cancelling, and
//   r log(prob) + y log(1 - prob) = y eta - (y + r) log(1 + exp(eta)).
// The right-hand side is exact at every eta -- the log(1 + exp(eta)) branches
// on the sign of eta the way log_lik_binomial_kernel does -- so the density
// needs neither an eta clamp nor a floor on prob to stay finite.
static double negbin_loglik_eta(
    const IntegerVector& y,
    const NumericVector& eta,
    double r
) {
  int n = y.size();
  double ll = 0.0;

  for (int i = 0; i < n; i++) {
    const double eta_i = eta[i];
    const double log1p_exp_eta = (eta_i > 0.0)
        ? eta_i + std::log1p(std::exp(-eta_i))
        : std::log1p(std::exp(eta_i));

    ll += R::lgammafn(y[i] + r) - R::lgammafn(y[i] + 1) - R::lgammafn(r);
    ll += y[i] * eta_i - (y[i] + r) * log1p_exp_eta;
  }

  return ll;
}

// Joint MH update for (r, beta_0) to break confounding
// Key insight: mu = r * exp(eta), so when r changes, adjust beta_0 to keep mu stable
// This "ancillary" proposal explores the posterior ridge efficiently
struct JointRBeta0Result {
  double r;
  double beta0;
  bool accepted;
};

JointRBeta0Result update_r_beta0_joint(
    const IntegerVector& y,
    const NumericMatrix& X,
    const NumericVector& beta_current,
    const NumericVector& re_contrib,
    double r_current,
    double prior_r_shape,
    double prior_r_rate,
    double prior_beta_sd
) {
  int n = y.size();
  double beta0_current = beta_current[0];

  // Proposal: log(r) changes, beta_0 adjusts to maintain approximate mean
  double log_r_current = std::log(r_current);
  double proposal_sd_r = 0.2;

  double log_r_prop = R::rnorm(log_r_current, proposal_sd_r);
  double r_prop = std::exp(log_r_prop);

  // Compensating adjustment to beta_0: keep mu ≈ r * exp(beta_0 + ...)
  // If r increases by factor k, decrease beta_0 by log(k)
  double delta_log_r = log_r_prop - log_r_current;
  double beta0_prop = beta0_current - delta_log_r;  // Compensating move

  // Add small perturbation to explore
  beta0_prop += R::rnorm(0.0, 0.05);

  JointRBeta0Result result;
  result.r = r_current;
  result.beta0 = beta0_current;
  result.accepted = false;

  // Rejecting a proposal outside the r bounds is the Metropolis step for the
  // posterior restricted to r in [0.1, 500], which is the documented support.
  // beta_0 carries no bound: the Gibbs step that draws it does not either, so
  // a box here would leave this move rejecting from a state it calls
  // off-support.
  if (r_prop < 0.1 || r_prop > 500.0) {
    return result;
  }

  // Compute eta for current and proposed
  NumericVector eta_current(n), eta_prop(n);
  for (int i = 0; i < n; i++) {
    double x_beta_other = 0.0;
    for (int j = 1; j < X.ncol(); j++) {
      x_beta_other += X(i, j) * beta_current[j];
    }
    eta_current[i] = beta0_current + x_beta_other + re_contrib[i];
    eta_prop[i] = beta0_prop + x_beta_other + re_contrib[i];
  }

  // Log likelihood
  double ll_current = negbin_loglik_eta(y, eta_current, r_current);
  double ll_prop = negbin_loglik_eta(y, eta_prop, r_prop);

  // Log prior for r: Gamma(shape, rate)
  double log_prior_r_current = (prior_r_shape - 1.0) * log_r_current - prior_r_rate * r_current;
  double log_prior_r_prop = (prior_r_shape - 1.0) * log_r_prop - prior_r_rate * r_prop;

  // Log prior for beta_0: N(0, prior_beta_sd^2)
  double log_prior_b0_current = -0.5 * beta0_current * beta0_current / (prior_beta_sd * prior_beta_sd);
  double log_prior_b0_prop = -0.5 * beta0_prop * beta0_prop / (prior_beta_sd * prior_beta_sd);

  // Jacobian for log(r) transform
  double log_jac_current = log_r_current;
  double log_jac_prop = log_r_prop;

  // MH ratio
  double log_alpha = (ll_prop + log_prior_r_prop + log_prior_b0_prop + log_jac_prop)
                   - (ll_current + log_prior_r_current + log_prior_b0_current + log_jac_current);

  if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
    result.r = r_prop;
    result.beta0 = beta0_prop;
    result.accepted = true;
  }

  return result;
}

// Standard MH update for r (used as backup/additional update)
double update_r_negbin(
    const IntegerVector& y,
    const NumericVector& eta,
    double r_current,
    double prior_shape,
    double prior_rate
) {
  double log_r_current = std::log(r_current);
  double proposal_sd = 0.15;

  double log_r_prop = R::rnorm(log_r_current, proposal_sd);
  double r_prop = std::exp(log_r_prop);

  if (r_prop < 0.1 || r_prop > 500.0) {
    return r_current;
  }

  double log_prior_current = (prior_shape - 1.0) * log_r_current - prior_rate * r_current;
  double log_prior_prop = (prior_shape - 1.0) * log_r_prop - prior_rate * r_prop;

  double log_lik_current = negbin_loglik_eta(y, eta, r_current);
  double log_lik_prop = negbin_loglik_eta(y, eta, r_prop);

  double log_alpha = (log_prior_prop + log_lik_prop + log_r_prop)
                   - (log_prior_current + log_lik_current + log_r_current);

  if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
    return r_prop;
  }
  return r_current;
}

// Method-of-moments initialization for NB model
// Returns (r_init, beta0_init) as a pair
static std::pair<double, double> nb_mom_init(const IntegerVector& y) {
  int n = y.size();

  // Compute mean and variance. y[i] * y[i] on the IntegerVector wraps for
  // counts above 46340, so the square is formed in double.
  double sum_y = 0.0, sum_y2 = 0.0;
  for (int i = 0; i < n; i++) {
    const double yi = y[i];
    sum_y += yi;
    sum_y2 += yi * yi;
  }
  double mu_hat = sum_y / n;
  double var_hat = (sum_y2 - sum_y * sum_y / n) / (n - 1);

  // For NB2: var = mu + mu^2/r
  // So: r = mu^2 / (var - mu)
  double r_init = 1.0;  // default
  if (var_hat > mu_hat && mu_hat > 0) {
    r_init = mu_hat * mu_hat / (var_hat - mu_hat);
    r_init = std::max(0.5, std::min(r_init, 100.0));  // Constrain
  }

  // In logit parameterization: mu = r * exp(eta)
  // So eta = log(mu/r), and with intercept-only: beta_0 = log(mu/r)
  double beta0_init = 0.0;
  if (mu_hat > 0 && r_init > 0) {
    beta0_init = std::log(mu_hat / r_init);
    beta0_init = std::max(-10.0, std::min(beta0_init, 10.0));
  }

  return std::make_pair(r_init, beta0_init);
}

// ---------------------------------------------------------------------
// Main Gibbs sampler for single NB process
// ---------------------------------------------------------------------

List pg_negbin_gibbs(
    IntegerVector y,
    NumericMatrix X,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  if (N < 1) Rcpp::stop("`y` is empty.");
  if (X.nrow() != N) {
    Rcpp::stop("`X` has %d row(s) but `y` has length %d.",
               static_cast<int>(X.nrow()), N);
  }
  if (n_groups < 0) Rcpp::stop("`n_groups` must be >= 0; got %d.", n_groups);
  if (n_groups > 0) pg_check_index(group, N, n_groups, "group");
  // The random-effect centring, the moment initializer and the compensating
  // (r, beta_0) move all write beta[0] as the intercept.
  pg_require_intercept(X, "negative-binomial");
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  const int team = tulpa_omp_team_size_req(n_threads, N);

  // Storage
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_groups);
  NumericVector sigma_draws(n_save);
  NumericVector r_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Smart initialization using method of moments
  std::pair<double, double> mom_init = nb_mom_init(y);
  double r_mom = mom_init.first;
  double beta0_mom = mom_init.second;

  // Use MOM estimates if r_init is default (0 or negative)
  NumericVector beta(p, 0.0);
  NumericVector re(n_groups, 0.0);
  double sigma_re = 1.0;
  double r;

  if (r_init <= 0) {
    // Use method of moments
    r = r_mom;
    beta[0] = beta0_mom;
    if (verbose) {
      Rcpp::Rcout << "Using MOM init: r = " << r << ", beta0 = " << beta[0] << std::endl;
    }
  } else {
    r = r_init;
    // Still use MOM for beta0 to start in a good region
    beta[0] = beta0_mom;
  }
  NumericVector omega(N, 1.0);
  NumericVector kappa(N);
  NumericVector eta(N);
  NumericVector X_beta(N);
  NumericVector re_contrib(N);

  // Initialize auxiliary variable for half-Cauchy prior on sigma_re
  // a ~ IG(1/2, 1/scale^2) => E[a] = 2*scale^2
  double sigma_aux = 2.0 * prior_sigma_scale * prior_sigma_scale;

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {

    // 1. Compute linear predictor. omega is drawn at this eta and update_beta
    // solves the Gaussian conditional the same eta defines, so it carries no
    // clamp: bounding it here would draw the weight of a saturated row at one
    // linear predictor and then solve for beta as if it came from another.
    // rpg_real is finite at every |eta|, and the spatial kernel below reads
    // the same unbounded eta.
    tulpa_parallel_for(team, N, [&](int i) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = (n_groups > 0) ? re[group[i] - 1] : 0.0;
      eta[i] = X_beta[i] + re_contrib[i];
    });

    // 2. Compute kappa = (y - r) / 2
    for (int i = 0; i < N; i++) {
      kappa[i] = (y[i] - r) / 2.0;
    }

    // 3. Sample omega ~ PG(y + r, eta) at the exact (real) shape: rounding
    // the shape changes the augmented joint, worst at zero counts with
    // small r, where the conditional weight of every zero is off by up to
    // a factor 2. rpg_real handles the fractional part exactly.
    for (int i = 0; i < N; i++) {
      omega[i] = rpg_real(y[i] + r, eta[i]);
    }

    // 4. Update beta | omega, re, y
    NumericVector beta_new = update_beta(kappa, omega, X, re_contrib, prior_beta_sd);

    // Keep the previous value if the solve returned a non-finite draw. The
    // guard is finiteness only: discarding a finite draw on a magnitude bound
    // would sample the posterior restricted to that box instead, and the
    // centring below can carry beta[0] out of any such box afterwards.
    bool beta_valid = true;
    for (int j = 0; j < p; j++) {
      if (!std::isfinite(beta_new[j])) {
        beta_valid = false;
        break;
      }
    }
    if (beta_valid) {
      beta = beta_new;
    }

    // 5. Recompute X_beta
    tulpa_parallel_for(team, N, [&](int i) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    });

    // 6. Update random effects
    if (n_groups > 0) {
      NumericVector re_new = update_re(kappa, omega, X_beta, group, n_groups, sigma_re);

      bool re_valid = true;
      for (int g = 0; g < n_groups; g++) {
        if (!std::isfinite(re_new[g])) {
          re_valid = false;
          break;
        }
      }
      if (re_valid) {
        re = re_new;
      }

      // Center random effects to prevent absorbing intercept
      // This is critical for correct sigma_re estimation
      center_random_effects(re, beta);

      // Recompute X_beta since beta[0] may have changed
      tulpa_parallel_for(team, N, [&](int i) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
          X_beta[i] += X(i, j) * beta[j];
        }
      });

      // Update sigma_re with proper half-Cauchy prior using auxiliary variable
      SigmaReState sigma_state = update_sigma_re_negbin_hc(re, prior_sigma_scale, sigma_aux);
      if (std::isfinite(sigma_state.sigma_re)) {
        sigma_re = sigma_state.sigma_re;
        sigma_aux = sigma_state.aux;
      }
    }

    // 7. Update (r, beta_0) jointly to break confounding
    // This is the key fix for the identification issue
    NumericVector re_contrib_for_joint(N);
    for (int i = 0; i < N; i++) {
      re_contrib_for_joint[i] = (n_groups > 0) ? re[group[i] - 1] : 0.0;
    }

    // Joint update of (r, beta_0)
    JointRBeta0Result joint_result = update_r_beta0_joint(
        y, X, beta, re_contrib_for_joint, r,
        prior_r_shape, prior_r_rate, prior_beta_sd
    );
    if (joint_result.accepted) {
      r = joint_result.r;
      beta[0] = joint_result.beta0;
      // Recompute X_beta with new beta[0]
      for (int i = 0; i < N; i++) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
          X_beta[i] += X(i, j) * beta[j];
        }
      }
    }

    // Also do a standard r update (interweaving for better mixing). This
    // refreshes eta from the beta / re / beta_0 the sweep ended on, so the
    // saved eta column belongs to the same draw as the saved beta column
    // rather than to the state the sweep started from.
    for (int i = 0; i < N; i++) {
      eta[i] = X_beta[i] + re_contrib_for_joint[i];
    }
    r = update_r_negbin(y, eta, r, prior_r_shape, prior_r_rate);

    // Save draws (reported on the NB2 mean scale; see store_beta_nb2).
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      store_beta_nb2(beta_draws, save_idx, beta, r);
      for (int g = 0; g < n_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_draws[save_idx] = sigma_re;
      r_draws[save_idx] = r;

      if (store_eta) {
        const double log_r = std::log(r);
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i] + log_r;  // NB2 log-mean
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter
                  << " (r = " << r << ")" << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  return List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("sigma_re") = sigma_draws,
    Named("r") = r_draws,
    Named("eta") = eta_draws
  );
}

// ---------------------------------------------------------------------
// Gibbs sampler for NB with spatial effects (ICAR)
// ---------------------------------------------------------------------

List pg_negbin_gibbs_spatial(
    IntegerVector y,
    NumericMatrix X,
    IntegerVector re_group,
    int n_re_groups,
    IntegerVector spatial_group,
    int n_spatial_units,
    List adj_list,
    IntegerVector n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_tau_shape,
    double prior_tau_rate,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  if (N < 1) Rcpp::stop("`y` is empty.");
  if (X.nrow() != N) {
    Rcpp::stop("`X` has %d row(s) but `y` has length %d.",
               static_cast<int>(X.nrow()), N);
  }
  if (n_re_groups < 0) {
    Rcpp::stop("`n_re_groups` must be >= 0; got %d.", n_re_groups);
  }
  if (n_re_groups > 0) pg_check_index(re_group, N, n_re_groups, "re_group");
  pg_check_index(spatial_group, N, n_spatial_units, "spatial_group");
  // Flat CSR once at entry, the same form the binomial ICAR kernels take. The
  // adjacency is fixed for the whole run, so reading it as an Rcpp proxy per
  // unit per sweep built n_iter * n_spatial_units R objects for a structure
  // that never moves -- and the field sweeps below index it unchecked, which
  // pg_build_adjacency validates once here instead.
  const tulpa::PgAdjacency adj =
      tulpa::pg_build_adjacency(adj_list, n_neighbors, n_spatial_units);
  pg_require_intercept(X, "negative-binomial ICAR spatial");
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  // The X * beta recomputations are the same independent per-row work the
  // non-spatial negbin kernel parallelizes, and this kernel took n_threads and
  // ran all of them serially.
  const int team = tulpa_omp_team_size_req(n_threads, N);

  // Storage
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_re_groups);
  NumericMatrix spatial_draws(n_save, n_spatial_units);
  NumericVector sigma_re_draws(n_save);
  NumericVector tau_draws(n_save);
  NumericVector r_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Initialize
  NumericVector beta(p, 0.0);
  NumericVector re(n_re_groups, 0.0);
  NumericVector spatial(n_spatial_units, 0.0);
  double sigma_re = 1.0;
  double sigma_aux = 2.0 * prior_sigma_re_scale * prior_sigma_re_scale;
  double tau = 1.0;  // Spatial precision
  double r = r_init;

  // The ICAR pseudo-density is tau^{(S - k)/2} exp(-tau Q / 2) with k = number
  // of graph components, so the tau posterior shape uses (S - k)/2; a
  // disconnected adjacency (spatial(by=) replication makes this routine) with
  // (S - 1)/2 biases tau upward. pg_build_adjacency counts the components on
  // the CSR form it just built.
  const int n_components = adj.n_components;

  NumericVector omega(N, 1.0);
  NumericVector kappa(N);
  NumericVector eta(N);
  NumericVector X_beta(N);
  NumericVector re_contrib(N);
  NumericVector spatial_contrib(N);

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {

    // 1. Compute linear predictor
    tulpa_parallel_for(team, N, [&](int i) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
      spatial_contrib[i] = spatial[spatial_group[i] - 1];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
      kappa[i] = (y[i] - r) / 2.0;
    });

    // 2. Sample omega ~ PG(y + r, eta) at the exact (real) shape.
    for (int i = 0; i < N; i++) {
      omega[i] = rpg_real(y[i] + r, eta[i]);
    }

    // 3. Update beta
    NumericVector offset(N);
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = update_beta(kappa, omega, X, offset, prior_beta_sd);

    // Recompute X_beta
    tulpa_parallel_for(team, N, [&](int i) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    });

    // 4. Update random effects
    if (n_re_groups > 0) {
      NumericVector offset_re(N);
      for (int i = 0; i < N; i++) {
        offset_re[i] = X_beta[i] + spatial_contrib[i];
      }

      double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);
      NumericVector sum_omega(n_re_groups);
      NumericVector sum_resid(n_re_groups);

      pg_accumulate_stats(N, re_group.begin(), n_re_groups, omega.begin(),
                          kappa.begin(), offset_re.begin(),
                          sum_omega.begin(), sum_resid.begin());

      for (int g = 0; g < n_re_groups; g++) {
        double post_var = 1.0 / (sum_omega[g] + prior_prec);
        double post_mean = post_var * sum_resid[g];
        re[g] = R::rnorm(post_mean, std::sqrt(post_var));
      }

      // Exact half-Cauchy via the auxiliary-variable scheme, matching the
      // non-spatial negbin kernel (one prior, both kernels).
      SigmaReState sigma_state =
          update_sigma_re_negbin_hc(re, prior_sigma_re_scale, sigma_aux);
      if (std::isfinite(sigma_state.sigma_re)) {
        sigma_re = sigma_state.sigma_re;
        sigma_aux = sigma_state.aux;
      }

      // Update re_contrib
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 5. Update spatial effects (ICAR)
    NumericVector offset_spatial(N);
    for (int i = 0; i < N; i++) {
      offset_spatial[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate by spatial unit
    NumericVector sum_omega_s(n_spatial_units);
    NumericVector sum_resid_s(n_spatial_units);

    pg_accumulate_stats(N, spatial_group.begin(), n_spatial_units,
                        omega.begin(), kappa.begin(), offset_spatial.begin(),
                        sum_omega_s.begin(), sum_resid_s.begin());

    // Sample spatial effects with ICAR prior
    for (int s = 0; s < n_spatial_units; s++) {
      // ICAR: phi_s | phi_{-s} ~ N(mean of neighbors, 1/(tau * n_neighbors))
      const int e0 = adj.row_ptr[s];
      const int e1 = adj.row_ptr[s + 1];
      const int n_neigh = e1 - e0;

      double neighbor_mean = 0.0;
      if (n_neigh > 0) {
        for (int e = e0; e < e1; e++) {
          neighbor_mean += spatial[adj.col_idx[e]];
        }
        neighbor_mean /= n_neigh;
      }

      // Combine ICAR prior with data likelihood
      double prior_prec_s = tau * n_neigh;
      double post_prec = sum_omega_s[s] + prior_prec_s;
      double post_mean = (sum_resid_s[s] + prior_prec_s * neighbor_mean) / post_prec;
      double post_sd = 1.0 / std::sqrt(post_prec);

      spatial[s] = R::rnorm(post_mean, post_sd);
    }

    // Center spatial effects (sum-to-zero constraint along the ICAR's
    // improper direction), absorbing the mean into the intercept so eta is
    // unchanged and the move is exactly posterior-invariant — the same
    // convention center_random_effects uses for the iid RE block.
    double spatial_mean = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      spatial_mean += spatial[s];
    }
    spatial_mean /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      spatial[s] -= spatial_mean;
    }
    beta[0] += spatial_mean;
    tulpa_parallel_for(team, N, [&](int i) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    });

    // Update tau (spatial precision)
    // Prior: tau ~ Gamma(shape, rate)
    // Posterior: tau ~ Gamma(shape + (S - k)/2, rate + Q/2), k = number of
    // adjacency components; Q = sum over edges (phi_i - phi_j)^2
    double Q = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      for (int e = adj.row_ptr[s]; e < adj.row_ptr[s + 1]; e++) {
        const int t = adj.col_idx[e];
        if (t > s) {  // Count each edge once
          double diff = spatial[s] - spatial[t];
          Q += diff * diff;
        }
      }
    }

    double tau_shape = prior_tau_shape
        + (n_spatial_units - static_cast<double>(n_components)) / 2.0;
    double tau_rate = prior_tau_rate + Q / 2.0;
    tau = R::rgamma(tau_shape, 1.0 / tau_rate);

    // Update spatial_contrib
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = spatial[spatial_group[i] - 1];
    }

    // 6. Update dispersion r
    for (int i = 0; i < N; i++) {
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }
    r = update_r_negbin(y, eta, r, prior_r_shape, prior_r_rate);

    // Save draws (reported on the NB2 mean scale; see store_beta_nb2).
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      store_beta_nb2(beta_draws, save_idx, beta, r);
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_draws(save_idx, s) = spatial[s];
      }
      sigma_re_draws[save_idx] = sigma_re;
      tau_draws[save_idx] = tau;
      r_draws[save_idx] = r;

      if (store_eta) {
        const double log_r = std::log(r);
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i] + log_r;  // NB2 log-mean
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter
                  << " (r=" << r << ", tau=" << tau << ")" << std::endl;
    }

    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  return List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("spatial") = spatial_draws,
    Named("sigma_re") = sigma_re_draws,
    Named("tau") = tau_draws,
    Named("r") = r_draws,
    Named("eta") = eta_draws
  );
}

} // namespace tulpa

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// Scores the negative-binomial log-likelihood the Gibbs kernels evaluate, at a
// linear predictor supplied directly, so it can be held against dnbinom() over
// a range of eta the sampler never visits.
// [[Rcpp::export]]
double cpp_test_negbin_loglik_eta(
    Rcpp::IntegerVector y,
    Rcpp::NumericVector eta,
    double r
) {
  if (eta.size() != y.size()) {
    Rcpp::stop("eta holds %d values for %d observations.",
               (int) eta.size(), (int) y.size());
  }
  return tulpa::negbin_loglik_eta(y, eta, r);
}

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  return tulpa::pg_negbin_gibbs(
    y, X, group, n_groups, n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_scale, prior_r_shape, prior_r_rate, r_init,
    store_eta, verbose, n_threads
  );
}


// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_gibbs_spatial(
    Rcpp::IntegerVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_tau_shape,
    double prior_tau_rate,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  return tulpa::pg_negbin_gibbs_spatial(
    y, X, re_group, n_re_groups, spatial_group, n_spatial_units,
    adj_list, n_neighbors, n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, prior_tau_shape, prior_tau_rate,
    prior_r_shape, prior_r_rate, r_init, store_eta, verbose, n_threads
  );
}
