#' Fit a beta-regression model via NUTS (joint sampling of beta + log_phi)
#'
#' @description
#' Bayesian counterpart to [tulpa_laplace_beta()]. Routes the Beta GLM
#' through tulpa's full NUTS backend via the generic `LikelihoodSpec`
#' interface, sampling the precision `phi` jointly with the regression
#' coefficients on the log scale. NUTS performs exact marginalisation
#' over `phi`, replacing the Brent outer-opt in [tulpa_laplace_beta()]
#' and the deterministic nested-Laplace grid that would otherwise
#' integrate over the Beta dispersion.
#'
#' Mean-precision parameterisation, default logit link:
#' `y_i ~ Beta(mu_i * phi, (1 - mu_i) * phi)` with
#' `mu_i = 1 / (1 + exp(-eta_i))` and `eta_i = X_i %*% beta`.
#'
#' @param y Response vector, strictly in `(0, 1)`.
#' @param X Fixed-effects design matrix.
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on each coefficient with SD `sd` (default
#'   `list(mean = 0, sd = 10)`).
#' @param log_phi_prior_sd Prior SD on `log(phi)`
#'   (`log_phi ~ N(0, log_phi_prior_sd)`). Default `3` (very weak;
#'   covers `phi` from ~0.001 to ~1000 within +-2 SD).
#' @param log_phi_init Starting value for `log(phi)`. Default `0`
#'   (i.e. `phi = 1`); a method-of-moments warm start can speed warmup
#'   in highly concentrated regimes.
#' @param control A named list of numerical / sampler knobs (statistical
#'   arguments stay in the signature): `n_iter` (default 2000), `n_warmup`
#'   (default 1000), `max_treedepth` (default 10), `adapt_delta` (default 0.8),
#'   `seed` (`NULL` draws from the session RNG), `verbose` (default FALSE).
#'
#' @return A list with:
#'   * `draws` -- `n_samples x (p + 1)` matrix of post-warmup draws,
#'     columns `beta[1] ... beta[p], log_phi`.
#'   * `means` -- posterior means.
#'   * `phi_summary` -- posterior mean / median / quantiles of
#'     `phi = exp(log_phi)`.
#'   * `accept_prob`, `divergent`, `treedepth`, `epsilon` -- NUTS
#'     diagnostics from the underlying chain.
#'
#' @seealso [tulpa_laplace_beta()] for the Laplace + Brent point
#'   estimate; [tulpa_laplace()] for the underlying Laplace engine.
#'
#' @examples
#' set.seed(1)
#' n <- 150L
#' X <- cbind(1, rnorm(n))
#' mu <- plogis(X %*% c(0.2, 0.7)); phi <- 8
#' y <- rbeta(n, mu * phi, (1 - mu) * phi)
#' \donttest{
#' fit <- tulpa_nuts_beta(y, X, control = list(n_iter = 500L, n_warmup = 250L))
#' colMeans(fit$draws)
#' }
#' @export
tulpa_nuts_beta <- function(y, X,
                            beta_prior       = list(mean = 0, sd = 10),
                            log_phi_prior_sd = 3,
                            log_phi_init     = 0,
                            control          = list()) {

  tulpa_check_control(control, .CONTROL_KEYS$nuts_beta, "tulpa_nuts_beta")
  stopifnot(is.numeric(y), is.matrix(X), nrow(X) == length(y))
  .assert_finite_model_inputs(NULL, y)
  .validate_family_support("beta", y)
  sigma_beta <- .beta_prior_ridge_sd(beta_prior, default_sd = 10)
  if (!is.numeric(log_phi_prior_sd) || length(log_phi_prior_sd) != 1L ||
      !is.finite(log_phi_prior_sd) || log_phi_prior_sd <= 0) {
    stop("`log_phi_prior_sd` must be a positive scalar.", call. = FALSE)
  }

  res <- cpp_tulpa_fit_beta_nuts(
    y_r              = as.numeric(y),
    X_r              = X,
    sigma_beta       = sigma_beta,
    log_phi_prior_sd = log_phi_prior_sd,
    log_phi_init     = log_phi_init,
    n_iter           = as.integer(control$n_iter %||% 2000L),
    n_warmup         = as.integer(control$n_warmup %||% 1000L),
    max_treedepth    = as.integer(control$max_treedepth %||% 10L),
    adapt_delta      = control$adapt_delta %||% 0.8,
    seed             = as.integer(control$seed %||% sample.int(.Machine$integer.max, 1L)),
    verbose          = isTRUE(control$verbose)
  )

  log_phi_draws <- res$draws[, "log_phi"]
  phi_draws     <- exp(log_phi_draws)
  res$phi_summary <- c(
    mean   = mean(phi_draws),
    median = stats::median(phi_draws),
    q05    = unname(stats::quantile(phi_draws, 0.05)),
    q95    = unname(stats::quantile(phi_draws, 0.95))
  )

  res
}
