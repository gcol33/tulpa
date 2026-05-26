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
#' Mean–precision parameterisation, default logit link:
#' `y_i ~ Beta(mu_i * phi, (1 - mu_i) * phi)` with
#' `mu_i = 1 / (1 + exp(-eta_i))` and `eta_i = X_i %*% beta`.
#'
#' @param y Response vector, strictly in `(0, 1)`.
#' @param X Fixed-effects design matrix.
#' @param sigma_beta Prior SD on each fixed-effect coefficient
#'   (`beta_j ~ N(0, sigma_beta)`). Default `10`.
#' @param log_phi_prior_sd Prior SD on `log(phi)`
#'   (`log_phi ~ N(0, log_phi_prior_sd)`). Default `3` (very weak;
#'   covers `phi` from ~0.001 to ~1000 within +-2 SD).
#' @param log_phi_init Starting value for `log(phi)`. Default `0`
#'   (i.e. `phi = 1`); a method-of-moments warm start can speed warmup
#'   in highly concentrated regimes.
#' @param n_iter Total NUTS iterations including warmup. Default `2000`.
#' @param n_warmup Warmup iterations. Default `1000`.
#' @param max_treedepth NUTS max tree depth. Default `10`.
#' @param adapt_delta Target acceptance rate for dual averaging.
#'   Default `0.8`.
#' @param seed RNG seed.
#' @param verbose Print sampler progress.
#'
#' @return A list with:
#'   * `draws` — `n_samples x (p + 1)` matrix of post-warmup draws,
#'     columns `beta[1] ... beta[p], log_phi`.
#'   * `means` — posterior means.
#'   * `phi_summary` — posterior mean / median / quantiles of
#'     `phi = exp(log_phi)`.
#'   * `accept_prob`, `divergent`, `treedepth`, `epsilon` — NUTS
#'     diagnostics from the underlying chain.
#'
#' @seealso [tulpa_laplace_beta()] for the Laplace + Brent point
#'   estimate; [tulpa_laplace()] for the underlying Laplace engine.
#'
#' @export
tulpa_nuts_beta <- function(y, X,
                            sigma_beta       = 10,
                            log_phi_prior_sd = 3,
                            log_phi_init     = 0,
                            n_iter           = 2000L,
                            n_warmup         = 1000L,
                            max_treedepth    = 10L,
                            adapt_delta      = 0.8,
                            seed             = 42L,
                            verbose          = FALSE) {

  stopifnot(is.numeric(y), is.matrix(X), nrow(X) == length(y))
  if (any(!is.finite(y)) || min(y) <= 0 || max(y) >= 1) {
    stop("`y` must be strictly in (0, 1) for tulpa_nuts_beta(). ",
         "Use cover(positive = 'beta') for hurdle handling of 0/1.",
         call. = FALSE)
  }
  if (!is.numeric(sigma_beta) || length(sigma_beta) != 1L ||
      !is.finite(sigma_beta) || sigma_beta <= 0) {
    stop("`sigma_beta` must be a positive scalar.", call. = FALSE)
  }
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
    n_iter           = as.integer(n_iter),
    n_warmup         = as.integer(n_warmup),
    max_treedepth    = as.integer(max_treedepth),
    adapt_delta      = adapt_delta,
    seed             = as.integer(seed),
    verbose          = isTRUE(verbose)
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
