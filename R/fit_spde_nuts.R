#' Sample an SPDE GLM via NUTS at fixed Matern hyperparameters
#'
#' @description
#' Bayesian counterpart to [fit_spde()] / [laplace_spde_at()]: routes the
#' SPDE-augmented GLM through tulpa's full NUTS backend via the generic
#' `LikelihoodSpec` interface. (Beta, mesh-node effects, log-phi/log-sigma)
#' are sampled jointly conditional on the supplied (range, sigma) — exact
#' Bayesian inference for the latent block, given the Matern hypers.
#'
#' Hyperparameter integration over (range, sigma) is left to an outer loop:
#' use [fit_spde()]'s nested-Laplace grid for the standard rectangular
#' integration, or call this function repeatedly across a CCD / star grid
#' (see [ccd_grid()]) and reweight by the posterior over hypers. Joint
#' NUTS sampling of (range, sigma) in the same chain is a future slice;
#' the Q matrix would have to be rebuilt with arena-AD types per gradient
#' eval, which is heavy enough to deserve its own design pass.
#'
#' @param y Response vector. Family-specific:
#'   * `gaussian`: any real
#'   * `poisson`, `neg_binomial_2`: non-negative integers
#'   * `binomial`: non-negative integers in `[0, n_trials]`
#'   * `gamma`: strictly positive reals
#'   * `beta`: strictly in `(0, 1)`
#' @param X Fixed-effects design matrix.
#' @param spatial A `tulpa_spatial` object from [spatial_spde()] /
#'   [spatial_spde_custom()] — supplies the FEM matrices (C0, G1) and
#'   projection (A) plus the smoothness `nu`.
#' @param family One of `"gaussian"`, `"poisson"`, `"binomial"`,
#'   `"gamma"`, `"neg_binomial_2"`, `"beta"`.
#' @param n_trials Integer vector for `family = "binomial"` (else ignored).
#' @param range,sigma Matern range and marginal SD on the field. Default
#'   to the SPDE prior modes (`spatial$prior_range[1]`, `spatial$prior_sigma[1]`).
#' @param sigma_beta Prior SD on each fixed-effect coefficient.
#' @param log_phi_prior_sd Prior SD on `log(phi)`. Role of `phi` is
#'   family-specific:
#'   * `gaussian`: `phi` is the residual SD (sampled jointly)
#'   * `gamma`: `phi` is the Gamma shape (sampled jointly)
#'   * `neg_binomial_2`: `phi` is the NB size `r` (sampled jointly)
#'   * `beta`: `phi` is the Beta precision (sampled jointly)
#'   * `poisson`, `binomial`: `log_phi` is held tight and ignored downstream.
#' @param log_phi_init Starting value for `log(phi)`.
#' @param n_iter,n_warmup,max_treedepth,adapt_delta,seed,verbose Standard
#'   NUTS controls.
#'
#' @return A list with `draws` (matrix `n_samples x (p + n_mesh + 1)`),
#'   `means`, `phi_summary` (Gaussian only), `accept_prob`, `divergent`,
#'   `treedepth`, `epsilon`, plus the supplied `range`, `sigma`, and
#'   `spatial` spec.
#'
#' @seealso [fit_spde()] for the Laplace counterpart and the nested-Laplace
#'   path over (range, sigma).
#'
#' @export
tulpa_nuts_spde <- function(y, X, spatial,
                            family           = c("gaussian", "poisson", "binomial",
                                                 "gamma", "neg_binomial_2", "beta"),
                            n_trials         = NULL,
                            range            = NULL,
                            sigma            = NULL,
                            sigma_beta       = 10,
                            log_phi_prior_sd = 3,
                            log_phi_init     = 0,
                            n_iter           = 2000L,
                            n_warmup         = 1000L,
                            max_treedepth    = 10L,
                            adapt_delta      = 0.8,
                            seed             = 42L,
                            verbose          = FALSE) {

  family <- match.arg(family)
  if (!inherits(spatial, "tulpa_spatial") || !identical(spatial$type, "spde")) {
    stop("`spatial` must be an SPDE tulpa_spatial object ",
         "(see `spatial_spde()` / `spatial_spde_custom()`).",
         call. = FALSE)
  }

  y <- as.numeric(y)
  N <- length(y)
  X <- as.matrix(X)
  if (nrow(X) != N) {
    stop("nrow(X) must equal length(y).", call. = FALSE)
  }

  if (is.null(range)) range <- spatial$prior_range[1]
  if (is.null(sigma)) sigma <- spatial$prior_sigma[1]
  if (!is.numeric(range) || length(range) != 1L || !is.finite(range) || range <= 0) {
    stop("`range` must be a positive scalar.", call. = FALSE)
  }
  if (!is.numeric(sigma) || length(sigma) != 1L || !is.finite(sigma) || sigma <= 0) {
    stop("`sigma` must be a positive scalar.", call. = FALSE)
  }

  kappa    <- sqrt(8 * spatial$nu) / range
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma)
  alpha    <- as.integer(round(spatial$nu)) + 1L
  rat      <- rational_spde_coefficients(spatial$nu)

  if (is.null(n_trials)) n_trials <- rep(1L, N)
  n_trials <- as.integer(n_trials)

  res <- cpp_tulpa_fit_spde_nuts(
    y_r              = y,
    n_trials_r       = n_trials,
    X_r              = X,
    A_x              = spatial$A_x,
    A_i              = spatial$A_i,
    A_p              = spatial$A_p,
    n_obs            = N,
    n_mesh           = spatial$n_mesh,
    C0_diag          = spatial$C0_diag,
    G1_x             = spatial$G1_x,
    G1_i             = spatial$G1_i,
    G1_p             = spatial$G1_p,
    kappa            = kappa,
    tau_spde         = tau_spde,
    family           = family,
    alpha            = alpha,
    sigma_beta       = sigma_beta,
    log_phi_prior_sd = log_phi_prior_sd,
    log_phi_init     = log_phi_init,
    n_iter           = as.integer(n_iter),
    n_warmup         = as.integer(n_warmup),
    max_treedepth    = as.integer(max_treedepth),
    adapt_delta      = adapt_delta,
    seed             = as.integer(seed),
    verbose          = isTRUE(verbose),
    rational_poles   = if (!rat$is_integer) rat$poles   else NULL,
    rational_weights = if (!rat$is_integer) rat$weights else NULL
  )

  res$range   <- range
  res$sigma   <- sigma
  res$spatial <- spatial

  # phi summary on the natural scale, for families that sample it. The
  # meaning of phi is family-specific: residual SD (gaussian), shape
  # (gamma), size r (neg_binomial_2), precision (beta).
  if (family %in% c("gaussian", "gamma", "neg_binomial_2", "beta")) {
    phi_draws <- exp(res$draws[, "log_phi"])
    res$phi_summary <- c(
      mean   = mean(phi_draws),
      median = stats::median(phi_draws),
      q05    = unname(stats::quantile(phi_draws, 0.05)),
      q95    = unname(stats::quantile(phi_draws, 0.95))
    )
  }

  res
}
