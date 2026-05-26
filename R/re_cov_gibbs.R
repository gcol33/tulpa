# Exact-target Gibbs estimation of a random-effect covariance Sigma.
#
# Bias-1 fix (the Laplace / PQL "approximation" bias): the Laplace marginal
# fits Sigma at the joint mode and reports a curvature-based covariance for the
# random effects. For binary / low-count responses with small groups the true
# random-effect conditional p(b_g | beta, Sigma, y) is non-Gaussian, so its
# Gaussian (Laplace) approximation under-disperses -> Sigma is biased low. This
# sampler removes that bias by sampling the exact joint posterior
# p(beta, b, Sigma | y) with a Metropolis-within-Gibbs scheme:
#
#   * b_g | beta, Sigma, y  -- random-walk Metropolis per group (groups are
#       conditionally independent given beta, Sigma), proposal SHAPE taken from
#       the Laplace per-group posterior covariance block (return_re_cov, the
#       item-1 surface) and a globally adapted scale. MH against the exact
#       group likelihood + Gaussian RE prior corrects the non-Gaussianity.
#   * beta | b, Sigma, y    -- random-walk Metropolis, proposal shape from the
#       Laplace fixed-effect Hessian block (H_beta).
#   * Sigma | b             -- EXACT conjugate draw. Under an inverse-Wishart
#       prior IW(nu0, Lambda0) and b_g ~ N(0, Sigma) i.i.d. over G groups,
#       Sigma | b ~ IW(nu0 + G, Lambda0 + sum_g b_g b_g'). No linearization.
#
# Composition: Laplace body (starting values + proposal shapes) + MH debias
# (exact b / beta conditionals) + conjugate outer (exact Sigma). The summary
# reuses `.re_cov_derived_summary` (shared with the grid integrator).

# inverse-Wishart sampler via Bartlett (native, no dependency) ---------------
# Sigma ~ IW(df, Lambda)  <=>  Sigma^{-1} ~ Wishart(df, Lambda^{-1}). Sample the
# Wishart by Bartlett (Lambda^{-1} = C C'; A lower-triangular with
# A_ii = sqrt(chisq_{df - i + 1}), A_ij ~ N(0,1) for i > j; W = C A A' C'), then
# invert. Requires df > p - 1.
.rinvwishart <- function(df, Lambda) {
  p <- nrow(Lambda)
  if (df <= p - 1L) {
    stop(sprintf("inverse-Wishart df = %g must exceed p - 1 = %d.", df, p - 1L),
         call. = FALSE)
  }
  V <- chol2inv(chol(Lambda))            # Lambda^{-1} (SPD)
  C <- t(chol(V))                        # V = C C'
  A <- matrix(0, p, p)
  for (i in seq_len(p)) {
    A[i, i] <- sqrt(stats::rchisq(1L, df - i + 1L))
    if (i > 1L) A[i, seq_len(i - 1L)] <- stats::rnorm(i - 1L)
  }
  CA <- C %*% A
  W  <- tcrossprod(CA)                   # Wishart(df, V) = Sigma^{-1}
  chol2inv(chol(W))                      # Sigma
}

# Per-observation log-likelihood matching tulpa's links (logit / log / identity
# / log). Full normalizing constants included (harmless: only differences enter
# the MH ratio, but keeps the routine self-contained / reusable).
.re_obs_loglik <- function(eta, y, n_trials, family, phi) {
  switch(family,
    binomial       = stats::dbinom(y, size = n_trials, prob = stats::plogis(eta),
                                   log = TRUE),
    poisson        = stats::dpois(y, lambda = exp(eta), log = TRUE),
    gaussian       = stats::dnorm(y, mean = eta, sd = sqrt(phi), log = TRUE),
    neg_binomial_2 = stats::dnbinom(y, size = phi, mu = exp(eta), log = TRUE),
    negbin         = stats::dnbinom(y, size = phi, mu = exp(eta), log = TRUE),
    stop(sprintf("tulpa_re_cov_gibbs: unsupported family '%s'.", family),
         call. = FALSE)
  )
}

# Symmetrize + eigen-floor a proposal covariance block to SPD, then return its
# lower Cholesky factor (the random-walk proposal shape).
.re_chol_spd <- function(S, floor = 1e-8) {
  S <- (S + t(S)) / 2
  e <- eigen(S, symmetric = TRUE)
  d <- pmax(e$values, floor)
  M <- e$vectors %*% (d * t(e$vectors))
  t(chol((M + t(M)) / 2))
}

#' Gibbs estimation of a random-effect covariance (exact-target debias)
#'
#' @description
#' For a single correlated random-effects term (e.g. `(1 + x | g)`), estimate
#' the random-effect covariance `Sigma` by sampling the exact joint posterior
#' `p(beta, b, Sigma | y)` rather than fixing `Sigma` at the Laplace mode. This
#' removes the Laplace / PQL "approximation" bias that shrinks variance
#' components low for binary and low-count responses with small groups.
#'
#' @details
#' Metropolis-within-Gibbs targeting `p(beta, b, Sigma | y)`:
#'
#' - **`b_g | beta, Sigma, y`** -- random-walk Metropolis per group (groups are
#'   conditionally independent given `beta`, `Sigma`). The proposal *shape* is
#'   the Laplace per-group posterior covariance block (`return_re_cov` from
#'   [tulpa_laplace()]); the Metropolis acceptance is computed against the exact
#'   group log-likelihood plus the Gaussian RE log-prior, which corrects the
#'   non-Gaussianity the Laplace approximation misses.
#' - **`beta | b, Sigma, y`** -- random-walk Metropolis, proposal shape from the
#'   Laplace fixed-effect Hessian block `H_beta`.
#' - **`Sigma | b`** -- exact conjugate inverse-Wishart draw: under prior
#'   `IW(prior_df, prior_scale)` and `b_g ~ N(0, Sigma)` i.i.d. over `G` groups,
#'   `Sigma | b ~ IW(prior_df + G, prior_scale + sum_g b_g b_g')`. No
#'   linearization enters this step -- the bias is removed by sampling `b`
#'   exactly, not by correcting the `Sigma` update.
#'
#' A single Laplace solve provides the starting values (`beta`, `b`) and the two
#' proposal shapes; the random-walk scales for the `beta` block and the (shared)
#' `b` blocks are adapted toward their target acceptance during burn-in
#' (Robbins-Monro), then held fixed for the recorded sweeps. The covariance
#' summary marginalizes the derived scale / correlation parameters over the
#' posterior draws via the same machinery as [tulpa_re_cov_nested()] (each
#' `sigma_i` / `rho_ij` computed per draw, then quantiled).
#'
#' For `family = "gaussian"` the response is already conditionally Gaussian, so
#' there is no Laplace bias to remove; `phi` (the residual variance) is treated
#' as known. The sampler still runs and is useful as a reference / for `Sigma`
#' uncertainty.
#'
#' @param y,n_trials,X,family,phi Passed to the likelihood and to
#'   [tulpa_laplace()] for the pilot solve. `n_trials = NULL` defaults to 1.
#' @param re_term A single random-effect term: a list with `idx` (1-based group
#'   index per observation), `n_groups`, `n_coefs` (`c`), and `Z` (the
#'   `n_obs x c` RE design, e.g. `cbind(1, x)` for `(1 + x | g)`). Any supplied
#'   `L` / `cov` / `sigma` is ignored -- `Sigma` is what this function samples.
#' @param n_iter Number of recorded post-burn-in sweeps (default 2000).
#' @param n_burnin Burn-in sweeps, used for proposal-scale adaptation (default
#'   1000).
#' @param thin Keep every `thin`-th recorded sweep (default 1).
#' @param prior_df Inverse-Wishart prior degrees of freedom `nu0` (default
#'   `n_coefs + 1`, the minimal proper / weakly informative choice). Must
#'   exceed `n_coefs - 1`.
#' @param prior_scale Inverse-Wishart prior scale matrix `Lambda0`
#'   (`n_coefs x n_coefs`, default `diag(n_coefs)`).
#' @param beta_prior_mean,beta_prior_sd Gaussian fixed-effect prior (defaults
#'   `0` and `100`, matching the weak built-in prior of [tulpa_laplace()]).
#'   Scalars are recycled to `ncol(X)`.
#' @param seed Optional integer seed for reproducibility.
#' @param max_iter,tol,n_threads Pilot-solve controls (see [tulpa_laplace()]).
#'
#' @return A list with:
#'   - `posterior`: data frame with one row per parameter (`sigma_i`, `rho_ij`,
#'     `Sigma_ij`) and columns `mean`, `sd`, `median`, `ci_lo`, `ci_hi`.
#'   - `Sigma_mean`: the posterior mean of `Sigma` (a `c x c` matrix).
#'   - `Sigma_draws`: list of recorded `Sigma` draws.
#'   - `beta_draws` / `draws`: matrix of recorded `beta` draws (`n_kept x ncol(X)`);
#'     `draws` (with `means`, `param_names`, `process_info`) drives the generic
#'     `tulpa_fit` methods (`coef`/`confint`/`vcov`/`summary`).
#'   - `accept`: list with `beta` and `b` acceptance rates over recorded sweeps.
#'   - `n_kept`, `n_coefs`, `prior`: bookkeeping.
#'
#' @seealso [tulpa_re_cov_nested()] for the grid-integration (summary-bias) fix;
#'   [tulpa_laplace()] for the pilot solve and per-group covariance blocks.
#'
#' @export
tulpa_re_cov_gibbs <- function(y, n_trials = NULL, X, re_term,
                               family = "binomial", phi = 1.0,
                               n_iter = 2000L, n_burnin = 1000L, thin = 1L,
                               prior_df = NULL, prior_scale = NULL,
                               beta_prior_mean = 0, beta_prior_sd = 100,
                               seed = NULL,
                               max_iter = 100L, tol = 1e-8, n_threads = 1L) {

  if (!is.null(seed)) set.seed(as.integer(seed))

  if (is.null(re_term$n_coefs)) re_term$n_coefs <- 1L
  c_re <- as.integer(re_term$n_coefs)
  if (c_re < 1L) stop("`re_term$n_coefs` must be >= 1.", call. = FALSE)
  if (is.null(re_term$idx) || is.null(re_term$n_groups)) {
    stop("`re_term` must supply `idx` and `n_groups`.", call. = FALSE)
  }
  if (c_re > 1L && is.null(re_term$Z)) {
    stop("`re_term$Z` (the n_obs x n_coefs RE design) is required when ",
         "n_coefs > 1.", call. = FALSE)
  }
  G   <- as.integer(re_term$n_groups)
  idx <- as.integer(re_term$idx)
  n_obs <- length(y)
  p   <- ncol(X)
  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)
  Z <- if (c_re == 1L) matrix(1, n_obs, 1L) else as.matrix(re_term$Z)

  # --- inverse-Wishart prior -------------------------------------------------
  nu0     <- if (is.null(prior_df)) c_re + 1L else prior_df
  Lambda0 <- if (is.null(prior_scale)) diag(c_re) else as.matrix(prior_scale)
  if (nu0 <= c_re - 1L) {
    stop(sprintf("`prior_df` = %g must exceed n_coefs - 1 = %d for a proper ",
                 "inverse-Wishart prior.", nu0, c_re - 1L), call. = FALSE)
  }
  if (!identical(dim(Lambda0), c(c_re, c_re))) {
    stop(sprintf("`prior_scale` must be %d x %d.", c_re, c_re), call. = FALSE)
  }

  # --- fixed-effect prior ----------------------------------------------------
  bmean <- if (length(beta_prior_mean) == 1L) rep(beta_prior_mean, p) else beta_prior_mean
  bsd   <- if (length(beta_prior_sd)   == 1L) rep(beta_prior_sd,   p) else beta_prior_sd
  if (length(bmean) != p || length(bsd) != p) {
    stop("`beta_prior_mean` / `beta_prior_sd` must be length 1 or ncol(X).",
         call. = FALSE)
  }
  log_prior_beta <- function(b) sum(stats::dnorm(b, bmean, bsd, log = TRUE))

  # --- pilot Laplace solve: starting values + proposal shapes ---------------
  # Sigma = I init; mode gives beta, b; H_beta and cov_blocks give the two
  # random-walk proposal shapes (item-1 surface).
  pilot <- tulpa_laplace(
    y = y, n_trials = n_trials, X = X,
    re_list = list(utils::modifyList(
      re_term, list(L = diag(c_re), cov = NULL, sigma = NULL))),
    family = family, phi = phi, return_hessian = TRUE, return_re_cov = TRUE,
    max_iter = max_iter, tol = tol, n_threads = n_threads
  )
  beta <- pilot$mode[seq_len(p)]
  re_vals <- pilot$mode[-seq_len(p)]
  B <- matrix(re_vals, ncol = c_re, byrow = TRUE)        # G x c, row g = b_g

  L_beta <- if (!is.null(pilot$H_beta)) {
    .re_chol_spd(tryCatch(solve(pilot$H_beta),
                          error = function(e) diag(p)))
  } else diag(p)

  # Per-group RW shape: the Laplace posterior covariance block, eigen-floored.
  if (!is.null(pilot$cov_blocks) && length(pilot$cov_blocks) == G) {
    L_g <- lapply(pilot$cov_blocks, .re_chol_spd)
  } else {
    L_g <- rep(list(diag(c_re)), G)
  }

  # --- precompute per-group observation indices and group designs -----------
  grp_obs <- split(seq_len(n_obs), idx)
  grp_obs <- grp_obs[as.character(seq_len(G))]           # ensure 1..G order
  Z_g <- lapply(seq_len(G), function(g) Z[grp_obs[[g]], , drop = FALSE])
  y_g <- lapply(seq_len(G), function(g) y[grp_obs[[g]]])
  n_g <- lapply(seq_len(G), function(g) n_trials[grp_obs[[g]]])

  # initial Sigma from the pilot (method-of-moments on the RE modes), PD-clamped
  S_init <- if (G > c_re) stats::cov(B) else diag(c_re)
  diag(S_init) <- pmax(diag(S_init), 1e-3)
  if (min(eigen(S_init, symmetric = TRUE, only.values = TRUE)$values) < 1e-8)
    S_init <- diag(c_re)
  Sigma <- S_init
  Q <- chol2inv(chol(Sigma))                             # Sigma^{-1}

  # random-walk scales (adapted during burn-in)
  s_beta <- 2.4 / sqrt(p)
  s_b    <- 2.4 / sqrt(c_re)
  tgt_beta <- if (p > 1L) 0.234 else 0.44
  tgt_b    <- if (c_re > 1L) 0.234 else 0.44

  n_sweep <- as.integer(n_burnin + n_iter)
  keep_at <- seq.int(n_burnin + 1L, n_sweep, by = as.integer(thin))
  n_kept  <- length(keep_at)
  Sigma_draws <- vector("list", n_kept)
  beta_draws  <- matrix(NA_real_, n_kept, p)
  acc_beta_rec <- 0L; acc_b_rec <- 0L; n_b_rec <- 0L
  kept <- 0L

  for (sweep in seq_len(n_sweep)) {
    adapting <- sweep <= n_burnin
    gamma_t  <- 1 / sqrt(sweep)

    # --- beta | b, Sigma, y : RW Metropolis ---------------------------------
    Zb <- rowSums(Z * B[idx, , drop = FALSE])            # RE contribution
    eta_cur <- as.numeric(X %*% beta) + Zb
    ll_cur  <- sum(.re_obs_loglik(eta_cur, y, n_trials, family, phi)) +
               log_prior_beta(beta)
    beta_prop <- beta + s_beta * as.numeric(L_beta %*% stats::rnorm(p))
    eta_prop  <- as.numeric(X %*% beta_prop) + Zb
    ll_prop   <- sum(.re_obs_loglik(eta_prop, y, n_trials, family, phi)) +
                 log_prior_beta(beta_prop)
    acc_beta <- log(stats::runif(1)) < (ll_prop - ll_cur)
    if (isTRUE(acc_beta)) beta <- beta_prop
    if (adapting) s_beta <- exp(log(s_beta) + gamma_t * ((acc_beta) - tgt_beta))

    # --- b_g | beta, Sigma, y : per-group RW Metropolis ---------------------
    xb <- as.numeric(X %*% beta)
    n_acc_b <- 0L
    for (g in seq_len(G)) {
      obs <- grp_obs[[g]]
      bg  <- B[g, ]
      eta_g_cur <- xb[obs] + as.numeric(Z_g[[g]] %*% bg)
      ll_g_cur  <- sum(.re_obs_loglik(eta_g_cur, y_g[[g]], n_g[[g]], family, phi)) -
                   0.5 * as.numeric(t(bg) %*% Q %*% bg)
      bg_prop <- bg + s_b * as.numeric(L_g[[g]] %*% stats::rnorm(c_re))
      eta_g_prop <- xb[obs] + as.numeric(Z_g[[g]] %*% bg_prop)
      ll_g_prop  <- sum(.re_obs_loglik(eta_g_prop, y_g[[g]], n_g[[g]], family, phi)) -
                    0.5 * as.numeric(t(bg_prop) %*% Q %*% bg_prop)
      if (log(stats::runif(1)) < (ll_g_prop - ll_g_cur)) {
        B[g, ] <- bg_prop
        n_acc_b <- n_acc_b + 1L
      }
    }
    if (adapting) s_b <- exp(log(s_b) + gamma_t * (n_acc_b / G - tgt_b))

    # --- Sigma | b : exact conjugate inverse-Wishart draw -------------------
    S_re  <- crossprod(B)                                # sum_g b_g b_g'
    Sigma <- .rinvwishart(nu0 + G, Lambda0 + S_re)
    Q     <- chol2inv(chol(Sigma))

    # --- record -------------------------------------------------------------
    if (!adapting && (sweep %in% keep_at)) {
      kept <- kept + 1L
      Sigma_draws[[kept]] <- Sigma
      beta_draws[kept, ]  <- beta
      acc_beta_rec <- acc_beta_rec + as.integer(acc_beta)
      acc_b_rec    <- acc_b_rec + n_acc_b
      n_b_rec      <- n_b_rec + G
    }
  }

  w <- rep(1 / n_kept, n_kept)
  summ <- .re_cov_derived_summary(Sigma_draws, w, c_re)

  # Fixed-effect posterior is the recorded beta draws directly (equal weight);
  # name the columns and expose the generic `tulpa_fit` accessors. The Sigma
  # posterior stays in `posterior` (weighted == sample quantiles at equal weight).
  beta_names <- colnames(X) %||% paste0("beta", seq_len(p))
  colnames(beta_draws) <- beta_names
  beta_mean <- colMeans(beta_draws)

  list(
    posterior   = summ$posterior,
    Sigma_mean  = summ$Sigma_mean,
    Sigma_draws = Sigma_draws,
    beta_draws  = beta_draws,
    beta        = beta_mean,
    draws       = beta_draws,
    means       = beta_mean,
    param_names = beta_names,
    process_info = list(list(name = "fixed_effects", p = p,
                             coef_names = beta_names)),
    n_samples   = n_kept,
    n_params    = p,
    N           = n_obs,
    accept      = list(beta = acc_beta_rec / n_kept,
                       b    = if (n_b_rec > 0L) acc_b_rec / n_b_rec else NA_real_),
    n_kept      = n_kept,
    n_coefs     = c_re,
    prior       = list(df = nu0, scale = Lambda0)
  )
}
