# glmm_logpost.R
# ------------------------------------------------------------------------------
# The Step-3 keystone: build a log-posterior closure for a GLMM from the
# design-matrix bundle (tulpa_build_model_data) + a character family + priors.
# This is what turns a formula model into the `log_posterior(theta)` /
# `grad_log_posterior(theta)` pair the sampler backends (mala, imh_laplace,
# pathfinder, ess, ...) consume, using the R-level family math in
# `R/family_loglik.R`.
#
# Parameterization. theta = (beta, u_1, ..., u_K), conditional on the random-
# effect standard deviations `sigma_re` (one per RE term). Each u_k is the
# flattened n_groups[k] x n_coefs[k] matrix of group-level coefficients, stored
# column-major (R default). The linear predictor is
#     eta = offset + X beta + sum_k L_k . u_k[group_k]
# where L_k = [1, slopes] are the per-observation RE loadings. The log-posterior
# is the family log-likelihood plus a Gaussian prior on beta and the conditional
# Gaussian prior u_k ~ N(0, sigma_re[k]^2). Conditioning on sigma matches the
# Laplace inner problem; sampling sigma is a future extension (the layout leaves
# room for it).
# ------------------------------------------------------------------------------

# Sum rows of `m` (n_obs x p) within groups, returning an n_groups x p matrix
# in canonical 1..n_groups order, with absent groups zero-filled. rowsum() sorts
# by group label as character, so we reindex by the integer rownames to avoid
# the "1","10","2" ordering trap.
.group_sum <- function(m, group_idx, n_groups) {
  rs <- rowsum(m, group_idx, reorder = TRUE)
  out <- matrix(0, n_groups, ncol(m))
  out[as.integer(rownames(rs)), ] <- rs
  out
}


#' Build a GLMM log-posterior (and gradient) from a model-data bundle.
#'
#' @param bundle Output of [tulpa_build_model_data()] (needs `y`, `X`,
#'   `offset`, `re_terms`, `n_obs`, `n_fixed`).
#' @param family Character family name (see [family_names()]).
#' @param sigma_re Numeric vector of random-effect SDs, one per RE term. Length
#'   must equal `length(bundle$re_terms)`. Ignored when there are no RE terms.
#' @param n_trials Binomial denominators (or `NULL`).
#' @param phi Dispersion/precision passed to the family.
#' @param beta_prior `list(mean, sd)` Gaussian prior on the fixed effects
#'   (scalars, recycled). Default `list(mean = 0, sd = 2.5)`.
#' @param weights Optional per-observation likelihood weights (length `n_obs`):
#'   each row's log-likelihood and score contribution is scaled by its weight.
#'
#' @return A list with:
#'   * `log_posterior(theta)` -- scalar log-posterior (up to a constant).
#'   * `grad_log_posterior(theta)` -- gradient vector.
#'   * `dim` -- length of `theta`.
#'   * `init` -- a zero starting vector of length `dim`.
#'   * `unpack(theta)` -- list(`beta`, `u` = list of n_groups x n_coefs matrices).
#'   * `param_names` -- character labels aligned with `theta`.
#' @keywords internal
build_glmm_logpost <- function(bundle, family, sigma_re = NULL,
                               n_trials = NULL, phi = 1.0,
                               beta_prior = list(mean = 0, sd = 2.5),
                               weights = NULL) {
  if (is.null(.FAMILY_OPS[[family]])) {
    stop(sprintf("Unknown family '%s'. Supported: %s.",
                 family, paste(family_names(), collapse = ", ")), call. = FALSE)
  }

  y       <- bundle$y
  X       <- bundle$X
  re      <- bundle$re_terms %||% list()
  K       <- length(re)
  n_obs   <- bundle$n_obs
  p       <- bundle$n_fixed
  offset0 <- bundle$offset %||% rep(0, n_obs)

  if (K > 0L) {
    if (is.null(sigma_re) || length(sigma_re) != K) {
      stop(sprintf("`sigma_re` must have one entry per RE term (need %d).", K),
           call. = FALSE)
    }
  }

  b_mean <- beta_prior$mean %||% 0
  b_sd   <- beta_prior$sd %||% 2.5

  # Precompute per-term loadings L_k (n_obs x n_coefs) and offsets into theta.
  L      <- vector("list", K)
  gidx   <- vector("list", K)
  ngrp   <- integer(K)
  ncoef  <- integer(K)
  block_len <- integer(K)
  for (k in seq_len(K)) {
    rt <- re[[k]]
    cols <- list()
    if (isTRUE(rt$has_intercept)) cols[[length(cols) + 1L]] <- rep(1, n_obs)
    if (!is.null(rt$slope_matrix)) {
      for (j in seq_len(ncol(rt$slope_matrix))) {
        cols[[length(cols) + 1L]] <- rt$slope_matrix[, j]
      }
    }
    L[[k]]    <- do.call(cbind, cols)
    gidx[[k]] <- rt$group_idx
    ngrp[k]   <- rt$n_groups
    ncoef[k]  <- rt$n_coefs
    block_len[k] <- rt$n_groups * rt$n_coefs
  }
  beta_off <- 0L
  u_off    <- cumsum(c(p, block_len))  # u_off[k] = start index (0-based) of u_k

  unpack <- function(theta) {
    beta <- theta[seq_len(p)]
    u <- vector("list", K)
    for (k in seq_len(K)) {
      blk <- theta[(u_off[k] + 1L):(u_off[k] + block_len[k])]
      u[[k]] <- matrix(blk, nrow = ngrp[k], ncol = ncoef[k])  # column-major
    }
    list(beta = beta, u = u)
  }

  linear_predictor <- function(par) {
    eta <- offset0 + as.numeric(X %*% par$beta)
    for (k in seq_len(K)) {
      Uk <- par$u[[k]]
      eta <- eta + rowSums(L[[k]] * Uk[gidx[[k]], , drop = FALSE])
    }
    eta
  }

  log_posterior <- function(theta) {
    par <- unpack(theta)
    eta <- linear_predictor(par)
    ll <- family_loglik(eta, y, family, n_trials, phi)
    ll <- if (is.null(weights)) sum(ll) else sum(weights * ll)
    lp_beta <- sum(stats::dnorm(par$beta, b_mean, b_sd, log = TRUE))
    lp_u <- 0
    for (k in seq_len(K)) {
      lp_u <- lp_u + sum(stats::dnorm(par$u[[k]], 0, sigma_re[k], log = TRUE))
    }
    ll + lp_beta + lp_u
  }

  grad_log_posterior <- function(theta) {
    par <- unpack(theta)
    eta <- linear_predictor(par)
    s   <- family_score_eta(eta, y, family, n_trials, phi)
    if (!is.null(weights)) s <- weights * s

    g_beta <- as.numeric(crossprod(X, s)) - (par$beta - b_mean) / (b_sd^2)

    g <- numeric(length(theta))
    g[seq_len(p)] <- g_beta
    for (k in seq_len(K)) {
      gb <- .group_sum(L[[k]] * s, gidx[[k]], ngrp[k]) - par$u[[k]] / (sigma_re[k]^2)
      g[(u_off[k] + 1L):(u_off[k] + block_len[k])] <- as.numeric(gb)
    }
    g
  }

  param_names <- c(
    bundle$fixed_names %||% paste0("beta", seq_len(p)),
    unlist(lapply(seq_len(K), function(k) {
      sprintf("re%d[g%d,c%d]",
              k,
              rep(seq_len(ngrp[k]), times = ncoef[k]),
              rep(seq_len(ncoef[k]), each = ngrp[k]))
    }))
  )

  list(
    log_posterior      = log_posterior,
    grad_log_posterior = grad_log_posterior,
    dim                = p + sum(block_len),
    init               = numeric(p + sum(block_len)),
    unpack             = unpack,
    param_names        = param_names
  )
}
