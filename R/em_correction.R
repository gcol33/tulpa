# ============================================================================
# Post-EM corrections for tulpa_em_laplace().
#
# Two routines, both run after EM convergence:
#
#   * MI    -- draw `n_imputations` independent hard z's from the converged
#             posterior weights P(z|y, theta_hat), refit each, pool via
#             Rubin's rules.
#   * Gibbs -- warm-started Markov chain z|theta -> theta|z. Each step draws
#             z from a fresh E-step run on the current fits, then refits.
#             Pooled via Rubin's rules.
#
# Both reuse the model package's existing `m_step_encode` contract: a hard
# z draw is fed through `m_step_encode(z, ...)` exactly the same way the
# continuous weights are. Bernoulli is the default draw rule; downstream
# packages with non-binary latent structure supply their own via `draw_z`.
# ============================================================================


# ----------------------------------------------------------------------------
# Default hard-z sampler: Bernoulli per observation. Assumes `weights` is a
# numeric vector of probabilities in [0, 1]. Multi-class / matrix-valued
# latent structures must supply `draw_z` to tulpa_em_laplace().
# ----------------------------------------------------------------------------
.draw_z_default <- function(weights) {
  if (!is.numeric(weights) || !is.null(dim(weights))) {
    stop("Default `draw_z` requires `weights` to be a numeric vector of ",
         "Bernoulli probabilities. Supply a custom `draw_z` callback for ",
         "multi-class or matrix-valued latent structures.",
         call. = FALSE)
  }
  p <- pmin(pmax(as.numeric(weights), 0), 1)
  rbinom(length(p), size = 1L, prob = p)
}


# ----------------------------------------------------------------------------
# Attach (beta, se) to a single block fit so it can be consumed by
# rubins_pool(). Pooling is over the fixed-effects block only -- RE values
# are nuisance and aren't comparable across draws (the RE encoding depends
# on the assembled block, not the latent variable).
#
# Shared with em_mc.R -- kept in this file because both correction paths
# and MCEM build their per-draw pool input the same way.
# ----------------------------------------------------------------------------
.attach_beta_se <- function(fit, n_fixed) {
  beta <- if (is.null(fit$mode) || length(fit$mode) < n_fixed) {
    numeric(0)
  } else {
    fit$mode[seq_len(n_fixed)]
  }
  h_dim <- if (is.null(fit$H_beta)) 0L else nrow(fit$H_beta)
  se <- if (length(beta) == n_fixed && n_fixed > 0L && h_dim >= n_fixed) {
    tryCatch(sqrt(diag(.spd_inverse(fit$H_beta))[seq_len(n_fixed)]),
             error = function(e) rep(NA_real_, n_fixed))
  } else {
    if (length(beta) == n_fixed && n_fixed > 0L && h_dim > 0L && h_dim < n_fixed) {
      warning(sprintf(
        "H_beta is %dx%d but n_fixed = %d; returning NA SEs. Model package likely returned a wrong-sized fixed-effects Hessian.",
        h_dim, h_dim, n_fixed),
        call. = FALSE)
    }
    rep(NA_real_, length(beta))
  }
  fit$beta <- beta
  fit$se <- se
  fit
}


# ----------------------------------------------------------------------------
# Fit one draw: encode + per-block validate + Laplace fit + attach (beta, se).
# `weights` here is the *user's weights object* (hard z, or continuous --
# this helper doesn't care, m_step_encode handles both forms).
# ----------------------------------------------------------------------------
.fit_one_draw <- function(weights, m_step_encode, label_prefix = "",
                          n_threads = 1L, beta_prior = NULL, ...) {
  blocks <- m_step_encode(weights, ...)
  if (!is.list(blocks) || length(blocks) == 0L) {
    stop("`m_step_encode` must return a non-empty list of blocks.",
         call. = FALSE)
  }
  block_names <- names(blocks)
  if (is.null(block_names)) {
    block_names <- paste0("submodel_", seq_along(blocks))
  }

  fits <- vector("list", length(blocks))
  names(fits) <- block_names
  for (k in seq_along(blocks)) {
    .validate_submodel_block(
      blocks[[k]],
      idx = if (nzchar(block_names[k])) {
        sprintf("%s'%s' (#%d)", label_prefix, block_names[k], k)
      } else {
        sprintf("%s#%d", label_prefix, k)
      })
    fit <- .fit_em_block(blocks[[k]], n_threads = n_threads,
                         beta_prior = beta_prior)
    fits[[k]] <- .attach_beta_se(fit, n_fixed = ncol(blocks[[k]]$X))
  }
  fits
}


# ----------------------------------------------------------------------------
# Multiple-imputation correction. Weights are fixed at the converged EM
# posterior; n_imputations independent hard draws yield n_imputations
# independent block refits, pooled via rubins_pool(). m_step_extra (if
# supplied) fires after each refit, matching the EM-loop semantics so
# dispersion / shape / precision updates also propagate into each draw.
# ----------------------------------------------------------------------------
.mi_correction <- function(weights, m_step_encode, draw_z, n_imputations,
                           m_step_extra, verbose, beta_prior = NULL, ...) {
  draws <- vector("list", n_imputations)
  for (m in seq_len(n_imputations)) {
    z <- draw_z(weights)
    fits <- .fit_one_draw(
      z, m_step_encode,
      label_prefix = sprintf("MI draw %d ", m),
      beta_prior = beta_prior, ...)
    if (!is.null(m_step_extra)) {
      fits <- apply_m_step_extra(m_step_extra, fits, weights, ...)
    }
    draws[[m]] <- fits
    if (verbose) {
      cat(sprintf("  MI draw %d / %d\n", m, n_imputations))
    }
  }
  list(pooled = rubins_pool(draws), draws = draws)
}


# ----------------------------------------------------------------------------
# Warm-started Gibbs correction. Each step:
#   1. E-step on the *current* fits to refresh weights (continuous P(z|y,th)).
#   2. Hard draw z from weights via draw_z.
#   3. M-step refit on z.
#   4. m_step_extra fires with the continuous E-step weights (NOT the hard
#      draw), matching the EM-loop semantics so the same callback can be
#      reused across phases.
# n_gibbs draws are pooled via Rubin's rules. Note these draws are
# autocorrelated; the within/between decomposition still gives a useful
# variance summary, but downstream callers can pull `draws` to do their
# own chain-level analysis (autocorrelation, thinning, ESS).
# ----------------------------------------------------------------------------
.gibbs_correction <- function(initial_fits, e_step, m_step_encode, draw_z,
                              n_gibbs, m_step_extra, verbose,
                              beta_prior = NULL, ...) {
  draws <- vector("list", n_gibbs)
  fits <- initial_fits

  for (g in seq_len(n_gibbs)) {
    e_result <- e_step(fits, ...)
    if (!is.list(e_result) || is.null(e_result$weights)) {
      stop("`e_step` must return a list with a `weights` element",
           call. = FALSE)
    }
    z <- draw_z(e_result$weights)
    fits <- .fit_one_draw(
      z, m_step_encode,
      label_prefix = sprintf("Gibbs step %d ", g),
      beta_prior = beta_prior, ...)
    if (!is.null(m_step_extra)) {
      fits <- apply_m_step_extra(m_step_extra, fits, e_result$weights, ...)
    }
    draws[[g]] <- fits
    if (verbose) {
      cat(sprintf("  Gibbs step %d / %d\n", g, n_gibbs))
    }
  }
  list(pooled = rubins_pool(draws), draws = draws, final_fits = fits)
}


# ----------------------------------------------------------------------------
# Louis' (1982) observed information at the EM estimate:
#   I_obs = E[-d2 l_c | y] - Var[d l_c | y].
# The first term is the block-diagonal M-step curvature H_beta at the
# converged weights; the second is the covariance of the complete-data
# fixed-effect score over hard draws z ~ draw_z(weights), each encoded by
# m_step_encode and scored at the estimate (no refit). The scores of blocks
# sharing a latent variable covary, so the result is one joint matrix.
# Defined for blocks whose fixed effects are their whole parameter.
# ----------------------------------------------------------------------------
.em_louis_information <- function(fits, blocks, weights, m_step_encode,
                                  draw_z, n_draws, ...) {
  decline <- function(reason) {
    list(vcov = NULL, se = NULL,
         info = list(n_draws = as.integer(n_draws), unavailable = reason))
  }
  nms <- names(fits)
  latent <- vapply(blocks, function(b) {
    !is.null(b$prior) || !is.null(b$spatial) || length(b$re_list) > 0L
  }, logical(1))
  if (any(latent)) {
    return(decline(paste0(
      "block(s) ", paste(nms[latent], collapse = ", "), " carry latent ",
      "structure (re_list / spatial / prior); the observed information is ",
      "computed for fixed-effect-only blocks")))
  }
  p <- vapply(blocks, function(b) ncol(b$X), integer(1))
  beta <- lapply(fits, `[[`, "beta")
  H <- lapply(seq_along(fits), function(k) fits[[k]]$H_beta)
  ok <- vapply(seq_along(fits), function(k) {
    length(beta[[k]]) == p[k] && is.matrix(H[[k]]) &&
      all(dim(H[[k]]) == p[k]) && all(is.finite(H[[k]]))
  }, logical(1))
  if (!all(ok)) {
    return(decline(paste0("block(s) ", paste(nms[!ok], collapse = ", "),
                          " returned no fixed-effect curvature H_beta")))
  }
  block_coefs <- lapply(seq_along(fits), function(k) {
    cn <- colnames(blocks[[k]]$X)
    if (is.null(cn)) cn <- character(p[k])
    unnamed <- !nzchar(cn)
    cn[unnamed] <- paste0("beta", seq_len(p[k]))[unnamed]
    cn
  })
  coef_names <- unlist(lapply(seq_along(fits), function(k) {
    paste0(nms[k], ":", block_coefs[[k]])
  }), use.names = FALSE)
  off <- c(0L, cumsum(p))
  P <- off[length(off)]

  score_draws <- matrix(NA_real_, n_draws, P)
  for (s in seq_len(n_draws)) {
    bl <- m_step_encode(draw_z(weights), ...)
    # A block with no design on this draw (no rows) has no width to compare.
    widths <- vapply(bl, function(b) if (is.null(b$X)) NA_integer_
                                     else as.integer(ncol(b$X)), integer(1))
    if (!identical(names(bl), names(blocks)) ||
        anyNA(widths) || !all(widths == p)) {
      return(decline(paste0("m_step_encode() returned a different block ",
                            "layout for a hard latent draw")))
    }
    for (k in seq_along(bl)) {
      b <- bl[[k]]
      eta <- drop(b$X %*% beta[[k]])
      if (!is.null(b$offset)) eta <- eta + b$offset
      n_tr <- if (is.null(b$n_trials)) rep(1, length(b$y)) else b$n_trials
      sc <- family_score_eta(eta, b$y, .canonical_family(b$family),
                             n_trials = n_tr, phi = b$phi %||% 1.0)
      w <- if (is.null(b$weights)) 1 else b$weights
      score_draws[s, (off[k] + 1L):off[k + 1L]] <- drop(crossprod(b$X, w * sc))
    }
  }
  if (!all(is.finite(score_draws))) {
    return(decline("a complete-data score was not finite at the estimate"))
  }

  I_complete <- matrix(0, P, P)
  for (k in seq_along(fits)) {
    idx <- (off[k] + 1L):off[k + 1L]
    I_complete[idx, idx] <- H[[k]]
  }
  I_missing <- stats::cov(score_draws)
  I_obs <- I_complete - I_missing
  I_obs <- (I_obs + t(I_obs)) / 2
  ev <- eigen(I_obs, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 0) {
    return(decline(sprintf(paste0(
      "the estimated observed information is not positive definite ",
      "(smallest eigenvalue %.3g); raise n_louis or use correction = \"mi\""),
      min(ev))))
  }
  V <- .spd_inverse(I_obs)
  dimnames(V) <- list(coef_names, coef_names)
  se_all <- sqrt(diag(V))
  se_complete <- sqrt(diag(.spd_inverse(I_complete)))
  se <- lapply(seq_along(fits), function(k) {
    stats::setNames(unname(se_all[(off[k] + 1L):off[k + 1L]]),
                    block_coefs[[k]])
  })
  names(se) <- nms
  mf <- 1 - se_complete^2 / se_all^2
  names(mf) <- coef_names
  list(vcov = V, se = se,
       info = list(n_draws = as.integer(n_draws), missing_fraction = mf))
}