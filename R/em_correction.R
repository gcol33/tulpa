# ============================================================================
# Post-EM corrections for tulpa_em_laplace().
#
# Two routines, both run after EM convergence:
#
#   * MI    — draw `n_imputations` independent hard z's from the converged
#             posterior weights P(z|y, theta_hat), refit each, pool via
#             Rubin's rules.
#   * Gibbs — warm-started Markov chain z|theta -> theta|z. Each step draws
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
# rubins_pool(). Pooling is over the fixed-effects block only — RE values
# are nuisance and aren't comparable across draws (the RE encoding depends
# on the assembled block, not the latent variable).
#
# Shared with em_mc.R — kept in this file because both correction paths
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
    tryCatch(sqrt(diag(solve(fit$H_beta))[seq_len(n_fixed)]),
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
# `weights` here is the *user's weights object* (hard z, or continuous —
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
