# Posterior-moment machinery for the nested-Laplace fits: weighted mean / SD /
# median / CI over the outer hyperparameter grid, per-axis weighted quantiles,
# the axis-marginal Laplace-refined SD, and the multi-block joint / per-block
# marginal moments. Consumed by the nested-Laplace drivers in nested_laplace.R.

# Compute weighted theta_mean / theta_sd / theta_median / theta_ci_lo /
# theta_ci_hi from grid + weights. The mean and SD are produced via the
# usual `sum(w * x)` / `sum(w * x^2)` moments; the median and 2.5/97.5
# CI are produced via `.nl_axis_quantiles` so that asymmetric scale-like
# hyperparameters (sigma, alpha, range, phi) have a calibrated headline
# summary alongside mean +/- SD. Median is the recommended summary for
# right-skewed posteriors: `mean` of a weakly-identified positive ratio
# is pulled up by the right tail; `median` matches truth at small n.
.nl_posterior_moments <- function(res, type) {
  w <- res$weights
  tg <- res$theta_grid
  if (is.matrix(tg)) {
    res$theta_mean <- as.numeric(crossprod(w, tg))
    names(res$theta_mean) <- colnames(tg)
    res$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                                  res$theta_mean^2))
    names(res$theta_sd) <- colnames(tg)
  } else {
    ms <- .nl_wtd_mean_sd(tg, w)
    res$theta_mean <- ms$mean
    res$theta_sd   <- ms$sd
  }
  qs <- .nl_axis_quantiles(tg, res$log_marginal, res$refining_axis)
  res$theta_median <- qs$median
  res$theta_ci_lo  <- qs$ci_lo
  res$theta_ci_hi  <- qs$ci_hi
  res
}

# Marginal log-density along a single hyperparameter axis (logsumexp over
# the other-axis cells at each unique value). `vals` and `log_marg` are
# length n_cells; `keep` is an optional logical mask (cartesian + same-
# axis slice cells). Returns sorted unique axis values and the matching
# marginal log-density.
# Weighted quantile on a discrete (value, weight) distribution. Uses
# midpoint-of-mass cumulative probability (Type 7-like) plus linear
# interpolation, so quantiles vary smoothly with weights rather than
# snapping to grid cells.
#
#  * Aggregates duplicate values: weights at equal `values` are summed
#    before interpolation. Idempotent on already-unique axes; needed when
#    the joint grid carries repeated values (e.g. slice-cell refinement
#    re-uses the modal axis value across multiple Newton-Laplace cells).
#  * Filters non-finite values and non-positive weights.
#  * Returns `NA` per requested `probs` when the support is empty;
#    returns the unique support value when only one survives.
#
# Used by `.alpha_grid_moments` to surface the posterior median and 95%
# interval of `alpha` (a direct hyperparameter axis after the
# (sigma, alpha) reparameterization) from the joint nested-Laplace
# posterior. Weighted quantiles handle skewed/heavy-tailed marginals that
# `mean +/- 1.96 sd` summaries misrepresent.
.nl_wtd_quantile <- function(values, weights, probs) {
  ord <- order(values)
  v <- as.numeric(values)[ord]
  w <- as.numeric(weights)[ord]
  keep <- is.finite(v) & is.finite(w) & w > 0
  if (!any(keep)) return(rep(NA_real_, length(probs)))
  v <- v[keep]; w <- w[keep]
  # Aggregate runs of strictly-equal adjacent values (already sorted).
  # Cannot use factor(v) here: distinct doubles that share an
  # `as.character` print form (e.g. 0.4/0.7 and a near-equal ratio off
  # by ~1e-16) trigger "factor level [k] is duplicated". Group by
  # integer run-IDs derived from numeric equality on the sorted vector.
  if (length(v) > 1L) {
    is_first <- c(TRUE, v[-1L] != v[-length(v)])
    if (!all(is_first)) {
      grp <- cumsum(is_first)
      w   <- as.numeric(tapply(w, grp, sum))
      v   <- v[is_first]
    }
  }
  w_tot <- sum(w)
  if (!is.finite(w_tot) || w_tot <= 0) return(rep(NA_real_, length(probs)))
  w <- w / w_tot
  if (length(v) == 1L) return(rep(v[1L], length(probs)))
  p <- cumsum(w) - w / 2
  # `approx` emits "collapsing to unique 'x' values" when tiny floor
  # weights against a dominant prefix sum produce numerically equal
  # midpoints. The collapse (tied p -> mean of v) is the right behavior
  # at that resolution -- the underlying mass between the tied cells is
  # zero up to floating-point error. Suppress the noise.
  vapply(probs, function(q) {
    if (q <= p[1L])          return(v[1L])
    if (q >= p[length(p)])   return(v[length(v)])
    suppressWarnings(approx(p, v, xout = q, method = "linear")$y)
  }, numeric(1L))
}

# Weighted mean and SD of `values` under pre-normalized `weights` (which
# sum to 1). SD uses the E[x^2] - E[x]^2 form, floored at 0 to absorb the
# floating-point negatives that form can produce near a degenerate axis.
.nl_wtd_mean_sd <- function(values, weights) {
  mu <- sum(weights * values)
  list(mean = mu, sd = sqrt(max(0, sum(weights * values^2) - mu^2)))
}

# Weighted-quantile median + 2.5/97.5 empirical CI for every axis of a
# (scalar or matrix) theta_grid. Returns named numeric vectors so the
# joint nested-Laplace surface exposes `theta_median / theta_ci_lo /
# theta_ci_hi` alongside `theta_mean / theta_sd` for every hyperparameter.
#
# `tg` is a vector or matrix; `log_marginal` aligns with `tg` rows;
# `refining` is the per-cell refining-axis tag from mode-tracked
# refinement (NULL or all-"" outside the joint path). For each axis,
# slice cells from OTHER axes are dropped before computing the quantile
# -- those cells pin the current axis at a single non-varying value, so
# including them oversamples that value. Cartesian cells, same-axis
# slice cells, and same-axis consistency cells are kept. This is the
# same per-axis mask used by `.joint_recalibrate_axis_moments` for the
# mean/SD path.
#
# Returns list(median = named_vec, ci_lo = named_vec, ci_hi = named_vec).
# For scalar tg the returned vectors are length-1 with names = "value".
.nl_axis_quantiles <- function(tg, log_marginal, refining = NULL,
                                probs = c(0.025, 0.5, 0.975),
                                weights = NULL) {
  if (is.null(dim(tg))) {
    tg <- matrix(as.numeric(tg), ncol = 1L,
                 dimnames = list(NULL, "value"))
  }
  # Empty grid (no outer-grid axes -- e.g. an lf-only fit): nothing to
  # quantilize; return empty named vectors. paste0 recycles the prefix
  # past zero-length integers, so we guard ncol(tg) == 0 explicitly.
  if (ncol(tg) == 0L) {
    empty <- setNames(numeric(0), character(0))
    return(list(median = empty, ci_lo = empty, ci_hi = empty))
  }
  nms <- colnames(tg) %||% paste0("V", seq_len(ncol(tg)))
  n_ax <- length(nms)
  lo  <- setNames(rep(NA_real_, n_ax), nms)
  med <- setNames(rep(NA_real_, n_ax), nms)
  hi  <- setNames(rep(NA_real_, n_ax), nms)
  if (is.null(refining)) refining <- rep("", nrow(tg))
  for (j in seq_len(n_ax)) {
    ax    <- nms[j]
    keep  <- refining == "" | refining == ax |
             refining == paste0("consistency_", ax)
    use   <- keep & is.finite(tg[, j])
    if (sum(use) == 0L) next
    # Precomputed integration weights (CCD design weights * exp(log-marginal),
    # passed for scattered node sets where the per-axis softmax of the raw
    # log-marginal is not the integration weight); otherwise the regular-grid
    # softmax of the log-marginal.
    if (!is.null(weights)) {
      ws  <- weights[use]
      if (!any(is.finite(ws)) || sum(ws, na.rm = TRUE) <= 0) next
      ws[!is.finite(ws)] <- 0
      ws  <- ws / sum(ws)
    } else {
      lm_u  <- log_marginal[use]
      m     <- max(lm_u)
      if (!is.finite(m)) next
      ws    <- exp(lm_u - m); ws <- ws / sum(ws)
    }
    qs    <- .nl_wtd_quantile(tg[use, j], ws, probs)
    lo[j]  <- qs[1L]
    med[j] <- qs[2L]
    hi[j]  <- qs[3L]
  }
  list(median = med, ci_lo = lo, ci_hi = hi)
}

.nl_axis_marginal_logdensity <- function(vals, log_marg, keep = NULL) {
  if (is.null(keep)) keep <- rep(TRUE, length(vals))
  v <- vals[keep]; l <- log_marg[keep]
  uv <- sort(unique(v))
  if (length(uv) == 0L) return(list(vals = uv, log_marg = numeric(0)))
  lm <- vapply(uv, function(u) {
    li <- l[v == u]
    if (length(li) == 0L) return(-Inf)
    m <- max(li)
    if (!is.finite(m)) return(-Inf)
    m + log(sum(exp(li - m)))
  }, numeric(1))
  list(vals = uv, log_marg = lm)
}

# Laplace-at-mode SD on a single axis. Fits a 3-point parabola to the
# marginal log-density at the modal cell and one neighbour on each side,
# returns sqrt(-1 / (2 a)) from the quadratic coefficient. When the
# axis values are positive (sigma / phi / tau) the parabola is fit on
# log(theta), and the SD is mapped back to the linear axis via the
# delta method (sigma_theta = theta_mode * sigma_log_theta). Returns NA
# when the mode sits at an axis edge or the parabola is concave up.
#
# `return_log_sd = TRUE` skips the delta back-map and returns sd(log theta)
# directly. Only meaningful with `log_axis = TRUE` -- returns NA otherwise.
.nl_laplace_at_mode_sd_axis <- function(vals, log_marg, log_axis = NULL,
                                        return_log_sd = FALSE) {
  if (length(vals) < 3L) return(NA_real_)
  ix <- which.max(log_marg)
  if (ix == 1L || ix == length(vals)) return(NA_real_)
  if (is.null(log_axis)) log_axis <- all(is.finite(vals)) && all(vals > 0)
  u <- if (log_axis) log(vals) else vals
  dm <- u[ix - 1L] - u[ix]
  dp <- u[ix + 1L] - u[ix]
  det <- dm * dp * (dm - dp)
  if (!is.finite(det) || abs(det) < .Machine$double.eps) return(NA_real_)
  lm_m <- log_marg[ix - 1L] - log_marg[ix]
  lm_p <- log_marg[ix + 1L] - log_marg[ix]
  a <- (lm_m * dp - lm_p * dm) / det
  if (!is.finite(a) || a >= 0) return(NA_real_)
  sd_u <- sqrt(-1 / (2 * a))
  if (return_log_sd) {
    if (log_axis) sd_u else NA_real_
  } else {
    if (log_axis) vals[ix] * sd_u else sd_u
  }
}

# Replace `theta_sd` (and `block_moments[[b]]$sd` when present) entries
# with the Laplace-at-mode SD wherever the 3-point fit succeeds. Axes
# with the mode at an edge or wrong-signed curvature keep their var-of-
# means SD. Grid-spacing-independent: fixes the symptom where a sharply
# peaked marginal log-likelihood on a coarse grid collapses var-of-
# means to ~0 and undercovers (gcol33/tulpa#20).
.nl_refit_axis_sd_laplace <- function(res, refining = NULL) {
  if (is.null(res$theta_grid) || is.null(res$log_marginal)) return(res)
  tg <- res$theta_grid
  if (!is.matrix(tg)) {
    marg <- .nl_axis_marginal_logdensity(as.numeric(tg), res$log_marginal)
    sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
    if (is.finite(sd_lam)) res$theta_sd <- sd_lam
    return(res)
  }
  if (is.null(refining)) refining <- res$refining_axis
  if (is.null(refining)) refining <- rep("", nrow(tg))
  col_names <- colnames(tg)
  if (!is.null(col_names) && !is.null(res$theta_sd)) {
    for (col in col_names) {
      if (!col %in% names(res$theta_sd)) next
      keep <- refining == "" | refining == col |
              refining == paste0("consistency_", col)
      marg <- .nl_axis_marginal_logdensity(tg[, col], res$log_marginal, keep)
      sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
      if (is.finite(sd_lam)) res$theta_sd[[col]] <- sd_lam
    }
  }
  if (!is.null(res$block_moments)) {
    for (b in seq_along(res$block_moments)) {
      bm <- res$block_moments[[b]]
      axis_cols <- bm$axis_cols
      if (is.null(axis_cols) || length(axis_cols) == 0L) next
      bare <- names(bm$sd)
      for (j in seq_along(axis_cols)) {
        col_ix <- axis_cols[j]
        col_name <- if (!is.null(col_names)) col_names[col_ix] else ""
        keep <- refining == "" | refining == col_name |
                refining == paste0("consistency_", col_name)
        marg <- .nl_axis_marginal_logdensity(tg[, col_ix], res$log_marginal,
                                              keep)
        sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
        if (is.finite(sd_lam)) res$block_moments[[b]]$sd[[j]] <- sd_lam
      }
    }
  }
  res
}

# Posterior moments for multi-block grids. Two flavours:
#  * joint moments: across all axes (same as single-block 2D scatter).
#  * per-block marginal moments: integrate out the other blocks' axes.
.nl_posterior_moments_multi <- function(out, prepared, axis_offsets, joint_grid) {
  w  <- out$weights
  tg <- joint_grid
  out$theta_mean <- as.numeric(crossprod(w, tg))
  names(out$theta_mean) <- colnames(tg)
  out$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                              out$theta_mean^2))
  names(out$theta_sd) <- colnames(tg)

  # Per-block marginals: for each block, sum weights over rows that share the
  # same per-block axis values, then take weighted mean/sd within those rows.
  # Because the joint grid is a Cartesian product, the "rows that share this
  # block's values" form n_block_rows groups indexed by idx[, b]. The marginal
  # weight for group g is sum of joint weights over its rows.
  B <- length(prepared)
  per_block_moments <- vector("list", B)
  for (b in seq_len(B)) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    sub <- joint_grid[, cols, drop = FALSE]
    block_mean <- as.numeric(crossprod(w, sub))
    block_sd   <- sqrt(pmax(0, as.numeric(crossprod(w, sub^2)) - block_mean^2))
    # Use bare per-block axis names (e.g. "tau", "rho") instead of the
    # joint-grid-prefixed "b<N>.tau" -- the block index is already implicit
    # in the list position.
    bare_names <- .nl_block_axis_grid(prepared[[b]])$names
    names(block_mean) <- bare_names
    names(block_sd)   <- bare_names
    per_block_moments[[b]] <- list(
      type      = tolower(prepared[[b]]$type),
      mean      = block_mean,
      sd        = block_sd,
      axis_cols = cols
    )
  }
  out$block_moments <- per_block_moments

  # Weighted-quantile median + 2.5/97.5 CI per axis (calibrated summary
  # for right-skewed scale-like hyperparameters; see `.nl_axis_quantiles`).
  qs <- .nl_axis_quantiles(joint_grid, out$log_marginal, out$refining_axis)
  out$theta_median <- qs$median
  out$theta_ci_lo  <- qs$ci_lo
  out$theta_ci_hi  <- qs$ci_hi
  out
}
