# The one-arm joint nested-Laplace route of tulpa(): a single response fitted on
# the joint door, whose outer grid carries a per-arm dispersion axis.
#
# Two kinds of call reach it. A censored / truncated Gaussian response
# (`y | cens()`, `y | trunc()`) has no other door: its kernels read a
# per-observation bound through the joint arm's response payload. And a family
# whose `phi` is a Gaussian residual variance, fitted with `phi` left out on a
# mode that integrates what the caller did not state: a caller who passed no
# `phi` chose no value, so the residual variance is integrated on the outer
# grid next to the random-effect and field scales, the way an unsupplied
# random-effect scale is (gcol33/tulpa#787, #1068).

# The arm's name on the joint door, which names its dispersion axis `phi_y`.
.JOINT_ARM_NAME <- "y"

# The modes that integrate an unsupplied dispersion on this route. A tier mode
# integrates what the caller did not state; the nested-Laplace modes name the
# integrating path itself.
.JOINT_ARM_PHI_MODES <- c("auto", "structured", "nested_laplace",
                          "nested_laplace_joint")

# The modes a bounded response is fitted under: the above, plus the
# conditional `laplace` mode, which on this route conditions on the scales the
# caller supplied and integrates the rest.
.JOINT_ARM_BOUND_MODES <- c(.JOINT_ARM_PHI_MODES, "laplace")

# Is the residual variance of this call integrated as an outer axis? The
# families whose `phi` is a Gaussian residual variance, under their own
# identity link on the scale the variance is read (log y for lognormal).
#' @keywords internal
.phi_axis_wanted <- function(family, phi_supplied, estimate_phi) {
  !isTRUE(phi_supplied) && !isTRUE(estimate_phi) &&
    isTRUE(family %in% .VARIANCE_PHI_FAMILIES)
}

# Why the joint door cannot carry this call, or NULL when it can. Each entry
# names a feature the one-arm joint fit has no channel for; a call carrying one
# keeps the backend it selected.
#' @keywords internal
.joint_arm_unfit <- function(spatial_type = NULL, temporal_spec = NULL,
                             st_block = NULL, has_latent = FALSE,
                             zi = FALSE, weights = FALSE, phi2 = FALSE) {
  sp <- tolower(spatial_type %||% "")
  tt <- tolower(temporal_spec$type %||% "")
  if (zi) return("a zero-inflation formula")
  if (weights) return("observation weights")
  if (phi2) return("a second dispersion (`phi2`)")
  if (nzchar(sp) && !sp %in% c(.NL_FRONTDOOR_AREAL, "nngp", "gp", "hsgp")) {
    return(sprintf("a '%s' spatial field", sp))
  }
  if (nzchar(tt) && !tt %in% c("rw1", "rw2", "ar1", "multiscale")) {
    return(sprintf("a '%s' temporal field", tt))
  }
  if (!is.null(temporal_spec$group_var)) return("a grouped (panel) temporal field")
  if (!is.null(st_block)) return("a spatiotemporal() interaction")
  if (isTRUE(has_latent)) return("latent() blocks")
  NULL
}

# The formula random-effect terms as nested-Laplace latent blocks, each scale an
# outer axis:
#   * `(1 | g)`: an `iid` block over the groups;
#   * a correlated slope term `(1 + x | g)`: one `miid` block, the coefficients'
#     covariance integrated over its log-Cholesky axes;
#   * an uncorrelated term `(1 + x || g)` / `(0 + x | g)`: one `iid` block per
#     coefficient, a slope's carrying its covariate as the per-row weight.
# `sigma_re`, one value per term, conditions a scalar `(1 | g)` term as a
# one-point SD grid; a term with slopes has a covariance, not one SD, and
# refuses it. Returns one list of blocks per term, in term order.
#' @keywords internal
.re_terms_to_nl_blocks <- function(re, n_obs, sigma_re = NULL) {
  if (!length(re)) return(list())
  if (!is.null(sigma_re)) {
    sigma_re <- as.numeric(sigma_re)
    if (length(sigma_re) == 1L) sigma_re <- rep(sigma_re, length(re))
    if (length(sigma_re) != length(re)) {
      stop(sprintf("`sigma_re` must have length 1 or %d (one per RE term).",
                   length(re)), call. = FALSE)
    }
  }
  lapply(seq_along(re), function(m) {
    rt <- re[[m]]
    idx <- as.integer(rt$group_idx)
    ng  <- as.integer(rt$n_groups)
    cols <- c(if (isTRUE(rt$has_intercept)) list(rep(1, n_obs)),
              if (!is.null(rt$slope_matrix))
                lapply(seq_len(ncol(rt$slope_matrix)),
                       function(j) as.numeric(rt$slope_matrix[, j])))
    scalar <- .is_scalar_re_intercept(rt)
    if (!is.null(sigma_re) && !scalar) {
      stop(sprintf(paste0(
        "`sigma_re` conditions a scalar `(1 | g)` term; RE term %d ",
        "(`%s`) carries slopes, whose covariance is integrated. Leave ",
        "`sigma_re` out."), m, rt$group_var), call. = FALSE)
    }
    if (scalar) {
      blk <- list(type = "iid", obs_idx = idx, n_units = ng)
      if (!is.null(sigma_re)) blk$sigma_grid <- sigma_re[m]
      return(list(blk))
    }
    if (isTRUE(rt$correlated) && length(cols) > 1L) {
      return(list(list(type = "miid", obs_idx = idx, n_groups = ng,
                       n_fields = length(cols),
                       field_weight = lapply(cols, function(w) list(w)))))
    }
    lapply(seq_along(cols), function(j) {
      blk <- list(type = "iid", obs_idx = idx, n_units = ng)
      if (!(isTRUE(rt$has_intercept) && j == 1L)) {
        blk$svc_weight <- list(cols[[j]])
      }
      blk
    })
  })
}

# The default outer axis of an unsupplied residual variance, marked as the
# engine's default so the placement pass lays it at the posterior. The nodes
# start from the residual variance of the least-squares fit of the fixed
# effects, which bounds the residual variance from above whatever else the
# model carries, and reach well below it: random effects and fields take their
# share of that variance from the residual. A lognormal's variance is the one
# of log y.
#' @keywords internal
.phi_axis_default <- function(family, y, X, offset = NULL) {
  if (identical(family, "lognormal")) y <- log(y)
  ok <- !is.na(y)
  yy <- y[ok] - if (is.null(offset)) 0 else offset[ok]
  XX <- X[ok, , drop = FALSE]
  r  <- stats::lm.fit(XX, yy)$residuals
  dfree <- max(sum(ok) - ncol(XX), 1L)
  v0 <- sum(r^2) / dfree
  if (!is.finite(v0) || v0 <= 0) v0 <- max(stats::var(yy), 1)
  auto_grid(v0 * exp(seq(-4, 1, length.out = 9L)))
}

# The argument list of the one-arm joint fit. `blocks` are the latent blocks of
# the formula's fields, smoothers and latent terms; the random-effect terms
# join them last, the order every nested accessor reads them in.
#' @keywords internal
.joint_arm_args <- function(bundle, family, blocks, sigma_re, n_trials, phi,
                            phi_axis, beta_prior, hyperprior, control) {
  N <- bundle$n_obs
  by_term <- .re_terms_to_nl_blocks(bundle$re_terms %||% list(), N, sigma_re)
  re_blocks <- unlist(by_term, recursive = FALSE)
  all_blocks <- c(blocks, re_blocks)
  bp <- .normalize_beta_prior(beta_prior, ncol(bundle$X))
  arm <- c(list(
    y        = bundle$y,
    X        = bundle$X,
    family   = family,
    phi      = if (is.null(phi_axis)) phi else stats::median(phi_axis),
    n_trials = n_trials %||% rep(1L, N),
    offset   = bundle$offset,
    beta_prior_mean = bp$mean,
    beta_prior_prec = 1 / bp$sd^2
  ), .response_bound_arm_fields(bundle$response_bounds))
  out <- list(
    responses  = stats::setNames(list(arm), .JOINT_ARM_NAME),
    prior      = all_blocks,
    phi_grid   = if (!is.null(phi_axis))
                   stats::setNames(list(phi_axis), .JOINT_ARM_NAME),
    hyperprior = hyperprior,
    control    = .control_subset(control, .CONTROL_KEYS$nested_laplace_joint,
                                 "nested_laplace_joint")
  )
  out$control$progress <- out$control$progress %||% FALSE
  # Each cell's precision is kept, which the per-group random effects are read
  # off (`.joint_arm_attach_re()`).
  if (length(re_blocks)) out$control$store_Q <- TRUE
  if (length(re_blocks)) {
    # Which blocks each term became: one per term except an uncorrelated
    # slope term, which is one per coefficient.
    first <- length(blocks) + cumsum(c(0L, lengths(by_term)))[seq_along(by_term)]
    attr(out, "re_block_index") <- lapply(seq_along(by_term), function(m)
      first[m] + seq_along(by_term[[m]]))
    attr(out, "re_block_conditioned") <- !is.null(sigma_re)
  }
  out
}

# The per-group random effects of a one-arm joint fit, retained per outer cell
# as `grid_re_mean` / `grid_re_var` (cells x effects), the shape ranef() and the
# group-level predictions read. Effects are ordered as the fit's random-effect
# layout names them: term by term, group by group, coefficient by coefficient.
# An `iid` block is non-centred, its effect the coordinate times the cell's
# `sigma`; an `miid` block holds the coefficients themselves, field-major (all
# groups of the first coefficient, then the second, ...).
#' @keywords internal
.joint_arm_attach_re <- function(fit, blocks, re_block_index, re_terms) {
  if (!length(re_block_index)) return(fit)
  starts <- fit$arm_layout$block_start
  tg <- fit$theta_grid
  Q_p <- fit$Q_csc_p_per_grid
  if (is.null(starts) || is.null(Q_p) || !is.matrix(fit$modes)) return(fit)
  cols <- integer(0)
  scale_block <- integer(0)   # the iid block whose sigma loads the effect, 0 = none
  for (m in seq_along(re_terms)) {
    rt <- re_terms[[m]]
    bi <- re_block_index[[m]]
    ng <- as.integer(rt$n_groups)
    nc <- as.integer(rt$n_coefs %||% 1L)
    miid <- length(bi) == 1L && identical(blocks[[bi]]$type, "miid")
    for (g in seq_len(ng)) for (j in seq_len(nc)) {
      if (miid) {
        cols <- c(cols, starts[bi] + (j - 1L) * ng + g)
        scale_block <- c(scale_block, 0L)
      } else {
        cols <- c(cols, starts[bi[j]] + g)
        scale_block <- c(scale_block, bi[j])
      }
    }
  }
  sigma_at <- function(k) {
    vapply(scale_block, function(b) {
      if (b == 0L) return(1)
      nm <- sprintf("b%d.sigma", b)
      if (is.matrix(tg) && nm %in% colnames(tg)) return(tg[k, nm])
      as.numeric(blocks[[b]]$sigma_grid)[1L]
    }, numeric(1))
  }
  load <- list(type = "iid", n_units = length(cols), cols = list(cols),
               weights = function(k) list(sigma_at(k)))
  fit <- .nl_attach_unit_loads(fit, list(re = load), Q_p, fit$Q_csc_i_per_grid,
                               fit$Q_csc_x_per_grid, fit$Q_csc_n,
                               fit$constraint_cols)
  # The per-node Gaussian of the group effects, under the names the
  # group-level predictions read it by on every nested random-effect fit.
  fit$re_nodes     <- fit$grid_re_mean
  fit$re_var_nodes <- fit$grid_re_var
  fit
}

# The dispersion a one-arm joint fit reports flat, as every tulpa() fit does:
# where the outer grid integrated the arm's `phi_y` axis, its posterior mean
# (in the R-level convention the axis is laid in), flagged as estimated rather
# than conditioned on; otherwise the `phi` it was conditioned on. Draw-level
# readers take the axis itself (`.tulpa_phi_draws()`).
#' @keywords internal
.joint_arm_dispersion <- function(fit, phi) {
  col <- paste0("phi_", .JOINT_ARM_NAME)
  tg <- fit$theta_grid
  integrated <- is.matrix(tg) && col %in% colnames(tg) &&
    length(unique(tg[, col])) > 1L
  if (integrated) {
    fit$phi <- unname(fit$theta_mean[col])
    fit$phi_estimated <- TRUE
  } else {
    fit$phi <- phi
  }
  fit
}
