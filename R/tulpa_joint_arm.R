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
  all_blocks <- c(lapply(blocks, .joint_arm_hsgp_block, n_obs = N), re_blocks)
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
  # With a block, the joint door keeps each cell's precision by default: the
  # per-group random effects are read off it (`.joint_arm_attach_re()`), and
  # the latent posterior is drawn from it at new rows
  # (`.joint_arm_attach_field_read()`).
  if (length(re_blocks)) {
    attr(out, "re_block_index") <- .re_block_index(length(blocks), by_term)
    attr(out, "re_block_conditioned") <- !is.null(sigma_re)
  }
  out
}

# Which blocks each random-effect term became when the terms' blocks
# (`.re_terms_to_nl_blocks()`, one list per term) follow `n_before` other
# blocks: one per term, except an uncorrelated slope term, which is one per
# coefficient.
#' @keywords internal
.re_block_index <- function(n_before, by_term) {
  first <- n_before + cumsum(c(0L, lengths(by_term)))[seq_along(by_term)]
  lapply(seq_along(by_term), function(m) first[m] + seq_along(by_term[[m]]))
}

# The random-effect record of a one-arm joint fit whose last blocks are the
# formula's random-effect terms: where each term's blocks sit, whether the
# scales were conditioned, the per-group effects retained per cell
# (`.joint_arm_attach_re()`) and the map the draw path at new rows reads them
# by (`.joint_arm_attach_field_read()`).
#' @keywords internal
.joint_arm_attach_re_record <- function(fit, blocks, re_block_index,
                                        re_block_conditioned, re_terms) {
  fit$re_block_index       <- re_block_index
  fit$re_block_conditioned <- re_block_conditioned
  fit <- .joint_arm_attach_re(fit, blocks, re_block_index, re_terms)
  .joint_arm_attach_field_read(
    fit, .joint_arm_re_map(fit, blocks, re_block_index, re_terms))
}

# An HSGP block of the registry path (the basis at every observation and its
# eigenvalues) in the form the joint door reads, the basis of its one arm; the
# coordinate extent the lengthscale prior is anchored on rides along. Every
# other block is returned as it is.
#' @keywords internal
.joint_arm_hsgp_block <- function(blk, n_obs) {
  if (!identical(tolower(blk$type %||% ""), "hsgp") || is.null(blk$phi_basis)) {
    return(blk)
  }
  B <- as.matrix(blk$phi_basis)
  blk$phi <- list(B)
  blk$eigenvalues <- as.numeric(blk$lambda_eig)
  blk$m_total <- ncol(B)
  blk$n_obs_per_arm <- as.integer(n_obs)
  blk$phi_basis <- NULL
  blk$lambda_eig <- NULL
  blk
}

# Where each random effect of a one-arm joint fit sits in its latent vector, in
# the order the fit's random-effect layout names them (term by term, group by
# group, coefficient by coefficient), and the `iid` block whose cell `sigma`
# scales it (0 for an `miid` coefficient, which is the effect itself). NULL for
# a fit without random-effect blocks.
#' @keywords internal
.joint_arm_re_map <- function(fit, blocks, re_block_index, re_terms) {
  starts <- fit$arm_layout$block_start
  if (!length(re_block_index) || is.null(starts)) return(NULL)
  cols <- integer(0)
  scale_block <- integer(0)
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
  sigma_fixed <- vapply(scale_block, function(b)
    if (b == 0L) 1 else as.numeric(blocks[[b]]$sigma_grid %||% NA_real_)[1L],
    numeric(1))
  list(cols = as.integer(cols), scale_block = as.integer(scale_block),
       sigma_fixed = sigma_fixed)
}

# The factor turning cell k's random-effect coordinates into effects under a
# map from `.joint_arm_re_map()`: the cell's `b<j>.sigma` for an `iid`
# coordinate (its block's fixed SD where the grid carries no such axis), 1 for
# an `miid` one.
#' @keywords internal
.joint_arm_re_scale <- function(map, theta_grid, k) {
  tg <- theta_grid
  vapply(seq_along(map$scale_block), function(i) {
    b <- map$scale_block[i]
    if (b == 0L) return(1)
    nm <- sprintf("b%d.sigma", b)
    if (is.matrix(tg) && nm %in% colnames(tg)) return(tg[k, nm])
    map$sigma_fixed[i]
  }, numeric(1))
}

# The record the one draw path at new rows reads a one-arm joint fit through
# (`.field_source_grid()`): where each latent block starts (the multi-block
# offsets `.field_layout()` locates a field by), the stored per-cell precision
# as the solve to draw from, and the random effects' map.
#' @keywords internal
.joint_arm_attach_field_read <- function(fit, re_map) {
  al <- fit$arm_layout
  if (is.null(al$block_start) || is.null(fit$Q_csc_p_per_grid)) return(fit)
  st <- as.integer(al$block_start)
  sz <- as.integer(al$block_size)
  if (length(st)) {
    fit$block_latent_offsets <- c(st, ncol(fit$modes))
    attr(fit$block_latent_offsets, "sizes") <- sz
  }
  fit$field_solve <- list(door = "stored")
  fit$re_effect_map <- re_map
  fit
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
  .joint_attach_re_cells(fit, .joint_arm_re_map(fit, blocks, re_block_index,
                                                re_terms))
}

# A joint fit's per-cell random-effect Gaussian under `map`, retained as
# `grid_re_mean` / `grid_re_var` and, under the names the group-level
# predictions read it by on every nested random-effect fit, `re_nodes` /
# `re_var_nodes`.
#' @keywords internal
.joint_attach_re_cells <- function(fit, map) {
  rc <- .joint_re_cells(fit, map)
  if (is.null(rc)) return(fit)
  fit$grid_re_mean <- fit$re_nodes <- rc$mean
  fit$grid_re_var <- fit$re_var_nodes <- rc$var
  fit
}

# The random effects a map locates (`cols`, the latent column of each effect;
# `.joint_arm_re_scale()` the factor turning a cell's coordinate into the
# effect) as one Gaussian per outer cell: `mean` and `var` (cells x effects).
# The variance is the cell's precision read along the effect's loading and
# conditioned on the fit's sum-to-zero groups, as its latent draws are
# (`.field_cell_solve()`); a fit that kept no precision carries the means
# alone (`var` NULL). NULL for a fit without such effects.
#' @keywords internal
.joint_re_cells <- function(fit, map) {
  M <- fit$modes
  if (is.null(map) || !length(map$cols) || !is.matrix(M)) return(NULL)
  tg <- fit$theta_grid
  load <- list(type = "iid", n_units = length(map$cols), cols = list(map$cols),
               weights = function(k) list(.joint_arm_re_scale(map, tg, k)))
  Q_p <- fit$Q_csc_p_per_grid
  if (is.null(Q_p)) {
    mu <- t(vapply(seq_len(nrow(M)), function(k)
      load$weights(k)[[1L]] * M[k, map$cols], numeric(length(map$cols))))
    return(list(mean = mu, var = NULL))
  }
  n_x <- ncol(M)
  r <- .nl_attach_unit_loads(list(modes = M), list(re = load), Q_p,
                             fit$Q_csc_i_per_grid, fit$Q_csc_x_per_grid, n_x,
                             .inner_constraint(fit))
  list(mean = r$grid_re_mean, var = r$grid_re_var)
}

# The random effects of a fit made through tulpa_nested_laplace_joint()
# directly, in latent order with their names: each `iid` block's units
# (`<block>[<unit>]`, non-centred, the coordinate times the cell's
# `b<k>.sigma`), each `miid` block's coefficients (`<block>.<field>[<group>]`,
# field-major, the coefficients themselves), and each arm's own random
# intercepts (`<arm>.re[<group>]`, conditioned at the arm's `sigma_re`, the
# coordinates themselves). A block is named by its `name`, else `b<k>`. NULL
# when the fit carries none.
#' @keywords internal
.joint_direct_re_map <- function(fit) {
  al <- fit$arm_layout
  blocks <- .field_block_list(fit$prior)
  starts <- al$block_start
  roles <- .nl_block_roles(blocks)
  cols <- integer(0); scale_block <- integer(0); sigma_fixed <- numeric(0)
  nm <- character(0)
  add <- function(c, b, s, n) {
    cols <<- c(cols, as.integer(c)); scale_block <<- c(scale_block, rep(b, length(c)))
    sigma_fixed <<- c(sigma_fixed, rep(s, length(c))); nm <<- c(nm, n)
  }
  if (length(starts) == length(blocks)) {
    for (k in seq_along(blocks)) {
      blk <- blocks[[k]]
      type <- tolower(blk$type %||% "")
      lab <- blk$name %||% paste0("b", k)
      if (type == "iid" && roles[k] == "other") {
        n <- as.integer(blk$n_units)
        add(starts[k] + seq_len(n), k,
            as.numeric(blk$sigma_grid %||% NA_real_)[1L],
            sprintf("%s[%d]", lab, seq_len(n)))
      } else if (type == "miid") {
        G <- as.integer(blk$n_groups); nf <- as.integer(blk$n_fields)
        fn <- blk$field_names %||% paste0("f", seq_len(nf))
        for (j in seq_len(nf)) {
          add(starts[k] + (j - 1L) * G + seq_len(G), 0L, 1,
              sprintf("%s.%s[%d]", lab, fn[j], seq_len(G)))
        }
      }
    }
  }
  arms <- names(fit$responses) %||% paste0("arm", seq_along(fit$responses))
  for (a in seq_along(al$n_re)) {
    n <- as.integer(al$n_re[a])
    if (n > 0L) {
      add(al$re_start[a] + seq_len(n), 0L, 1, sprintf("%s.re[%d]", arms[a],
                                                      seq_len(n)))
    }
  }
  if (!length(cols)) return(NULL)
  list(cols = cols, scale_block = scale_block, sigma_fixed = sigma_fixed,
       names = nm)
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
