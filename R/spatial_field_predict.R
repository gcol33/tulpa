# spatial_field_predict.R
# ------------------------------------------------------------------------------
# The linear predictor of a fit carrying latent fields at any set of rows: the
# training rows or the rows of `newdata`. One map serves every backend and
# every field the map knows:
#
#   * an SPDE Matern field: w(s) = A(s) w_mesh, the finite-element projector at
#     s applied to the field at the mesh nodes;
#   * an HSGP field: f(s) = Phi(s) diag(sqrt(S)) z, the Laplacian basis at s on
#     the domain the fit was built on, with the spectral scales of the draw's
#     (sigma2, lengthscale);
#   * an NNGP field: w(s) | w ~ N(alpha' w_nb, sigma2 - c' C^-1 c), the NNGP
#     predictive given the field at the nearest fitted locations under the
#     draw's covariance; new locations are conditionally independent given w;
#   * a field indexed by a unit (an areal ICAR / CAR / BYM2 field, a temporal
#     RW1 / RW2 / AR1 field): the row's own unit or time, which has to be one
#     the field is defined on -- except a temporal walk or GP at a time it was
#     not fitted at, which is drawn from its conditional given the fitted path
#     (`.field_temporal_rows()`): the recursion run past the fitted times for
#     RW1 / RW2 / AR1, the Brownian bridge between them for RW1, the GP
#     conditional at any time;
#   * a multi-scale field: its components, each mapped with its own
#     hyperparameters, summed.
#
# A restricted field (`spatial_rsr()`) enters eta as P w, P the projector
# orthogonal to the restriction design X_u at the field's own support, so its
# value at unit u is w_u - x_u' delta with delta = (X_u' X_u)^-1 X_u' w. At a
# row the field is not fitted at, the same coefficient delta removes the
# restriction design's span from the field there: w(s) - x(s)' delta, x(s) the
# restriction design at the row. At the fitted support this is P w exactly.
#
# The posterior the map is applied to is a mixture over "cells" of Gaussians
# in [fixed effects, random effects, the fields at their support]:
#
#   * a grid fit (nested Laplace, the SPDE grid, a fit at fixed
#     hyperparameters as one cell): one inner-Laplace Gaussian N(m_k, Q_k^-1)
#     per hyperparameter cell, weighted by the cell's posterior weight;
#   * a sampler fit (the ModelData samplers, a Polya-Gamma Gibbs chain): each
#     stored draw is a cell of weight one with no spread, its hyperparameters
#     the draw's own.
#
# One draw of eta picks a cell by its weight, draws the latent vector from the
# cell's Gaussian (a stored draw is its own value) and maps it; the draws are
# therefore draws from the posterior predictive of eta, carrying the
# uncertainty of the coefficients and the fields, an NNGP field's conditional
# (kriging) variance away from the data and the spread over the
# hyperparameters. The point prediction is the exact posterior mean, the
# weighted mean of the same map at every cell's mode. predict(),
# posterior_linpred() and posterior_predict() all read these functions.
#
# A grid cell's precision is not kept on the fit: it is as large as the latent
# vector's sparse graph and a grid can carry hundreds of cells. The cells a set
# of draws lands in are solved again, with the fit's own inputs, by the kernel
# that solved them (`.field_cell_solve()`), and the re-solved modes are checked
# against the stored ones. An inline `spatial()` / `temporal()` field fit keeps
# its cells' precisions, and they are read as kept.
# ------------------------------------------------------------------------------

# Field block types by the map they take to a row.
.FIELD_UNIT_SPATIAL  <- c("icar", "car", "car_proper", "bym2", "rsr")
.FIELD_UNIT_TEMPORAL <- c("rw1", "rw2", "ar1")

# The map a field of type `type` takes to a row ("spde", "nngp", "hsgp",
# "unit"), NULL for a type no map is defined for.
#' @keywords internal
.field_block_kind <- function(type) {
  type <- tolower(type %||% "")
  if (type == "spde") return("spde")
  if (type %in% c("gp", "nngp", "gp_rsr")) return("nngp")
  if (type == "hsgp") return("hsgp")
  if (type %in% c(.FIELD_UNIT_SPATIAL, .FIELD_UNIT_TEMPORAL)) return("unit")
  NULL
}

# The fit's latent posterior as cells, with the layout the map reads: `modes`
# (cells x n_x), `weights`, `theta` (cells x axes, named), `solve` (the
# inner-solve record a grid cell is solved again from; NULL where there is
# none), `point` (TRUE when every cell is a stored draw with no spread) and
# `layout` (`.field_layout()`). NULL for a fit without a latent term; a fit
# carrying a latent term the read does not map is refused, naming the term,
# since a linear predictor read without it would be another model's.
#' @keywords internal
.field_source <- function(object) {
  src <- .field_source_read(object)
  miss <- if (is.null(src)) .field_latent_terms(object) else src$layout$unmapped
  if (length(miss)) {
    stop(sprintf(paste0(
      "The fit carries %s, which the linear predictor at the prediction rows ",
      "cannot be read for, so predict(), posterior_linpred() and ",
      "posterior_predict() would report a predictor without it. Pass ",
      "include_field = FALSE (predict()) for the prediction without the ",
      "latent terms."), paste(unique(miss), collapse = ", ")), call. = FALSE)
  }
  src
}

#' @keywords internal
.field_source_read <- function(object) {
  if (is.list(object$model_inputs) && is.matrix(object$draws) &&
      nrow(object$draws) > 0L) {
    return(.field_source_sampler(object))
  }
  if (identical(object$backend, "gibbs") && is.matrix(object$draws) &&
      nrow(object$draws) > 0L) {
    return(.field_source_gibbs(object))
  }
  if (!is.null(object$bar_fields)) return(.field_source_bar(object))
  .field_source_grid(object)
}

# The latent terms a fit's linear predictor carries beyond its fixed effects
# and formula random effects, by name: an inline spatial() / temporal() field,
# every non-random-effect block of a nested-Laplace block list (a field, an
# s() smoother, a spatiotemporal() interaction, a latent() block), and the
# spatial / temporal field of any other backend. A varying coefficient is
# read by its own path (`.varying_coef_eta_contrib()`) and a latent-factor
# fit's factors enter through its design (`.lf_newdata_design()`), so neither
# is listed.
#' @keywords internal
.field_latent_terms <- function(object) {
  out <- vapply(object$bar_fields %||% list(), function(rec)
    sprintf("%s(%s)", rec$space, rec$spec$group_var %||% ""), character(1))
  blocks <- .field_block_list(object$prior)
  re_b <- unlist(object$re_block_index)
  roles <- .nl_block_roles(blocks)
  for (k in seq_along(blocks)) {
    blk <- blocks[[k]]
    type <- tolower(blk$type %||% "")
    if (k %in% re_b || (type == "iid" && roles[k] == "other") ||
        (type == "lf" && !is.null(object$factor_spec))) next
    out <- c(out, .field_term_label(object, blk, roles[k], k))
  }
  sp <- object$spatial
  if (is.list(sp) && !inherits(sp, "tulpa_svc")) {
    out <- c(out, .field_term_label(object, sp, "spatial"))
  }
  tp <- object$temporal
  if (is.list(tp) && !inherits(tp, "tulpa_tvc")) {
    out <- c(out, .field_term_label(object, tp, "temporal"))
  }
  unique(out)
}

# A fit's latent blocks as a list of blocks: its block list, a single block
# wrapped, and NULL for anything else (a random-effect covariance fit's `prior`
# is its hyperprior, not a block).
#' @keywords internal
.field_block_list <- function(prior) {
  if (!is.list(prior) || !length(prior)) return(NULL)
  if (!is.null(prior$type)) return(list(prior))
  ok <- vapply(prior, function(b) is.list(b) && !is.null(b$type), logical(1))
  if (all(ok)) prior else NULL
}

# How a latent term is named to the user: a field by its space and type, a
# smoother as s(<covariate>), the interaction and a latent() block by their
# constructors.
#' @keywords internal
.field_term_label <- function(object, blk, role, k = NULL) {
  type <- tolower(blk$type %||% "")
  if (identical(role, "smooth")) {
    roles <- .nl_block_roles(.field_block_list(object$prior))
    j <- sum(roles[seq_len(k %||% 1L)] == "smooth")
    return(sprintf("s(%s)", object$smooth_terms[[j]]$var %||% "?"))
  }
  if (identical(role, "spatiotemporal") || type == "st_interaction") {
    return("the spatiotemporal() interaction")
  }
  if (type == "tgmrf") return(sprintf("latent(%s)", blk$name %||% "tgmrf"))
  if (role %in% c("spatial", "temporal")) {
    return(sprintf("the %s field (%s)", role, type))
  }
  sprintf("latent block %s (%s)", k %||% "", type)
}

# ---- inline bar fields -----------------------------------------------------------

# An inline `spatial(graph =, formula =)` / `temporal(formula =)` fit: a
# one-arm joint nested-Laplace grid that keeps each cell's precision, so its
# cells are drawn from the stored precision (`solve$door = "stored"`). Each
# block is one field per design column -- the intercept field, or a slope field
# entering as the row's column value times the field at its unit -- and a
# correlated (MCAR) spatial block is field-major over its columns.
#' @keywords internal
.field_source_bar <- function(object) {
  modes <- object$modes
  w <- object$weights
  al <- object$arm_layout
  blocks <- object$prior
  if (!is.matrix(modes) || length(w) != nrow(modes) || is.null(al) ||
      !is.list(blocks) || length(al$block_start) != length(blocks)) {
    return(NULL)
  }
  p <- as.integer(object$n_fixed %||% ncol(object$model_matrix))
  fields <- list()
  b <- 0L
  for (rec in object$bar_fields) {
    n_blk <- if (isTRUE(rec$coupled)) 1L else rec$n_cols
    for (j in seq_len(n_blk)) {
      b <- b + 1L
      blk <- blocks[[b]]
      n_u <- as.integer(blk$n_spatial_units %||% blk$n_times)
      idx <- (blk$spatial_idx %||% blk$temporal_idx)[[1L]]
      start <- al$block_start[b]
      if (isTRUE(rec$coupled)) {
        for (a in seq_len(rec$n_cols)) {
          fields[[length(fields) + 1L]] <- .field_bar_comp(
            rec, a, start + (a - 1L) * n_u + seq_len(n_u), idx,
            blk$field_weight[[a]][[1L]])
        }
      } else {
        comp <- .field_bar_comp(
          rec, j, start + seq_len(n_u), idx,
          if (!is.null(blk$svc_weight)) blk$svc_weight[[1L]])
        comp$path <- .field_bar_path(rec, blk, paste0("b", b, "."),
                                     .nl_theta_matrix(object))
        fields[[length(fields) + 1L]] <- comp
      }
    }
  }
  theta <- .nl_theta_matrix(object)
  # The formula's random-effect terms are blocks after the fields, located by
  # the fit's random-effect map as on every one-arm joint fit.
  re <- if (!is.null(object$re_effect_map)) {
    .field_re_from_map(object$re_effect_map, theta)
  }
  list(modes = modes, weights = as.numeric(w), theta = theta,
       solve = list(door = "stored"), point = FALSE,
       layout = list(beta = as.integer(al$beta_start[1L] + seq_len(p)),
                     beta_names = object$fixed_names, fields = fields, re = re))
}

# The random-effect layout a one-arm joint fit's map (`.joint_arm_re_map()`)
# gives: where each effect sits, and the factor turning cell k's coordinates
# into effects (the cell's SD for an `iid` coordinate, 1 for an `miid` one).
#' @keywords internal
.field_re_from_map <- function(jm, theta) {
  list(cols = jm$cols, sd_per_cell = all(jm$scale_block > 0L),
       scale = function(k) .joint_arm_re_scale(jm, theta, k))
}

# How an inline temporal() field's block `blk` (axes prefixed `prefix` on the
# outer grid `theta`) is read at a time it was not fitted at
# (`.field_temporal_path()`); NULL for a spatial field.
#' @keywords internal
.field_bar_path <- function(rec, blk, prefix, theta) {
  type <- tolower(blk$type %||% "")
  if (!identical(rec$space, "temporal") || !type %in% c("rw1", "rw2", "ar1")) {
    return(NULL)
  }
  tau <- .field_axis_or(theta, "tau", prefix, blk$tau_grid)
  rho <- if (type == "ar1") .field_axis_or(theta, "rho", prefix, blk$rho_grid)
  if (is.null(tau) || (type == "ar1" && is.null(rho))) return(NULL)
  spec <- list(time_var = rec$spec$group_var, time_levels = rec$time_levels)
  .field_temporal_path(spec, type, n_times = length(rec$time_levels),
                       hyper = function(cells) list(
                         tau = tau[cells], rho = if (!is.null(rho)) rho[cells]))
}

# One field of an inline bar block: its latent columns, its unit (or time) at
# each training row and its per-row design weight there (NULL for the
# intercept field, whose weight is one).
#' @keywords internal
.field_bar_comp <- function(rec, column, cols, idx, weight) {
  list(kind = "unit", space = rec$space, cols = list(as.integer(cols)),
       load = NULL, block = NULL, prefix = "", spec = NULL, restrict = NULL,
       bar = list(rec = rec, column = column, idx = as.integer(idx),
                  weight = if (!is.null(weight)) as.numeric(weight)))
}

# The unit (or time) and the design weight of each prediction row for an
# inline bar field: at the training rows the ones the block was built with, at
# `newdata` the node column resolved against the graph (the one resolver the
# fit used), in the copy of the row's `by` level for a replicated field, or the
# time column matched to the fitted times, and the field's design column
# evaluated on `newdata`. `index = FALSE` leaves the unit out, for a temporal
# field read at new times through its path.
#' @keywords internal
.field_bar_rows <- function(comp, newdata, index = TRUE) {
  bar <- comp$bar
  if (is.null(newdata)) return(list(idx = bar$idx, weight = bar$weight))
  spec <- bar$rec$spec
  gv <- spec$group_var
  if (!index) {
    idx <- NULL
  } else if (identical(bar$rec$space, "spatial")) {
    if (is.null(newdata[[gv]])) {
      stop("newdata has no column `", gv, "` naming the graph node of each ",
           "row. Add it, or pass include_field = FALSE for the prediction ",
           "without the field.", call. = FALSE)
    }
    adj <- as.matrix(spec$adjacency)
    idx <- .resolve_spatial_idx(newdata[[gv]], nrow(adj), adj, gv)
    # A replicated field: the row's node in the copy of its `by` level.
    if (!is.null(spec$by_var)) {
      lev <- .field_level_at(newdata, spec$by_var, bar$rec$by_levels,
                             paste0("`", spec$by_var, "` level"))
      idx <- idx + (lev - 1L) * as.integer(bar$rec$n_nodes)
    }
  } else {
    idx <- .field_level_at(newdata, gv, bar$rec$time_levels, "time")
  }
  cols <- .bar_field_columns(spec, newdata, fname = bar$rec$space)
  if (length(cols) != bar$rec$n_cols) {
    stop("The inline ", bar$rec$space, "() field's formula expands to ",
         length(cols), " design column(s) at newdata; the fit has ",
         bar$rec$n_cols, ".", call. = FALSE)
  }
  col <- cols[[bar$column]]
  list(idx = idx, weight = if (!isTRUE(col$is_intercept)) col$weight)
}

# ---- grid fits ---------------------------------------------------------------

#' @keywords internal
.field_source_grid <- function(object) {
  cells <- .field_cells(object)
  if (is.null(cells)) return(NULL)
  cells$point <- FALSE
  cells$layout <- .field_layout(object, cells)
  if (is.null(cells$layout)) return(NULL)
  cells
}

# Where the cells sit on each door: a nested-Laplace grid on the fit itself, an
# SPDE grid in `$nested`, and a fit at fixed hyperparameters as one cell
# (`.field_fixed_cell()`). `prior` is the latent block list the layout is read
# from (the fit's own, or the block a one-cell fit is solved again with).
#' @keywords internal
.field_cells <- function(object) {
  modes <- object$modes
  if (is.matrix(modes) && length(object$weights) == nrow(modes)) {
    return(list(modes = modes, weights = as.numeric(object$weights),
                theta = .nl_theta_matrix(object), solve = object$field_solve,
                prior = object$prior))
  }
  sp <- object$spatial
  if (identical(tolower(sp$type %||% ""), "spde")) {
    nst <- object$nested
    if (is.list(nst) && is.matrix(nst$modes) &&
        length(nst$weights) == nrow(nst$modes)) {
      return(list(modes = nst$modes, weights = as.numeric(nst$weights),
                  theta = nst$theta_grid, solve = object$field_solve,
                  prior = NULL))
    }
    if (is.numeric(object$mode) && is.numeric(object$range) &&
        is.numeric(object$sigma)) {
      return(list(modes = matrix(object$mode, nrow = 1L), weights = 1,
                  theta = cbind(range = object$range, sigma = object$sigma),
                  solve = object$field_solve, prior = NULL))
    }
    return(NULL)
  }
  .field_fixed_cell(object)
}

# A conditional-Laplace fit (`mode = "laplace"`) of an NNGP, HSGP or areal
# field: one cell at the hyperparameters its kernel was run at. That kernel
# is the registry block's at the same values (each `laplace_*_at()` builds the
# block the nested integrator builds), so the cell is solved again by the
# registry kernel on a one-row grid, which returns its precision, and its mode
# is checked against the fit's. NULL for any other fit.
#' @keywords internal
.field_fixed_cell <- function(object) {
  sp <- object$spatial
  if (!identical(object$backend, "laplace") || is.null(sp) ||
      !is.numeric(object$mode)) {
    return(NULL)
  }
  theta <- .laplace_field_theta(sp, object)
  if (is.null(theta)) return(NULL)
  rec <- object$field_solve
  blk <- rec$prior
  if (is.null(blk)) blk <- .spatial_spec_to_nl_prior(sp)
  list(modes = matrix(object$mode, nrow = 1L), weights = 1,
       theta = matrix(theta, nrow = 1L, dimnames = list(NULL, names(theta))),
       solve = rec, prior = blk)
}

# The latent columns of each component: `beta` (the fixed effects, named by
# `beta_names`), `fields` (one entry per field component, `.field_comp()`),
# and `re` (the formula random effects, with `scale(k)` the factor turning
# cell k's coordinates into effects: the cell's random-effect SD on a
# multi-block grid, whose iid block is non-centred, and 1 where the
# coordinates are the effects; `sd_per_cell` marks the first, where a new
# group's effect is drawn at the row's cell SD). NULL when a field block
# cannot be located or no map is defined for it.
#' @keywords internal
.field_layout <- function(object, cells) {
  n_x <- ncol(cells$modes)
  p <- as.integer(object$n_fixed %||% ncol(object$model_matrix))
  prior <- cells$prior
  multi <- is.list(prior) && is.null(prior$type) && length(prior)
  unmapped <- character(0)
  if (multi) {
    roles <- .nl_block_roles(prior)
    re_b <- unlist(object$re_block_index)
    types <- vapply(prior, function(b) tolower(b$type %||% ""), character(1))
    ks <- which(!(seq_along(prior) %in% re_b) &
                  !(types == "iid" & roles == "other") &
                  !(types == "lf" & !is.null(object$factor_spec)))
    fields <- lapply(ks, function(k) {
      cols <- .nl_block_cols(object$block_latent_offsets, n_x, k, length(roles))
      comp <- if (!is.null(cols)) {
        .field_comp(object, prior[[k]], cols, paste0("b", k, "."), roles[k],
                    cells$theta, spec = .field_block_spec(object, prior, k))
      }
      if (is.null(comp)) {
        unmapped <<- c(unmapped,
                       .field_term_label(object, prior[[k]], roles[k], k))
      }
      comp
    })
    fields <- Filter(Negate(is.null), fields)
    # A one-arm joint fit records where each random effect sits and what
    # scales it (`.joint_arm_re_map()`), its slope terms included; a registry
    # multi-block fit carries intercept `iid` blocks only.
    jm <- object$re_effect_map
    rl <- if (is.null(jm)) {
      .nl_re_field_load(prior, object$block_latent_offsets, n_x, cells$theta)
    }
    re <- if (!is.null(jm)) {
      .field_re_from_map(jm, cells$theta)
    } else if (!is.null(rl)) {
      list(cols = rl$cols[[1L]], sd_per_cell = TRUE,
           scale = function(k) as.numeric(rl$weights(k)[[1L]]))
    }
  } else {
    role <- if (is.list(prior) && !is.null(prior$type)) .nl_block_roles(prior)
            else "spatial"
    width <- .field_block_width(object, prior, role)
    if (is.null(width) || width > n_x - p) return(NULL)
    spec <- if (is.list(prior) && !is.null(prior$type)) {
      .field_block_spec(object, list(prior), 1L)
    }
    fields <- list(.field_comp(object, prior, seq.int(n_x - width + 1L, n_x),
                               "", role, cells$theta, spec = spec))
    if (is.null(fields[[1L]])) return(NULL)
    n_re <- n_x - width - p
    re <- if (n_re > 0L) {
      list(cols = p + seq_len(n_re), sd_per_cell = FALSE, scale = function(k) 1)
    }
  }
  if (!length(fields)) return(NULL)
  list(beta = seq_len(p), beta_names = object$fixed_names, fields = fields,
       re = re, unmapped = unmapped)
}

# The front-door spec block `k` of `blocks` is read at rows with: the fit's
# spatial or temporal spec for a field, the smoother's metadata for an s()
# block (the smoothers follow the fit's `smooth_terms` in block order), and
# NULL for a block that carries its own index.
#' @keywords internal
.field_block_spec <- function(object, blocks, k) {
  roles <- .nl_block_roles(blocks)
  switch(roles[k],
    smooth = object$smooth_terms[[sum(roles[seq_len(k)] == "smooth")]],
    temporal = object$temporal,
    spatial = object$spatial,
    NULL)
}

# Width in the latent vector of a single-block fit's field: the mesh, the
# distinct locations, the basis, or the units / times a block is laid over.
#' @keywords internal
.field_block_width <- function(object, prior, role) {
  if (!role %in% c("spatial", "temporal")) return(.nl_single_block_width(prior))
  sp <- if (identical(role, "temporal")) object$temporal else object$spatial
  type <- tolower((if (is.list(prior)) prior$type) %||% sp$type %||% "")
  kind <- .field_block_kind(type)
  if (is.null(kind)) return(NULL)
  switch(kind,
    spde = as.integer(sp$n_mesh),
    nngp = as.integer(sp$n_unique %||% nrow(sp$unique_coords)),
    hsgp = min(as.integer(sp[["m"]])^2L, nrow(sp$coords_matrix)),
    unit = .nl_single_block_width(prior))
}

# One field component: `kind` (the map), `space` ("spatial" / "temporal"),
# `cols` (one latent column set, or two for BYM2's structured and
# unstructured parts), `load(k)` (each set's weight at cell k; NULL for one
# set at weight 1), `block` (its latent block), `prefix` (what its
# hyperparameter axes carry on the outer grid, `""` on a single-block grid),
# `spec` (the front-door spec the row index is read from) and `restrict` (the
# restriction design of a projected field, `.field_restriction()`).
# NULL for a block no map is defined for.
#' @keywords internal
.field_comp <- function(object, blk, cols, prefix, role, theta, spec = NULL) {
  if (is.null(spec) && role %in% c("spatial", "temporal")) {
    spec <- if (identical(role, "temporal")) object$temporal else object$spatial
  }
  type <- tolower((if (is.list(blk)) blk$type) %||% spec$type %||% "")
  # The short-term component of a multi-scale temporal field is an `iid`
  # block over the times, non-centred: the field at a time is the coordinate
  # times the cell's SD.
  iid_time <- type == "iid" && identical(role, "temporal")
  # A smoother is a walk over its covariate's nodes; a spatiotemporal()
  # interaction and a latent() block enter eta at their coordinate
  # `x[obs_idx[i]]`.
  own_index <- type %in% c("st_interaction", "tgmrf")
  if (own_index && is.null(blk$obs_idx)) return(NULL)
  if (identical(role, "smooth") && is.null(spec)) return(NULL)
  if (!role %in% c("spatial", "temporal", "smooth") && !own_index) return(NULL)
  kind <- if (iid_time || own_index) "unit" else .field_block_kind(type)
  if (is.null(kind)) return(NULL)
  if (own_index) role <- if (type == "tgmrf") "latent" else "spatiotemporal"
  cols <- as.integer(cols)
  comp <- list(kind = kind, space = role, cols = list(cols), load = NULL,
               block = blk, prefix = prefix, spec = spec, restrict = NULL)
  # A cyclic walk (a multi-scale field's seasonal component) is laid over one
  # season: a row reads it at its time's position in the season.
  if (is.list(blk) && isTRUE(blk$cyclic)) comp$index <- as.integer(blk$n_times)
  if (iid_time) {
    nm <- paste0(prefix, "sigma")
    sg <- if (!is.null(colnames(theta)) && nm %in% colnames(theta)) {
      as.numeric(theta[, nm])
    } else rep(as.numeric(blk$sigma_grid %||% NA_real_)[1L], NROW(theta))
    comp$load <- function(k) sg[k]
  }
  if (type == "bym2") {
    n_u <- length(cols) %/% 2L
    if (2L * n_u != length(cols)) return(NULL)
    comp$cols <- list(cols[seq_len(n_u)], cols[n_u + seq_len(n_u)])
    sf <- as.numeric(blk$scale_factor %||% spec$scale_factor %||% 1)
    sg <- .field_theta(theta, "sigma", prefix)
    rh <- .field_theta(theta, "rho", prefix)
    comp$load <- function(k) cbind(sg[k] * sqrt(rh[k]) * sf, sg[k] * sqrt(1 - rh[k]))
  }
  if (is.list(blk) && !is.null(blk$projector)) {
    comp$restrict <- .field_restriction(spec)
  }
  if (identical(role, "temporal") && is.null(comp$index) && !iid_time &&
      type %in% c("rw1", "rw2", "ar1")) {
    tau <- .field_axis_or(theta, "tau", prefix, blk$tau_grid)
    sg <- if (is.null(tau)) .field_axis_or(theta, "sigma", prefix, blk$sigma_grid)
    rho <- if (type == "ar1") .field_axis_or(theta, "rho", prefix, blk$rho_grid)
    if (is.null(tau) && !is.null(sg)) tau <- 1 / sg^2
    if (!is.null(tau) && (type != "ar1" || !is.null(rho))) {
      comp$path <- .field_temporal_path(
        spec, type, n_times = as.integer(blk$n_times %||% spec$n_times),
        hyper = function(cells) list(tau = tau[cells],
                                     rho = if (!is.null(rho)) rho[cells]))
    }
  }
  comp
}

# A hyperparameter of every cell by its bare axis name and prefix, the block's
# own fixed value where the grid carries no such axis, NULL where neither is
# there.
#' @keywords internal
.field_axis_or <- function(theta, axis, prefix, fixed) {
  nm <- paste0(prefix, axis)
  if (!is.null(colnames(theta)) && nm %in% colnames(theta)) {
    return(as.numeric(theta[, nm]))
  }
  if (length(fixed) == 1L && is.finite(fixed)) {
    return(rep(as.numeric(fixed), NROW(theta)))
  }
  NULL
}

# How a temporal walk or GP over `n_times` fitted times is read at a time it
# was not fitted at: the placement of new times on its levels
# (`.temporal_query()`, from the time and group columns of `spec`), its
# `structure`, `hyper(cells)` (each prediction row's `tau` / `rho`, or GP
# `sigma2` / `phi`, at its cell) and, for a GP, its kernel.
#' @keywords internal
.field_temporal_path <- function(spec, structure, n_times, hyper,
                                 kernel = NULL) {
  lv <- spec$time_levels
  num <- suppressWarnings(as.numeric(as.character(lv)))
  info <- list(time_var = spec$time_var,
               group_var = spec$group_var,
               group_levels = as.character(spec$group_levels),
               time_levels = as.character(lv),
               level_values = if (!anyNA(num)) num,
               n_times = as.integer(n_times), structure = structure,
               time_center = spec$time_center, time_scale = spec$time_scale,
               label = sprintf("The temporal field (%s)", structure),
               noun = "temporal field")
  list(info = info, structure = structure, n_times = as.integer(n_times),
       hyper = hyper, kernel = kernel)
}

# A temporal component's values at the rows of `newdata` (R x n) for its
# support `Fr` drawn in cells `cells`: a fitted time reads the path there, any
# other time the path's conditional given it (`cpp_temporal_field_rows()`),
# drawn when `draw` and its mean otherwise.
#' @keywords internal
.field_temporal_rows <- function(comp, Fr, cells, newdata, draw) {
  path <- comp$path
  q <- .temporal_query(path$info, newdata)
  hy <- path$hyper(cells)
  kg <- path$kernel
  R <- nrow(Fr)
  cpp_temporal_field_rows(
    unname(as.matrix(Fr)), path$n_times, path$structure,
    tau = as.numeric(hy$tau %||% numeric(0)),
    rho = as.numeric(hy$rho %||% rep(0, if (is.null(hy$tau)) 0L else R)),
    sigma2 = as.numeric(hy$sigma2 %||% numeric(0)),
    phi = as.numeric(hy$phi %||% numeric(0)),
    time_values = as.numeric(kg$time_values %||% numeric(0)),
    cov_type = as.integer(kg$cov_type %||% 0L), nu = as.numeric(kg$nu %||% 0.5),
    period = as.numeric(kg$period %||% 1), query = q, draw = isTRUE(draw))
}

# The restriction a projected field carries: the restriction design at the
# field's own support (`rsr_design`, one row per unit or location) and the
# formula it is evaluated with at new rows.
#' @keywords internal
.field_restriction <- function(spec) {
  Xu <- spec$rsr_design
  if (!is.matrix(Xu)) {
    stop("The restricted field carries no restriction design at its support ",
         "($spatial$rsr_design), so it cannot be evaluated at other rows; ",
         "refit it with this version of tulpa.", call. = FALSE)
  }
  list(design = Xu, formula = spec$rsr_formula)
}

# ---- sampler fits --------------------------------------------------------------

# A ModelData sampler fit's draws as point cells: the fixed effects, the random
# effects and every field at its support per draw, read through the engine's
# own state (`cpp_sampler_field_draws()`), so a field is the one the
# likelihood read at that draw. NULL when the fit carries no field the map
# knows.
#' @keywords internal
.field_source_sampler <- function(object) {
  mi <- object$model_inputs
  if (is.null(mi$spatial_spec) && is.null(mi$temporal_spec)) return(NULL)
  fd <- do.call(cpp_sampler_field_draws,
                c(list(draws = object$draws), .sampler_model_args(object)))
  lay <- .tulpa_sampler_layout(object)
  D <- object$draws
  S <- nrow(D)
  bcols <- unlist(lapply(lay$beta, .layout_span_cols), use.names = FALSE)
  zcols <- .layout_span_cols(lay$beta_zi)
  nm <- object$fixed_names
  b_names <- if (length(nm) == length(bcols) + length(zcols)) {
    nm[seq_along(bcols)]
  } else colnames(D)[bcols]
  z_names <- if (length(zcols)) nm[length(bcols) + seq_along(zcols)]
  acc <- .field_cell_builder(D[, c(bcols, zcols), drop = FALSE])
  beta_cols <- seq_along(bcols)
  zi_cols <- length(bcols) + seq_along(zcols)

  des <- object$re_design
  M <- if (length(des)) .tulpa_re_map(object, des)
  re <- NULL
  if (!is.null(M) && nrow(M) > 0L) {
    rd <- .tulpa_re_coef_draws(object, D[, bcols, drop = FALSE], NULL, nrow(M))
    if (!is.null(rd)) {
      re <- list(cols = acc$add(rd), sd_per_cell = FALSE, scale = function(k) 1)
    }
  }

  sp <- object$spatial
  tp <- object$temporal
  fields <- list()
  theta <- list()
  add_comp <- function(kind, space, values, spec, prefix = "", block = NULL,
                       hyper = NULL) {
    fields[[length(fields) + 1L]] <<- list(
      kind = kind, space = space, cols = list(acc$add(values)), load = NULL,
      block = block, prefix = prefix, spec = spec, restrict = NULL)
    for (h in names(hyper)) theta[[paste0(prefix, h)]] <<- hyper[[h]]
  }
  if (ncol(fd$spatial_unit)) add_comp("unit", "spatial", fd$spatial_unit, sp)
  if (ncol(fd$temporal)) {
    add_comp("unit", "temporal", fd$temporal, tp)
    fields[[length(fields)]]$path <- .field_sampler_temporal_path(fd, tp)
  }
  # A multi-scale temporal field enters as its total at each time point, the
  # sum of its centred components the likelihood read.
  if (ncol(fd$ms_temporal)) {
    add_comp("unit", "temporal", fd$ms_temporal, tp)
  }
  if (ncol(fd$gp_w)) {
    add_comp("nngp", "spatial", fd$gp_w, sp,
             hyper = list(sigma2 = fd$gp_sigma2, phi_gp = fd$gp_phi))
  }
  if (ncol(fd$hsgp_z)) {
    add_comp("hsgp", "spatial", fd$hsgp_z, sp,
             hyper = list(sigma2 = fd$hsgp_sigma2,
                          lengthscale = fd$hsgp_lengthscale))
  }
  if (ncol(fd$ms_local)) {
    kind <- if (isTRUE(fd$ms_is_hsgp)) "hsgp" else "nngp"
    ax <- if (kind == "hsgp") "lengthscale" else "phi_gp"
    for (sc in c("local", "regional")) {
      blk <- list(nn = sp[[paste0("nn_", sc)]] %||% sp$nn, name = sc)
      add_comp(kind, "spatial", fd[[paste0("ms_", sc)]], sp,
               prefix = paste0(sc, "."), block = blk,
               hyper = stats::setNames(
                 list(fd[[paste0("ms_sigma2_", sc)]], fd[[paste0("ms_phi_", sc)]]),
                 c("sigma2", ax)))
    }
  }
  # Every field the model carries has to be mapped, or the source would report
  # a predictor missing one of them.
  spaces <- vapply(fields, `[[`, character(1), "space")
  if (!is.null(mi$spatial_spec) && !"spatial" %in% spaces) return(NULL)
  if (!is.null(mi$temporal_spec) &&
      (!"temporal" %in% spaces || is.null(tp$time_index))) {
    return(NULL)
  }
  .field_point_source(acc$matrix(), theta, S,
                      list(beta = beta_cols, beta_names = b_names,
                           zi = if (length(zcols)) zi_cols, zi_names = z_names,
                           fields = fields, re = re))
}

# How a sampler's temporal walk or GP (spec `tp`) is read at a time it was
# not fitted at: each stored draw's own `tau` / `rho` or GP `sigma2` / `phi`
# and the GP's kernel, as the engine read them (`cpp_sampler_field_draws()`).
# NULL for a cyclic walk, which a step past its season does not continue.
#' @keywords internal
.field_sampler_temporal_path <- function(fd, tp) {
  th <- fd$temporal_hyper
  st <- th$temporal_structure
  if (!st %in% c("rw1", "rw2", "ar1", "gp") || isTRUE(th$temporal_cyclic)) {
    return(NULL)
  }
  gp <- identical(st, "gp")
  .field_temporal_path(
    tp, st, n_times = as.integer(fd$n_times),
    hyper = function(cells) if (gp) {
      list(sigma2 = th$temporal_gp_sigma2[cells],
           phi = th$temporal_gp_phi[cells])
    } else {
      list(tau = th$temporal_tau[cells],
           rho = if (st == "ar1") th$temporal_rho[cells])
    },
    kernel = if (gp) list(time_values = th$temporal_gp_time_values,
                          cov_type = th$temporal_gp_cov_type,
                          nu = th$temporal_gp_nu,
                          period = th$temporal_gp_period))
}

# A Polya-Gamma Gibbs chain's draws as point cells. Its field columns are the
# field the kernel added to eta: an areal field per unit
# (`.gibbs_field_unit_draws()`), an NNGP field per location with its
# `log_sigma2_<suffix>` / `log_phi_<suffix>` (`.gibbs_gp_fields()`), and a
# multi-scale temporal field per component. A restricted continuous field is
# read before its projection (`gp_raw_w`) and carries its restriction, so it is
# projected at every row as the nested fit's is.
#' @keywords internal
.field_source_gibbs <- function(object) {
  D <- object$draws
  dn <- colnames(D)
  if (is.null(dn)) return(NULL)
  S <- nrow(D)
  fx <- .fixed_draws_mat(object)
  if (is.null(fx) || nrow(fx) != S) return(NULL)
  acc <- .field_cell_builder(fx)
  des <- object$re_design
  M <- if (length(des)) .tulpa_re_map(object, des)
  re <- NULL
  if (!is.null(M) && nrow(M) > 0L) {
    rd <- .tulpa_re_coef_draws(object, fx, NULL, nrow(M))
    if (!is.null(rd)) {
      re <- list(cols = acc$add(rd), sd_per_cell = FALSE, scale = function(k) 1)
    }
  }
  sp <- object$spatial
  tp <- object$temporal
  fields <- list()
  theta <- list()
  add_comp <- function(kind, space, values, spec, prefix = "", block = NULL,
                       hyper = NULL, index = NULL) {
    fields[[length(fields) + 1L]] <<- list(
      kind = kind, space = space, cols = list(acc$add(values)), load = NULL,
      block = block, prefix = prefix, spec = spec, restrict = NULL,
      index = index)
    for (h in names(hyper)) theta[[paste0(prefix, h)]] <<- hyper[[h]]
  }
  U <- .gibbs_field_unit_draws(object, seq_len(S))
  if (!is.null(U)) add_comp("unit", "spatial", U, sp)
  gpf <- if (!is.null(sp)) .gibbs_gp_fields(sp)
  restricted <- !is.null(sp) &&
    (isTRUE(sp$rsr) || identical(tolower(sp$type %||% ""), "gp_rsr"))
  for (g in gpf) {
    stem <- if (restricted) "gp_raw_w" else g$stem
    cw <- grep(paste0("^", stem, "\\["), dn)
    if (!length(cw)) {
      stop("The Gibbs chain carries no `", stem, "` columns for its field.",
           call. = FALSE)
    }
    pars <- paste0(c("log_sigma2_", "log_phi_"), g$suffix)
    if (!all(pars %in% dn)) {
      stop("The Gibbs chain carries no `", pars[!pars %in% dn][1L],
           "` column for its `", g$stem, "` field.", call. = FALSE)
    }
    add_comp("nngp", "spatial", D[, cw, drop = FALSE], sp,
             prefix = if (length(gpf) > 1L) paste0(g$suffix, ".") else "",
             block = list(nn = g$nn),
             hyper = list(sigma2 = exp(D[, pars[1L]]), phi_gp = exp(D[, pars[2L]])))
    if (restricted) {
      fields[[length(fields)]]$restrict <- .field_restriction(sp)
    }
  }
  if (!is.null(tp) && !is.null(tp$time_index)) {
    period <- as.integer(tp$seasonal %||% 0L)
    for (stem in c("trend", "seasonal", "short_term")) {
      cs <- grep(paste0("^", stem, "\\["), dn)
      if (!length(cs) || (stem == "seasonal" && period < 1L)) next
      add_comp("unit", "temporal", D[, cs, drop = FALSE], tp,
               index = if (stem == "seasonal") period)
    }
  }
  if (!length(fields)) return(NULL)
  .field_point_source(acc$matrix(), theta, S,
                      list(beta = seq_len(ncol(fx)), beta_names = colnames(fx),
                           fields = fields, re = re))
}

# Accumulates draw columns into one latent matrix: `add(V)` appends V's
# columns and returns their positions, `matrix()` the matrix.
#' @keywords internal
.field_cell_builder <- function(first) {
  parts <- list(unname(as.matrix(first)))
  width <- ncol(first)
  list(add = function(V) {
         V <- unname(as.matrix(V))
         parts[[length(parts) + 1L]] <<- V
         cols <- width + seq_len(ncol(V))
         width <<- width + ncol(V)
         cols
       },
       matrix = function() do.call(cbind, parts))
}

# The point-cell source of a sampler fit: every draw a cell of weight one.
#' @keywords internal
.field_point_source <- function(L, theta, S, layout) {
  th <- if (length(theta)) {
    do.call(cbind, lapply(theta, as.numeric))
  } else matrix(numeric(0), S, 0L)
  if (length(theta)) colnames(th) <- names(theta)
  list(modes = L, weights = rep(1, S), theta = th, solve = NULL, point = TRUE,
       layout = layout)
}

# ---- the map -------------------------------------------------------------------

# A field hyperparameter of every cell, by its bare axis name and the prefix
# its component's block carries on the outer grid (`""` on a single-block
# grid).
#' @keywords internal
.field_theta <- function(theta, axis, prefix = "") {
  j <- which(colnames(theta) == paste0(prefix, axis))
  if (length(j) != 1L) {
    stop("The fit's outer grid carries no single `", axis, "` axis for its ",
         "spatial field.", call. = FALSE)
  }
  as.numeric(theta[, j])
}

# The coordinates of the rows to predict at, on the scale the field was fitted
# on: the training coordinates when `newdata` is NULL, else the field's own
# coordinate columns of `newdata`, standardized as the training ones were.
#' @keywords internal
.field_coords_at <- function(object, newdata) {
  sp <- object$spatial
  cm <- sp$coords_matrix
  if (is.null(newdata)) return(.coords_plain(cm))
  cv <- sp$coord_vars
  miss <- setdiff(cv, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s) the spatial field was ",
         "fitted on: ", paste(miss, collapse = ", "), call. = FALSE)
  }
  nc <- as.matrix(newdata[, cv, drop = FALSE])
  if (isTRUE(sp$scale_coords)) {
    ctr <- attr(cm, "scaled:center")
    scl <- attr(cm, "scaled:scale")
    if (!is.null(ctr) && !is.null(scl)) {
      nc <- sweep(sweep(nc, 2, ctr, "-"), 2, scl, "/")
    }
  }
  .coords_plain(nc)
}

# The fitted location of each prediction row whose coordinates are those of a
# fitted location (to within 1e-10 of the coordinates' scale), NA for every
# other row.
#' @keywords internal
.field_fitted_location <- function(nc, uc) {
  tol <- 1e-10 * max(1, max(abs(uc)))
  vapply(seq_len(nrow(nc)), function(i) {
    d <- sqrt(colSums((t(uc) - nc[i, ])^2))
    j <- which.min(d)
    if (length(j) && d[j] <= tol) j else NA_integer_
  }, integer(1))
}

# The support position each prediction row reads a unit-indexed component at:
# the fitted unit of an areal field (by the field's unit column), the fitted
# block position `(g - 1) * n_times + t` of a temporal field (by its time and
# group columns; `comp$index` a season length reads `(t - 1) %% period + 1`).
# At the training rows (`newdata` NULL) each row's own. A row the field is not
# defined at is refused.
#' @keywords internal
.field_row_index <- function(object, comp, newdata) {
  spec <- comp$spec
  if (identical(comp$space, "smooth")) {
    return(if (is.null(newdata)) as.integer(spec$idx)
           else .smooth_node_at(spec, newdata))
  }
  if (identical(comp$space, "latent")) {
    if (is.null(newdata)) return(as.integer(comp$block$obs_idx))
    stop("The latent(", comp$block$name %||% "tgmrf", ") block is indexed by ",
         "the vector it was built with, which names no column of newdata, so ",
         "a new row cannot be placed in it. Pass include_field = FALSE for ",
         "the prediction without the latent terms.", call. = FALSE)
  }
  if (identical(comp$space, "spatiotemporal")) {
    blk <- comp$block
    if (is.null(newdata)) return(as.integer(blk$obs_idx))
    s_idx <- .field_row_index(object, list(space = "spatial",
                                           spec = object$spatial), newdata)
    t_idx <- .field_time_at(object$temporal, newdata)
    return((t_idx - 1L) * as.integer(blk$n_spatial) + s_idx)
  }
  if (identical(comp$space, "temporal")) {
    t_idx <- if (is.null(newdata)) as.integer(spec$time_index)
             else .field_time_at(spec, newdata)
    if (!is.null(comp$index)) return((t_idx - 1L) %% as.integer(comp$index) + 1L)
    g_idx <- if (is.null(spec$group_var)) rep(1L, length(t_idx))
             else if (is.null(newdata)) as.integer(spec$group_index)
             else .field_level_at(newdata, spec$group_var, spec$group_levels,
                                  "temporal group")
    return((g_idx - 1L) * as.integer(spec$n_times) + t_idx)
  }
  if (is.null(newdata)) return(as.integer(spec$spatial_idx))
  uv <- spec$unit_var %||% spec$group_var
  if (identical(spec$level, "obs") || is.null(uv)) {
    stop("The areal field is defined per observation (level = \"obs\"), so a ",
         "new row is a unit the field was not fitted at. Pass include_field = ",
         "FALSE for the prediction without it.", call. = FALSE)
  }
  if (!uv %in% names(newdata)) {
    stop("newdata has no column `", uv, "` naming the areal unit of each row. ",
         "Add it, or pass include_field = FALSE for the prediction without ",
         "the field.", call. = FALSE)
  }
  adj <- as.matrix(spec$adjacency)
  .resolve_spatial_idx(newdata[[uv]], nrow(adj), adj, uv)
}

# The fitted time index of each row of `newdata`, matched to the time levels
# the fit recorded. A time the fit did not see is refused: the temporal field
# carries a value at its fitted times only.
#' @keywords internal
.field_time_at <- function(spec, newdata) {
  .field_level_at(newdata, spec$time_var, spec$time_levels, "time")
}

#' @keywords internal
.field_level_at <- function(newdata, var, levels, what) {
  if (is.null(var) || !var %in% names(newdata)) {
    stop("newdata has no column `", var %||% "<none>", "` giving the ", what,
         " of each row. Add it, or pass include_field = FALSE for the ",
         "prediction without the field.", call. = FALSE)
  }
  v <- newdata[[var]]
  idx <- match(as.character(v), as.character(levels))
  if (anyNA(idx)) {
    bad <- unique(as.character(v[is.na(idx)]))
    stop(sprintf(paste0(
      "newdata has %s value(s) the fit did not see (%s); the field carries a ",
      "value at its fitted %ss only. Pass include_field = FALSE for the ",
      "prediction without it."), what,
      paste(utils::head(bad, 5L), collapse = ", "), what), call. = FALSE)
  }
  as.integer(idx)
}

# The component at its support (R x width) for latent rows `L` drawn in cells
# `cells`: its one column set, or its sets weighted by each row's cell load.
#' @keywords internal
.field_support <- function(comp, L, cells) {
  if (is.null(comp$load)) return(L[, comp$cols[[1L]], drop = FALSE])
  W <- do.call(rbind, lapply(cells, comp$load))
  out <- 0
  for (j in seq_along(comp$cols)) {
    out <- out + L[, comp$cols[[j]], drop = FALSE] * W[, j]
  }
  out
}

# The field component `comp` (an entry of the layout's `fields`) as R x n rows
# at the prediction rows, for its support `Fr` (R x width) drawn in cells
# `cells` (indices into the source's cells). `draw = TRUE` adds the NNGP
# conditional deviation; `FALSE` gives the conditional mean, which for every
# other map is the map itself.
#' @keywords internal
.field_at <- function(object, src, newdata, Fr, cells, draw, comp) {
  sp <- comp$spec
  v <- if (!is.null(comp$bar)) {
    path <- !is.null(newdata) && !is.null(comp$path)
    br <- .field_bar_rows(comp, newdata, index = !path)
    g <- if (path) .field_temporal_rows(comp, Fr, cells, newdata, draw)
         else Fr[, br$idx, drop = FALSE]
    if (is.null(br$weight)) g else sweep(g, 2L, br$weight, "*")
  } else if (!is.null(newdata) && !is.null(comp$path)) {
    .field_temporal_rows(comp, Fr, cells, newdata, draw)
  } else if (identical(comp$kind, "unit")) {
    Fr[, .field_row_index(object, comp, newdata), drop = FALSE]
  } else switch(comp$kind,
    spde = {
      A <- if (is.null(newdata)) sp$A else .spde_A_at(object, newdata)
      as.matrix(Fr %*% Matrix::t(A))
    },
    hsgp = {
      ct <- .coords_2col(.coords_plain(sp$coords_matrix), "hsgp() prediction")
      cn <- .coords_2col(.field_coords_at(object, newdata), "hsgp() prediction")
      bs <- .hsgp_block_basis(sp, comp$block$name)
      cc <- bs$c
      dom <- cpp_hsgp_domain_2d(ct, cc)
      out <- abs(cn[, 1L] - dom[1L]) > dom[3L] | abs(cn[, 2L] - dom[2L]) > dom[4L]
      if (any(out)) {
        stop(sum(out), " of ", nrow(cn), " prediction coordinate(s) lie ",
             "outside the HSGP domain (the fitted extent widened by c = ", cc,
             "), where the basis approximates no Gaussian process. Refit with ",
             "a larger `c` in spatial_gp(approx = \"hsgp\"), or use the NNGP ",
             "field.", call. = FALSE)
      }
      basis <- cpp_hsgp_basis_2d_domain(cn, bs$m, dom)
      s2 <- .field_theta(src$theta, "sigma2", comp$prefix)
      ls <- .field_theta(src$theta, "lengthscale", comp$prefix)
      u <- unique(cells)
      sq <- do.call(rbind, lapply(u, function(k) as.numeric(
        cpp_hsgp_sqrt_spectral(basis$lambda_eig, s2[k], ls[k],
                               .hsgp_kernel_code(bs$kernel)))))
      if (ncol(Fr) < ncol(basis$phi_basis)) {
        train <- comp$block$phi_basis %||% if (is.list(comp$block$phi))
          do.call(rbind, comp$block$phi) else
          cpp_hsgp_basis_2d_domain(ct, bs$m, dom)$phi_basis
        result <- matrix(0, nrow(Fr), nrow(cn))
        for (j in seq_along(u)) {
          ix <- which(cells == u[j])
          projection <- cpp_hsgp_observation_projection(train, basis$phi_basis, sq[j, ])
          result[ix, ] <- Fr[ix, , drop = FALSE] %*% t(projection$basis)
          if (draw) result[ix, ] <- result[ix, , drop = FALSE] +
            matrix(stats::rnorm(length(ix) * ncol(train)), length(ix)) %*%
              t(projection$residual_basis)
        }
        return(result)
      }
      (Fr * sq[match(cells, u), , drop = FALSE]) %*% t(basis$phi_basis)
    },
    nngp = {
      if (is.null(newdata)) {
        otl <- sp$obs_to_loc
        if (is.null(otl)) {
          stop("The fit records no map from its observations to the field's ",
               "locations ($spatial$obs_to_loc).", call. = FALSE)
        }
        Fr[, otl, drop = FALSE]
      } else {
        blk <- comp$block
        nn  <- as.integer(blk$nn %||% sp$nn %||% 10L)
        cty <- as.integer(blk$cov_type %||% gp_cov_type(sp))
        nc <- .field_coords_at(object, newdata)
        u <- unique(cells)
        noise <- if (draw) matrix(stats::rnorm(nrow(Fr) * nrow(nc)), nrow(Fr))
                 else matrix(0, nrow(Fr), nrow(nc))
        cpp_gp_field_rows(nc, .coords_plain(sp$unique_coords), Fr,
                          as.integer(match(cells, u) - 1L),
                          .field_theta(src$theta, "sigma2", comp$prefix)[u],
                          .field_theta(src$theta, "phi_gp", comp$prefix)[u],
                          nn, cty, noise)
      }
    })
  if (!is.null(comp$restrict)) {
    v <- v - .field_restriction_shift(object, comp, newdata, Fr)
  }
  v
}

# The part of a restricted field's support values `Fr` (R x n_u) that lies in
# the span of the restriction design, at the prediction rows: x_r' delta per
# row, delta = (X_u' X_u)^-1 X_u' w for each draw, x_r the design row of the
# fitted unit / location a row sits at, and at any other row the restriction
# formula evaluated on `newdata`.
#' @keywords internal
.field_restriction_shift <- function(object, comp, newdata, Fr) {
  Xu <- comp$restrict$design
  delta <- qr.coef(qr(Xu), t(Fr))                          # q x R
  delta[is.na(delta)] <- 0
  if (identical(comp$kind, "unit")) {
    Z <- Xu[.field_row_index(object, comp, newdata), , drop = FALSE]
  } else if (is.null(newdata)) {
    Z <- Xu[as.integer(comp$spec$obs_to_loc), , drop = FALSE]
  } else {
    loc <- .field_fitted_location(.field_coords_at(object, newdata),
                                  .coords_plain(comp$spec$unique_coords))
    Z <- matrix(0, nrow(newdata), ncol(Xu))
    on <- !is.na(loc)
    Z[on, ] <- Xu[loc[on], , drop = FALSE]
    if (any(!on)) {
      nd <- newdata[!on, , drop = FALSE]
      Zn <- if (is.null(comp$restrict$formula)) .tulpa_fixed_design(object, nd)
            else stats::model.matrix(comp$restrict$formula, nd)
      if (!identical(colnames(Zn), colnames(Xu))) {
        stop("The restriction formula gives the columns ",
             paste(colnames(Zn), collapse = ", "), " at newdata; the fit's ",
             "restriction design has ", paste(colnames(Xu), collapse = ", "),
             ".", call. = FALSE)
      }
      Z[!on, ] <- Zn
    }
  }
  t(Z %*% delta)
}

# The linear predictor (R x n) at the prediction rows for latent rows `L`
# (R x n_x) drawn in cells `cells`: the fixed effects and the offset of design
# `D`, every field component (`.field_at()`), and the fitted groups' random
# effects of the terms `sel` keeps on term design `des`. The random-effect
# coefficient rows travel as the `"re"` attribute.
#' @keywords internal
.field_eta_rows <- function(object, src, D, newdata, L, cells, draw, des,
                            sel = NULL) {
  lay <- src$layout
  beta <- L[, lay$beta, drop = FALSE]
  nm <- lay$beta_names %||% object$fixed_names
  nm <- if (length(nm) >= ncol(beta)) nm[seq_len(ncol(beta))]
  X <- D$X
  if (!is.null(nm) && all(nm %in% colnames(X))) {
    X <- X[, nm, drop = FALSE]
  } else if (ncol(X) != ncol(beta)) {
    stop("The prediction design does not match the fit's fixed effects.",
         call. = FALSE)
  }
  eta <- beta %*% t(X)
  if (any(D$offset != 0)) eta <- sweep(eta, 2, rep_len(D$offset, ncol(eta)), "+")
  for (comp in lay$fields) {
    eta <- eta + .field_at(object, src, newdata, .field_support(comp, L, cells),
                           cells, draw, comp)
  }
  B <- NULL
  if (!is.null(lay$re)) {
    nr <- length(lay$re$cols)
    sc <- vapply(cells, function(k) rep_len(lay$re$scale(k), nr), numeric(nr))
    B <- L[, lay$re$cols, drop = FALSE] * t(matrix(sc, nr))
    if (!is.null(sel) && any(sel)) {
      re_eta <- .tulpa_re_eta(object, des, B, sel)
      if (!is.null(re_eta)) eta <- eta + re_eta
    }
  }
  if (src$point && !is.null(newdata)) {
    vc <- .varying_coef_eta_contrib(object, newdata, rows = cells)
    if (!is.null(vc)) eta <- eta + vc
  }
  dimnames(eta) <- NULL
  attr(eta, "re") <- B
  eta
}

# The posterior mean of the linear predictor at the prediction rows: the
# weighted mean of `.field_eta_rows()` at every cell's mode, a group the fit
# did not see at its prior mean, zero.
#' @keywords internal
.field_point_eta <- function(object, src, newdata, sel = NULL,
                             accessor = "predict") {
  w <- src$weights
  on <- is.finite(w) & w > 0 & is.finite(rowSums(src$modes))
  if (!any(on)) {
    .accessor_unavailable(accessor, object, "an outer-grid cell carrying weight")
  }
  cells <- which(on)
  D <- .tulpa_designs(object, newdata, accessor)
  des <- if (is.null(newdata)) object$re_design
         else .tulpa_re_design_at(object, newdata)
  E <- .field_eta_rows(object, src, D, newdata,
                       src$modes[cells, , drop = FALSE], cells, draw = FALSE,
                       des = des, sel = sel)
  as.numeric(colSums(w[cells] * E) / sum(w[cells]))
}

# `S` draws of the latent vector from the cell mixture. On a grid the rows are
# allocated over the cells by their weights (`.nl_mixture_cells()`), the cells
# drawn in are solved again for their precision, and each row is drawn from
# its cell's Gaussian N(m_k, Q_k^-1), conditioned on the cell's sum-to-zero
# groups. On a sampler fit the rows are stored draws: every stored draw once
# when `S` reaches their number, the rest resampled, so a field's conditional
# at new rows is drawn afresh for each. Returns the S x n_x rows with the cell
# of each row as `"cells"`.
#' @keywords internal
.field_latent_draws <- function(object, src, S) {
  if (isTRUE(src$point)) {
    n <- nrow(src$modes)
    rows <- if (S <= n) sample.int(n, S)
            else c(seq_len(n), sample.int(n, S - n, replace = TRUE))
    L <- src$modes[rows, , drop = FALSE]
    attr(L, "cells") <- rows
    return(L)
  }
  w <- src$weights
  on <- is.finite(w) & w > 0 & is.finite(rowSums(src$modes))
  al <- .nl_mixture_cells(w[on] / sum(w[on]), which(on), S)
  need <- which(on)[al$counts > 0L]
  sol <- .field_cell_solve(object, src, need)
  n_x <- ncol(src$modes)
  L <- matrix(0, S, n_x)
  row_cells <- integer(S)
  pos <- 0L
  for (j in seq_along(need)) {
    n_j <- al$counts[match(need[j], which(on))]
    rows <- pos + seq_len(n_j)
    Z <- .joint_draw_cell(sol$Q_p[[j]], sol$Q_i[[j]], sol$Q_x[[j]], n_x,
                          .inner_constraint_cell(sol$constraint, j), n_j)
    L[rows, ] <- t(Z + src$modes[need[j], ])
    row_cells[rows] <- need[j]
    pos <- pos + n_j
  }
  attr(L, "cells") <- row_cells
  L
}

# The precision (lower-triangle CSC, one list entry per cell) and mode of the
# fit's cells `cells`, solved again with the inputs the fit was solved with by
# the kernel that solved it (`.field_cell_resolve()`); a fit that kept every
# cell's precision (`door = "stored"`) reads it. `constraint` is the cells'
# inner Gaussian constraint (`.inner_constraint()`), aligned with the
# precisions. A solved cell is kept in the session's cache under the record's
# `key` (`.field_solve_cache_get()`), so a fit read again -- another predict(),
# temporal() or summary() call -- solves only the cells it has not solved yet.
# The re-solved modes are checked against the stored ones, since a precision
# solved at another mode is not the fit's.
#' @keywords internal
.field_cell_solve <- function(object, src, cells) {
  rec <- src$solve
  if (is.null(rec)) {
    stop("This fit keeps no record of its inner solve, so its latent ",
         "posterior cannot be drawn at new rows (a fractional-nu SPDE ",
         "field, a fit through a model-supplied likelihood, or a ",
         "conditional-Laplace fit with observation weights). The point ",
         "prediction, predict(se.fit = FALSE), is available.", call. = FALSE)
  }
  n_x <- ncol(src$modes)
  if (identical(rec$door, "stored")) {
    return(list(Q_p = object$Q_csc_p_per_grid[cells],
                Q_i = object$Q_csc_i_per_grid[cells],
                Q_x = object$Q_csc_x_per_grid[cells],
                modes = src$modes[cells, , drop = FALSE],
                constraint = .inner_constraint(object, cells)))
  }
  out <- .field_solve_cache_get(rec$key, cells, src$modes)
  miss <- which(!out$found)
  if (length(miss)) {
    res <- .field_cell_resolve(object, src, rec, cells[miss])
    M <- src$modes[cells[miss], , drop = FALSE]
    dev <- max(abs(res$modes - M))
    if (!is.finite(dev) || dev > 1e-3 * (1 + max(abs(M)))) {
      stop("Solving the fit's outer cells again did not reproduce their ",
           "stored modes (largest difference ", signif(dev, 3), "), so the ",
           "precision it returned is not the fit's.", call. = FALSE)
    }
    .field_solve_cache_put(rec$key, cells[miss], res, M)
    out$Q_p[miss] <- res$Q_p
    out$Q_i[miss] <- res$Q_i
    out$Q_x[miss] <- res$Q_x
    out$soft_D[miss] <- res$constraint$soft_D %||% vector("list", length(miss))
    out$hard <- res$constraint$hard
    out$soft <- res$constraint$soft
  }
  soft_D <- if (!all(vapply(out$soft_D, is.null, logical(1)))) out$soft_D
  list(Q_p = out$Q_p, Q_i = out$Q_i, Q_x = out$Q_x,
       modes = src$modes[cells, , drop = FALSE],
       constraint = list(hard = out$hard %||% list(),
                         soft = if (!is.null(soft_D)) out$soft,
                         soft_D = soft_D))
}

# The cells `cells` of a grid fit solved by the kernel that solved them -- the
# SPDE grid kernel, the registry kernel of a single block, or the multi-block
# driver -- at the cells' own hyperparameters, keeping each precision.
#' @keywords internal
.field_cell_resolve <- function(object, src, rec, cells) {
  theta <- src$theta[cells, , drop = FALSE]
  prior <- rec$prior %||% object$prior
  res <- switch(rec$door,
    spde = .spde_grid_solve(
      object$spatial, y = as.numeric(object$y),
      n_trials = as.integer(object$n_trials), X = object$model_matrix,
      family = object$family, phi = object$phi, offset = object$offset,
      solve = rec$args, range_vec = as.numeric(theta[, "range"]),
      sigma_vec = as.numeric(theta[, "sigma"]), store_Q = TRUE,
      extra = list(compute_fitted_var = FALSE)),
    registry = ,
    multi = {
      a <- rec$args
      a$y <- as.numeric(object$y)
      a$X <- object$model_matrix
      a$store_Q <- TRUE
      a$prune_tol <- 0
      a$compute_fitted_var <- FALSE
      if (identical(rec$door, "multi")) {
        blocks <- if (is.list(prior) && !is.null(prior$type)) list(prior)
                  else prior
        .nl_dispatch_multi(a, blocks, theta_grid_override = theta)
      } else {
        # A field block's axis `<name>` is its `<name>_grid` entry; the cells
        # are written as one paired grid.
        blk <- prior
        for (nm in colnames(theta)) {
          blk[[paste0(nm, "_grid")]] <- as.numeric(theta[, nm])
        }
        .nl_dispatch(rec$type, a, blk, prior)
      }
    })
  if (!is.list(res$Q_csc_x_per_grid) ||
      length(res$Q_csc_x_per_grid) != length(cells)) {
    stop("The inner solve returned no precision for the fit's cells.",
         call. = FALSE)
  }
  list(Q_p = res$Q_csc_p_per_grid, Q_i = res$Q_csc_i_per_grid,
       Q_x = res$Q_csc_x_per_grid,
       modes = matrix(res$modes, nrow = length(cells)),
       constraint = .inner_constraint(res))
}

# ---- the session cache of solved cells -----------------------------------------

# Solved cells by record key: each entry holds the cells' precisions, the modes
# they were solved at and the constraint groups. Entries are dropped oldest
# first past `.FIELD_SOLVE_CACHE_MAX` stored values (the precisions' indices
# and entries, at most 200 MB as doubles); a record whose cells alone exceed it
# is not cached.
.FIELD_SOLVE_CACHE <- new.env(parent = emptyenv())
.FIELD_SOLVE_CACHE_MAX <- 2.5e7

# A key naming one fit's solve record within and across sessions: the process,
# the clock to the microsecond and a per-session counter.
#' @keywords internal
.field_solve_key <- function() {
  n <- (.FIELD_SOLVE_CACHE$counter %||% 0) + 1
  assign("counter", n, envir = .FIELD_SOLVE_CACHE)
  sprintf("%d-%s-%d", Sys.getpid(), format(Sys.time(), "%Y%m%d%H%M%OS6"), n)
}

# The cached precisions of `cells` under `key`: `found` marks the cells whose
# entry exists and was solved at the mode `modes` carries for the cell now, so
# a record whose fit was changed after it was cached is solved again. The
# constraint's hard and soft rows are the fit's (`hard`, `soft`); the soft
# rows' precision is per cell (`soft_D`).
#' @keywords internal
.field_solve_cache_get <- function(key, cells, modes) {
  e <- if (is.character(key)) .FIELD_SOLVE_CACHE$entries[[key]]
  n <- length(cells)
  out <- list(found = logical(n), Q_p = vector("list", n),
              Q_i = vector("list", n), Q_x = vector("list", n),
              soft_D = vector("list", n), hard = e$hard, soft = e[["soft"]])
  if (is.null(e)) return(out)
  j <- match(cells, e$cells)
  for (i in which(!is.na(j))) {
    if (identical(as.numeric(e$modes[j[i], ]), as.numeric(modes[cells[i], ]))) {
      out$found[i] <- TRUE
      out$Q_p[[i]] <- e$Q_p[[j[i]]]
      out$Q_i[[i]] <- e$Q_i[[j[i]]]
      out$Q_x[[i]] <- e$Q_x[[j[i]]]
      if (!is.null(e$soft_D[[j[i]]])) out$soft_D[[i]] <- e$soft_D[[j[i]]]
    }
  }
  out
}

# Adds the solved cells `cells` (`res`, solved at the fit's modes `M`) to the
# entry of `key`, replacing any it held for them; the entry becomes the newest,
# and the oldest entries are dropped until the cache is within its size.
#' @keywords internal
.field_solve_cache_put <- function(key, cells, res, M) {
  if (!is.character(key)) return(invisible())
  size <- function(e) {
    2 * sum(lengths(e$Q_x)) + sum(lengths(e$Q_p)) + length(e$modes)
  }
  ent <- .FIELD_SOLVE_CACHE$entries %||% list()
  e <- ent[[key]] %||% list(cells = integer(0), Q_p = list(), Q_i = list(),
                            Q_x = list(), soft_D = list(), modes = NULL)
  keep <- !(e$cells %in% cells)
  e$cells <- c(e$cells[keep], as.integer(cells))
  e$Q_p <- c(e$Q_p[keep], res$Q_p)
  e$Q_i <- c(e$Q_i[keep], res$Q_i)
  e$Q_x <- c(e$Q_x[keep], res$Q_x)
  e$soft_D <- c(e$soft_D[keep],
                res$constraint$soft_D %||% vector("list", length(cells)))
  e$modes <- rbind(e$modes[keep, , drop = FALSE], unname(M))
  e$hard <- res$constraint$hard
  e$soft <- res$constraint$soft
  ent[[key]] <- NULL
  if (size(e) <= .FIELD_SOLVE_CACHE_MAX) ent[[key]] <- e
  while (length(ent) > 1L &&
         sum(vapply(ent, size, numeric(1))) > .FIELD_SOLVE_CACHE_MAX) {
    ent[[1L]] <- NULL
  }
  assign("entries", ent, envir = .FIELD_SOLVE_CACHE)
  invisible()
}

# Linear-predictor draws (S x n) at the prediction rows of a field fit, with
# the random effects of the terms `sel` keeps: a fitted group's from the same
# latent draw, a group the fit did not see from its covariance at the row's
# cell. The grid cell of each row travels as `"cells"`, and on a
# zero-inflated sampler fit the structural-zero logit of the same draws as
# `"logit_zi"`.
#' @keywords internal
.field_eta_draws <- function(object, src, newdata, S, sel = NULL) {
  L <- .field_latent_draws(object, src, S)
  cells <- attr(L, "cells")
  D <- .tulpa_designs(object, newdata, "posterior_predict")
  des <- if (is.null(newdata)) object$re_design
         else .tulpa_re_design_at(object, newdata)
  eta <- .field_eta_rows(object, src, D, newdata, L, cells, draw = TRUE,
                         des = des, sel = sel)
  attr(eta, "re") <- NULL
  if (!is.null(newdata) && !is.null(sel) && any(sel) && length(des)) {
    sig <- if (isTRUE(src$layout$re$sd_per_cell) && length(des) == 1L) {
      list(lapply(cells, function(k) matrix(src$layout$re$scale(k)[1L]^2)))
    }
    eta <- eta + .re_new_group_eta(object, des,
                                   if (isTRUE(src$point)) cells, S = nrow(eta),
                                   sel, sig = sig)
  }
  zi <- src$layout$zi
  if (length(zi)) {
    X_zi <- D$X_zi[, src$layout$zi_names, drop = FALSE]
    attr(eta, "logit_zi") <- L[, zi, drop = FALSE] %*% t(X_zi)
  }
  if (!isTRUE(src$point)) attr(eta, "cells") <- cells
  eta
}
