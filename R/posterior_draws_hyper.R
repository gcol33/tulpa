# The hyperparameter half of a nested-Laplace posterior draw.
#
# A draw from the outer-grid mixture is two steps -- pick a cell by weight, then
# draw that cell's inner Gaussian -- and `tulpa_posterior_draws()` gives the
# LATENT half of the second step a genuine within-cell Gaussian. The
# hyperparameter half had none: a draw's `sigma` / `alpha` / `phi` was read
# straight off the chosen cell's grid coordinate, so a fit integrating a 5-node
# axis produced draws taking 5 distinct values on it whatever the sample size.
# That is an ATOM, not a marginal, and it is the same object gcol33/tulpa#337
# named ("the read collapses each box to its midpoint") -- which #337 and #353
# fixed for the reported interval (`.nl_summary_quantile()`) and nowhere else,
# so every consumer of raw draws still had the atom (gcol33/tulpa#823).
#
# THE FIX IS NOT A NEW CONSTRUCTION. Each of the two within-cell reads the
# engine ships already DEFINES a continuous density on an axis, and each is a
# per-cell mixture whose component is available in closed form. Sampling that
# component conditional on the cell the draw came from is what this file does,
# so the hyperparameter draws reproduce the fit's own `theta_ci_lo` /
# `theta_median` / `theta_ci_hi` to Monte Carlo error rather than approximating
# them:
#
#   * `box_uniform` -- cell c's whole mass `w_c` sits uniformly on its own box
#     `[e_c, e_{c+1}]` (`.nl_box_quantile()` interpolates linearly between those
#     edges), so the conditional is `Uniform(e_c, e_{c+1})`. Summed over cells
#     the density on box c is `w_c / (e_{c+1} - e_c)`: the box read exactly.
#   * `chord` -- the CDF's knots are `(cumsum(w) - w / 2, v)`, so between two
#     adjacent coordinates the density is uniform with mass
#     `(w_c + w_{c+1}) / 2` and cell c contributes HALF its mass to the segment
#     on either side. The conditional is therefore an equal mixture of
#     `Uniform(v_{c-1}, v_c)` and `Uniform(v_c, v_{c+1})`. Summed over cells the
#     density on `[v_c, v_{c+1}]` is `(w_c + w_{c+1}) / (2 (v_{c+1} - v_c))`:
#     the chord read exactly.
#
# Neither conditional depends on the weights, only on the partition geometry --
# which is what makes this safe: the marginal comes out right for ANY cell
# weighting, so the jitter cannot disagree with the sampler that chose the cell.
# The weights are read only where the READ itself filters on them (the chord
# knots are the positive-weight coordinates, the box partition is every
# coordinate; see `.nl_box_quantile()`'s own note on why the two differ).
#
# The outer half-cell is part of it. A `density` support's outermost cell
# reaches half a spacing past its coordinate (`outside = "extend"`), which is
# the region a truth drawn beyond the outermost node lands in and the region no
# node-valued draw can reach. A `sample` support clamps instead, and there the
# extreme cell's outward half IS an atom at its coordinate -- handled by the
# same code, with the neighbour set to the coordinate itself.

# The within-cell geometry of ONE axis: the coordinates a draw's cell value is
# matched against, and the interval each of them spreads its mass over.
#
# `kind` is the construction that produced it -- `"box_uniform"`, `"chord"`, or
# `"none"` when the axis carries no continuization and a draw keeps its node
# value. `lo` / `hi` are per-coordinate: the box's two edges under
# `box_uniform`, the left and right NEIGHBOUR coordinates (outer edge where
# there is none) under `chord`. `declined` is why the requested construction did
# not run, from the vocabulary `.nl_summary_quantile_read()` already uses, plus
# the two this path adds for a support that reaches no CDF at all.
.nl_hyper_axis_geometry <- function(v, w, domain, within, outside) {
  none <- function(declined) {
    list(kind = "none", values = numeric(0), lo = numeric(0),
         hi = numeric(0), declined = declined)
  }
  if (is.na(outside)) return(none("support_moment_rule"))

  chord <- function(declined = NA_character_) {
    # The chord read's knots ARE the positive-weight coordinates, so its
    # partition is taken off the same atoms `.nl_wtd_quantile()` builds.
    a <- .nl_axis_atoms(v, w)
    if (is.null(a)) return(none("no_usable_node"))
    uv <- a$v
    n <- length(uv)
    if (n < 2L) return(none("single_node"))
    e <- if (identical(outside, "extend")) {
      .nl_cell_edges(uv, domain)
    } else {
      # `clamp`: nothing is known past the extreme order statistic, so the
      # outward half of each extreme cell's mass stays on its coordinate.
      c(uv[1L], uv[n])
    }
    list(kind = "chord", values = uv,
         lo = c(e[1L], uv[-n]), hi = c(uv[-1L], e[2L]),
         declined = declined)
  }

  if (identical(within, "chord")) return(chord())
  fin <- is.finite(v)
  if (!any(fin)) return(none("no_usable_node"))
  uv <- sort(unique(as.numeric(v[fin])))
  if (length(uv) < 2L) return(none("single_node"))
  e <- .nl_box_edges_from(.nl_cell_partition(uv, domain), uv)
  if (is.null(e)) return(chord("boxes_do_not_tile"))
  list(kind = "box_uniform", values = uv,
       lo = e[-length(e)], hi = e[-1L], declined = NA_character_)
}

# One axis's per-draw coordinate: the cell value continuized across that cell's
# own geometry. A draw whose cell value is not one of the geometry's
# coordinates -- reachable only where the draws were allocated over a cell set
# the axis read filtered out -- keeps its node value rather than being matched
# to a neighbouring box.
.nl_hyper_axis_draw <- function(geom, v_cell) {
  if (identical(geom$kind, "none")) return(v_cell)
  k <- match(v_cell, geom$values)
  n <- length(v_cell)
  u <- stats::runif(n)
  out <- if (identical(geom$kind, "box_uniform")) {
    geom$lo[k] + u * (geom$hi[k] - geom$lo[k])
  } else {
    # Half the cell's mass on either side, each uniform on its own segment.
    left <- stats::runif(n) < 0.5
    ifelse(left,
           geom$lo[k] + u * (geom$values[k] - geom$lo[k]),
           geom$values[k] + u * (geom$hi[k] - geom$values[k]))
  }
  miss <- is.na(k) | !is.finite(out)
  out[miss] <- v_cell[miss]
  out
}

#' Hyperparameter draws from a nested-Laplace fit
#'
#' @description
#' The hyperparameter half of a draw from the outer-grid mixture: one row per
#' draw, one column per outer-grid axis. A draw's coordinate on each axis is
#' sampled from the within-cell density its cell carries under the fit's own
#' within-cell read, so the columns are a continuous marginal rather than the
#' handful of grid-node values `fit$theta_grid[cells, ]` returns.
#'
#' Reading the node coordinate directly is what
#' [tulpa_posterior_draws()] consumers used to do, and on a 5- to 15-node axis
#' it makes every draw an atom: a truth between two nodes, or past the
#' outermost one, has no draw that can land near it. The continuization is the
#' cell-conditional of the SAME construction the fit reports its interval from
#' (`box_uniform` by default, `chord` when the fit asked for it or its support
#' does not admit the box read), so the draws reproduce the fit's own
#' `theta_ci_lo` / `theta_median` / `theta_ci_hi` to Monte Carlo error.
#'
#' @param fit A nested-Laplace fit ([tulpa_nested_laplace()] or
#'   [tulpa_nested_laplace_joint()]).
#' @param cells Integer vector of outer-grid cell indices, one per draw -- the
#'   `"cells"` attribute of a [tulpa_posterior_draws()] matrix, so the
#'   hyperparameter row and the latent row of a draw come from the same cell.
#'   `NULL` (default) allocates `n` fresh draws across the cells by weight.
#' @param n Number of draws when `cells` is `NULL` (default 1000); ignored
#'   otherwise.
#' @param within Within-cell construction, `"box_uniform"` or `"chord"`.
#'   `NULL` (default) takes the one the fit was read with
#'   (`fit$within_cell_requested`).
#'
#' @return A numeric matrix `[n x n_axes]` with the outer grid's axis names as
#'   columns, or `NULL` when the fit carries no outer-grid axis. Carries
#'   `attr(., "cells")` -- the cell each row's coordinates were drawn in --
#'   plus `attr(., "within_cell")` and `attr(., "within_cell_declined")`, the
#'   per-axis construction that ran and why a requested one did not.
#'
#' @seealso [tulpa_posterior_draws()], [tulpa_nested_laplace()]
#' @export
tulpa_hyper_draws <- function(fit, cells = NULL, n = 1000, within = NULL) {
  tg <- .nl_theta_matrix(fit)
  if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L || nrow(tg) == 0L) {
    return(NULL)
  }
  w <- fit$weights
  if (is.null(w) || length(w) != nrow(tg) || !any(is.finite(w) & w > 0)) {
    # No stored measure: every coordinate is an atom of the chord partition,
    # which is what an unweighted read of the same node set would take.
    w <- rep(1, nrow(tg))
  }
  if (is.null(cells)) {
    n <- .nl_draws_check_n(n)
    ok <- is.finite(w) & w > 0
    cells <- .nl_mixture_cells(w[ok] / sum(w[ok]), which(ok), n)$row_cells
  } else {
    cells <- as.integer(cells)
    if (anyNA(cells) || any(cells < 1L) || any(cells > nrow(tg))) {
      stop("`cells` must be 1-based outer-grid cell indices (1..",
           nrow(tg), ").", call. = FALSE)
    }
  }

  within  <- .nl_within_cell_mode(within %||% fit$within_cell_requested)
  support <- .nl_node_support(fit$integration, fit$weight_kind)
  outside <- .NL_SUPPORT[[support]]$outside
  # A support that does not admit the requested construction falls back to the
  # chord read exactly as the interval does, and says so per axis.
  req <- if (within %in% .NL_SUPPORT[[support]]$within) within else "chord"
  fell <- if (identical(req, within)) NA_character_ else
    paste0("support_", support)
  # An axis whose support the registry will not name carries NA and mirrors its
  # outer edge in the coordinate the guess picks, exactly as the interval read
  # does; a fit the registry cannot read at all leaves every axis undeclared
  # rather than taking the draw down.
  doms <- tryCatch(.nl_axis_domains(fit), error = function(e) NULL)

  nms <- .nl_axis_names(tg)
  out <- matrix(0.0, length(cells), ncol(tg), dimnames = list(NULL, nms))
  used <- stats::setNames(rep(NA_character_, ncol(tg)), nms)
  decl <- stats::setNames(rep(NA_character_, ncol(tg)), nms)
  for (j in seq_len(ncol(tg))) {
    dm <- if (length(doms) < j) NA_character_ else doms[[j]]
    g <- .nl_hyper_axis_geometry(as.numeric(tg[, j]), w, dm, req, outside)
    out[, j] <- .nl_hyper_axis_draw(g, as.numeric(tg[cells, j]))
    used[j] <- if (identical(g$kind, "none")) NA_character_ else g$kind
    decl[j] <- if (is.na(g$declined)) fell else g$declined
  }
  attr(out, "cells") <- cells
  attr(out, "within_cell") <- used
  attr(out, "within_cell_declined") <- decl
  out
}

# The `"theta"` attribute both `tulpa_posterior_draws()` methods carry: the
# hyperparameter half of the same rows, drawn in the cells the latent half came
# from. Attached rather than left to the caller because the caller's own
# alternative is `theta_grid[cells, ]`, which is the atom this exists to
# replace -- a consumer reading the attribute is correct by default, and one
# calling `tulpa_hyper_draws()` on the `"cells"` attribute gets the same object.
.nl_attach_hyper_draws <- function(out, fit) {
  # Attached without the caller asking, so it runs with the RNG RESTORED: an
  # automatic extra must not move the stream, or every number downstream of an
  # existing `set.seed()` script shifts. The latent draws are formed before it
  # and are bit-for-bit what they were; an explicit `tulpa_hyper_draws()` call
  # consumes the stream normally, as a draw should.
  th <- .with_preserved_seed(
    tryCatch(tulpa_hyper_draws(fit, cells = attr(out, "cells")),
             error = function(e) conditionMessage(e)))
  # A fit with no outer axis has no hyperparameter half, which is not a
  # failure; anything else that stopped the continuization is RECORDED rather
  # than swallowed, because a silently absent attribute sends its reader back
  # to `theta_grid[cells, ]` -- the atom this replaces.
  if (is.character(th)) {
    attr(out, "theta_declined") <- th
    return(out)
  }
  if (is.null(th)) return(out)
  attr(out, "theta") <- th
  attr(out, "theta_within_cell") <- attr(th, "within_cell")
  attr(out, "theta_within_cell_declined") <- attr(th, "within_cell_declined")
  out
}
