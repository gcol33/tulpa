# Generic axis spec for outer hyperparameter-grid integration.
#
# A `hyper_axis_spec` describes one outer-grid axis -- its name, its candidate
# values, its prior, and the metadata the refinement / consistency passes need
# (log-scale flag, natural bounds, opt-in for refinement). The same object is
# consumed by:
#   * `tulpa_hyper_grid()` (public driver), and
#   * the generic refinement / consistency helpers used internally by both
#     `tulpa_hyper_grid()` and `tulpa_nested_laplace_joint()`.
#
# Field reference -- see `?hyper_axis_spec`.
#   name        character(1)        axis label, used as the column name of the
#                                   grid matrix and in posterior summaries.
#   grid        numeric              the per-axis candidate values (the outer
#                                   integration nodes on this axis).
#   log_prior   function(x) | NULL   optional log-prior density on the axis.
#                                   `NULL` -> flat (zero log-prior).
#   log_scale   logical(1)           does the axis live naturally on a log
#                                   scale (sigma, tau, lengthscale, ...)?
#                                   Drives geometric vs arithmetic spacing in
#                                   refinement and log-axis quantile fits.
#   bounds      numeric(2) | NULL    natural support of the axis (e.g. (0, Inf)
#                                   for `sigma`, (0, 1) for `rho`). `NULL`
#                                   means unbounded. Refinement clamps new
#                                   points to the open interior.
#   refinable   logical(1)           opt in to adaptive-grid refinement and the
#                                   var-of-means consistency pass on this axis.
#   slab_bounds numeric(2) | NULL    fixed support of the continuum prior. The
#                                   measure is normalised over it, so it is a
#                                   prior choice and refinement may not leave
#                                   it. NULL means the node span is the support
#                                   (legacy, data-dependent).
#   atom_mass   numeric(1) | NULL    prior probability of a zero level on a
#                                   log-scale axis (a point mass outside the
#                                   log continuum). Required when the grid
#                                   carries a 0; NULL otherwise.

#' Describe one outer-grid hyperparameter axis
#'
#' @description
#' Builds a single axis spec for [tulpa_hyper_grid()]. Carries the candidate
#' values, the optional log-prior, and the metadata (log-scale, bounds,
#' refinable flag) that the generic refinement / consistency passes need.
#'
#' @param name Character. Axis label, used as the column name of the grid
#'   matrix and in posterior summaries.
#' @param grid Numeric vector of length >= 1. The per-axis candidate values
#'   (the outer integration nodes on this axis). The full outer grid is the
#'   Cartesian product across axes.
#' @param log_prior Optional `function(x)` returning the scalar log prior
#'   density at axis value `x`. `NULL` (default) is a flat / improper prior
#'   (zero log-prior contribution).
#' @param log_scale Logical. Does the axis live naturally on a log scale
#'   (`sigma`, `tau`, `lengthscale`, ...)? Drives geometric vs arithmetic
#'   spacing in refinement and log-axis quantile fits. Default `FALSE`.
#' @param bounds Numeric vector of length 2 giving the natural support
#'   `(lower, upper)` of the axis, e.g. `c(0, Inf)` for `sigma`, `c(0, 1)`
#'   for a BYM2 mixing coefficient. `NULL` (default) is unbounded.
#' @param slab_bounds Numeric `c(lower, upper)`, or `NULL` (default). Fixed
#'   support of the continuum part of the prior. A flat measure on a log axis is
#'   improper, so the truncation bounds are what make it a proper density and
#'   they are therefore a prior choice: the weights normalise over `slab_bounds`
#'   and refinement is not allowed to move outside it. `NULL` leaves the node
#'   span acting as the support, which makes the prior depend on where
#'   refinement put the outermost node.
#' @param atom_mass Numeric in `[0, 1)`, or `NULL` (default). Prior probability
#'   of the zero level on a log-scale axis. A `0` is a point mass, not a point
#'   of the log continuum, so its prior share has to be declared rather than
#'   inherited from the node count; the continuum nodes then share
#'   `1 - atom_mass` by quadrature weight. Required when `grid` contains a `0`
#'   on a `log_scale` axis.
#' @param refinable Logical. When `TRUE`, the axis participates in the
#'   adaptive-grid and var-of-means consistency passes (when those are
#'   enabled at the driver level). Spatial prior amplitudes (`sigma`) are
#'   typically left at the user-specified grid (`refinable = FALSE`); the
#'   copy coefficient `alpha` and per-arm dispersion `phi` typically opt in.
#'   Default `FALSE`.
#'
#' @return An object of class `tulpa_hyper_axis_spec` (a validated list with
#'   the six fields above).
#'
#' @seealso [tulpa_hyper_grid()].
#' @keywords internal
#' @export
hyper_axis_spec <- function(name, grid, log_prior = NULL,
                            log_scale = FALSE, bounds = NULL,
                            refinable = FALSE, atom_mass = NULL,
                            slab_bounds = NULL) {
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("`name` must be a non-empty character string.", call. = FALSE)
  }
  grid <- as.numeric(grid)
  if (length(grid) < 1L || any(!is.finite(grid))) {
    stop(sprintf("Axis '%s': `grid` must be a non-empty numeric vector ",
                 name), "of finite values.", call. = FALSE)
  }
  if (!is.null(log_prior) && !is.function(log_prior)) {
    stop(sprintf("Axis '%s': `log_prior` must be NULL or a function.", name),
         call. = FALSE)
  }
  log_scale <- isTRUE(log_scale)
  if (log_scale && any(grid < 0)) {
    stop(sprintf("Axis '%s': `log_scale = TRUE` is incompatible with ",
                 name), "negative `grid` values.", call. = FALSE)
  }
  # A 0 in a log-scale grid is valid as a boundary / no-effect atom (e.g. the
  # alpha = 0 cell representing "no copy" in the joint driver); refinement's
  # log-midpoint formulae yield 0 there, which the `pts > bounds[1L]` filter
  # in `.hyper_propose_axis_extension` correctly drops.
  if (!is.null(bounds)) {
    bounds <- as.numeric(bounds)
    if (length(bounds) != 2L || bounds[1L] >= bounds[2L]) {
      stop(sprintf("Axis '%s': `bounds` must be `c(lower, upper)` with ",
                   name), "lower < upper.", call. = FALSE)
    }
    if (any(grid < bounds[1L]) || any(grid > bounds[2L])) {
      stop(sprintf("Axis '%s': `grid` values outside `bounds`.", name),
           call. = FALSE)
    }
  }
  if (!is.null(atom_mass)) {
    atom_mass <- as.numeric(atom_mass)
    if (length(atom_mass) != 1L || !is.finite(atom_mass) ||
        atom_mass < 0 || atom_mass >= 1) {
      stop(sprintf("Axis '%s': `atom_mass` must be a single value in [0, 1).",
                   name), call. = FALSE)
    }
  }
  if (!is.null(slab_bounds)) {
    slab_bounds <- as.numeric(slab_bounds)
    if (length(slab_bounds) != 2L || !all(is.finite(slab_bounds)) ||
        slab_bounds[1L] >= slab_bounds[2L]) {
      stop(sprintf("Axis '%s': `slab_bounds` must be finite `c(lower, upper)` ",
                   name), "with lower < upper.", call. = FALSE)
    }
  }
  out <- list(
    name        = name,
    grid        = grid,
    atom_mass   = atom_mass,
    slab_bounds = slab_bounds,
    log_prior = log_prior,
    log_scale = log_scale,
    bounds    = bounds,
    refinable = isTRUE(refinable)
  )
  class(out) <- c("tulpa_hyper_axis_spec", "list")
  out
}

# Normalise a list of axis specs to a list-of-tulpa_hyper_axis_spec.
# Accepts: a single spec, a list of specs, or a list of plain `list(name=...,
# grid=...)` entries (auto-wrapped via hyper_axis_spec()).
.hyper_axis_specs_normalise <- function(specs) {
  if (inherits(specs, "tulpa_hyper_axis_spec")) {
    return(list(specs))
  }
  if (!is.list(specs) || length(specs) < 1L) {
    stop("`hyper_specs` must be a non-empty list of axis specs.",
         call. = FALSE)
  }
  out <- lapply(seq_along(specs), function(j) {
    s <- specs[[j]]
    if (inherits(s, "tulpa_hyper_axis_spec")) return(s)
    if (!is.list(s)) {
      stop(sprintf("hyper_specs[[%d]] must be a list or hyper_axis_spec ",
                   j), "object.", call. = FALSE)
    }
    do.call(hyper_axis_spec, s)
  })
  names_taken <- vapply(out, `[[`, character(1), "name")
  if (anyDuplicated(names_taken)) {
    stop("Duplicate axis names in `hyper_specs`: ",
         paste(names_taken[duplicated(names_taken)], collapse = ", "),
         ".", call. = FALSE)
  }
  out
}
