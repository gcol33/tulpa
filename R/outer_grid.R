#' The outer hyperparameter grid of a nested-Laplace fit
#'
#' @description
#' A nested-Laplace fit integrates its hyperparameters on an outer grid: each
#' cell is one hyperparameter value, carries an integration weight, and holds
#' the inner Laplace solve at that value. This returns that grid, cell by cell,
#' so the integration can be inspected or reused (a discrete hyperparameter
#' posterior for [sbc_discrete()], a grid-marginalized latent field).
#'
#' The per-cell latent modes average to the grid-marginalized posterior mean of
#' any latent coordinate: `colSums(g$weights * g$modes)`. Columns
#' `seq_len(g$n_fixed)` are the fixed effects and the rest the latent field and
#' random effects, in the order the fit's latent vector stores them.
#'
#' @param fit A nested-Laplace fit: [tulpa_nested_laplace()],
#'   [tulpa_nested_laplace_joint()], or a [tulpa()] fit routed to that tier.
#' @return A list:
#'   \describe{
#'     \item{`nodes`}{numeric matrix `[n_cells x n_axes]`, the hyperparameter
#'       value of each cell, with the axis names as column names.}
#'     \item{`weights`}{the cells' integration weights, normalized to sum to
#'       one.}
#'     \item{`log_marginal`}{each cell's log marginal likelihood, hyperprior
#'       density included; with the cell's quadrature volume it gives the
#'       weight. `NULL` when the fit does not keep it.}
#'     \item{`support`}{named list of natural-scale `c(lo, hi)` intervals, the
#'       span each axis's outer measure integrates (including levels a
#'       refinement pass added); `NULL` when the fit records none.}
#'     \item{`modes`}{numeric matrix `[n_cells x n_latent]`, each cell's inner
#'       Laplace mode, or `NULL` when the fit keeps none.}
#'     \item{`n_fixed`}{how many leading columns of `modes` are fixed effects.}
#'   }
#' @seealso [tulpa_hyper_draws()] for continuous draws of the hyperparameters,
#'   [tulpa_posterior_draws()] for draws of the latent field, and
#'   `summary(fit)$hyperparameters` for their marginal summaries.
#' @examples
#' \donttest{
#' set.seed(1)
#' S <- 30L
#' nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
#' nn <- lengths(nb)
#' field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
#' idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
#' y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
#' prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
#'               adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
#'               n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
#' fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
#'                             family = "binomial",
#'                             control = list(progress = FALSE))
#' g <- tulpa_outer_grid(fit)
#' cbind(g$nodes, weight = g$weights)
#' field_hat <- colSums(g$weights * g$modes)[-seq_len(g$n_fixed)]
#' }
#' @export
tulpa_outer_grid <- function(fit) {
  src <- if (is.null(fit[["theta_grid"]]) && is.list(fit[["joint_fit"]])) {
    fit[["joint_fit"]]
  } else {
    fit
  }
  nodes <- .nl_theta_matrix(src)
  w <- as.numeric(src[["weights"]])
  if (is.null(nodes) || !length(w)) {
    stop("tulpa_outer_grid(): this fit carries no outer hyperparameter grid ",
         "(backend '", fit[["backend"]] %||% "unknown", "'); only a ",
         "nested-Laplace fit integrates its hyperparameters on one.",
         call. = FALSE)
  }
  if (length(w) != nrow(nodes)) {
    stop("tulpa_outer_grid(): the fit's weights (", length(w), ") and grid ",
         "cells (", nrow(nodes), ") do not match.", call. = FALSE)
  }
  modes <- src[["modes"]]
  list(
    nodes        = nodes,
    weights      = w / sum(w),
    log_marginal = if (length(src[["log_marginal"]]) == length(w))
                     as.numeric(src[["log_marginal"]]),
    support      = src[["axis_support"]],
    modes        = if (is.matrix(modes) && nrow(modes) == length(w)) modes,
    n_fixed      = as.integer(src[["n_fixed"]] %||% fit[["n_fixed"]] %||% 0L)
  )
}
