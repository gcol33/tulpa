# latent_effects.R
# ------------------------------------------------------------------------------
# Per-coordinate posterior of a fit's latent() block. A nested-Laplace fit
# retains each cell's block coordinates as a loading of the cell's latent
# Gaussian (`grid_latent_mean` / `grid_latent_var`, `.nl_attach_unit_loads()`),
# the construction field_effects() and ranef() read the spatial field and the
# random effects with.
# ------------------------------------------------------------------------------

#' Latent block effects per coordinate
#'
#' @description
#' The posterior of one `latent()` block ([tgmrf()], [tgmrf_cpp()]) of a
#' nested-Laplace fit, one row per coordinate of the block: each outer cell
#' contributes the coordinate's conditional mean and marginal variance under
#' the cell's latent Gaussian, and the rows are the exact mean, SD and 2.5% /
#' 97.5% quantiles of that Gaussian mixture over the cell weights. Each
#' coordinate's value is its contribution to the linear predictor of the
#' observations `obs_idx` maps to it.
#'
#' @param object A nested-Laplace `tulpa_fit` carrying at least one `latent()`
#'   block.
#' @param block Which latent block: its position among the fit's `latent()`
#'   blocks (default 1), or the `name` given to [tgmrf()].
#' @return A data frame with one row per coordinate: `index`, `estimate`, `sd`,
#'   `conf.low`, `conf.high` and `source` (`"mixture"`, or `"mode"` where the
#'   fit retained no per-cell variance).
#' @seealso [field_effects()] for a spatial field, [ranef()] for random
#'   effects, [temporal()] for a temporal field.
#' @examples
#' \donttest{
#' set.seed(1)
#' K <- 12
#' W <- matrix(0, K, K)
#' for (i in 1:K) { j <- if (i < K) i + 1 else 1; W[i, j] <- W[j, i] <- 1 }
#' d <- data.frame(region = rep(1:K, each = 10), x = rnorm(120))
#' d$y <- rpois(120, exp(0.5 + 0.4 * d$x + sin(2 * pi * d$region / K)))
#' blk <- tgmrf(Q = function(theta) Matrix::Diagonal(120, exp(theta[1])),
#'              prior = function(theta) dnorm(theta[1], 2, 1.5, log = TRUE),
#'              init = c(log_tau = 2), bounds = list(lower = -1, upper = 7),
#'              obs_idx = seq_len(120), name = "obs_iid")
#' fit <- tulpa(y ~ x + spatial(region) + latent(blk), data = d,
#'              family = "poisson", spatial = list(type = "icar", adjacency = W),
#'              mode = "auto", control = list(progress = FALSE))
#' head(latent_effects(fit, "obs_iid"))
#' }
#' @export
latent_effects <- function(object, block = 1L) {
  if (!inherits(object, "tulpa_fit")) {
    stop("latent_effects() needs a `tulpa_fit` object.", call. = FALSE)
  }
  mu <- object$grid_latent_mean
  sizes <- object$grid_latent_sizes
  if (!is.matrix(mu) || !length(sizes)) {
    stop("latent_effects(): this fit retains no latent() block posterior ",
         "(a nested-Laplace fit carrying a latent(tgmrf()) block does).",
         call. = FALSE)
  }
  blk_idx <- as.integer(names(sizes))
  names_blk <- vapply(blk_idx, function(b)
    as.character(object$blocks[[b]]$name %||% NA_character_), character(1))
  j <- if (is.character(block)) match(block, names_blk) else as.integer(block)
  if (length(j) != 1L || is.na(j) || j < 1L || j > length(sizes)) {
    stop(sprintf(paste0(
      "latent_effects(): `block` must be 1..%d or one of the block names (%s)."),
      length(sizes), paste(stats::na.omit(names_blk), collapse = ", ")),
      call. = FALSE)
  }
  cols <- sum(sizes[seq_len(j - 1L)]) + seq_len(sizes[j])
  V <- object$grid_latent_var
  w <- as.numeric(object$weights)
  keep <- is.finite(w) & w > 0 & is.finite(rowSums(mu[, cols, drop = FALSE]))
  if (is.matrix(V)) keep <- keep & is.finite(rowSums(V[, cols, drop = FALSE]))
  if (!any(keep)) {
    stop("latent_effects(): no outer-grid cell carries a finite block and a ",
         "positive weight.", call. = FALSE)
  }
  var <- if (is.matrix(V)) pmax(V[keep, cols, drop = FALSE], 0)
  mx <- .nl_gauss_mixture_summary(mu[keep, cols, drop = FALSE], var,
                                  w[keep] / sum(w[keep]),
                                  probs = c(0.025, 0.975))
  out <- data.frame(index = seq_along(cols), estimate = mx$mean, sd = mx$sd,
                    conf.low = mx$quantiles[, 1L],
                    conf.high = mx$quantiles[, 2L],
                    source = if (is.null(var)) "mode" else "mixture",
                    row.names = NULL, stringsAsFactors = FALSE)
  attr(out, "block") <- if (!is.na(names_blk[j])) names_blk[j] else j
  out
}
