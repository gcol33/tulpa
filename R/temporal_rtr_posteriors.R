#' Restricted temporal regression (RTR)
#'
#' @description
#' The temporal analogue of [spatial_rsr()]: constrain a temporal random effect
#' to be orthogonal to a set of covariates, so a temporally smooth covariate does
#' not have its fixed-effect coefficient attenuated by a confounded temporal
#' field, \eqn{u_{RTR} = (I - P_X) u}.
#'
#' No tulpa backend applies that projection to a temporal field, so this
#' constructor errors rather than returning a specification that would fit as the
#' unrestricted temporal model. [spatial_rsr()] is the wired spatial analogue.
#'
#' @param temporal A `tulpa_temporal` specification (e.g. [temporal_rw1()],
#'   [temporal_ar1()]).
#' @param restrict_to A one-sided formula giving the covariate space the temporal
#'   effect is made orthogonal to, e.g. `~ x`.
#'
#' @return Nothing: the call always signals an error.
#'
#' @seealso [spatial_rsr()], [temporal_rw1()], [temporal_ar1()]
#'
#' @export
temporal_rtr <- function(temporal, restrict_to) {
  stop("Restricted temporal regression is not fitted by any tulpa backend: ",
       "no temporal solver applies the (I - P_X) projection, so an RTR ",
       "specification would fit as the unrestricted temporal model. Use ",
       "spatial_rsr() for the spatial analogue, which the binomial ",
       "Polya-Gamma Gibbs sampler carries.", call. = FALSE)
}



#' Extract temporal effects from a fitted model
#'
#' @description
#' Extract posterior distributions of temporal effects from a fitted tulpa
#' model with temporal specification.
#'
#' @details
#' `temporal()` is overloaded. Given a fitted model it is the accessor described
#' here. Given a one-sided formula (or a named `formula =` / `structure =`
#' argument) it is instead the inline varying-coefficient field constructor used
#' in a `tulpa()` model formula, the temporal mirror of [spatial()]:
#' `temporal(formula = ~ 1 + x || time, structure = "rw1")` declares a smooth
#' temporal level (the intercept column) plus a temporally varying slope on each
#' covariate column. `structure` is one of `"rw1"` (default), `"rw2"`, or
#' `"ar1"`; only the double bar `||` (independent fields) is supported.
#'
#' On a sampler fit the posterior is the field's own draws. On a
#' nested-Laplace fit it is the Gaussian mixture the outer grid defines: each
#' hyperparameter cell contributes its conditional mode and marginal variance,
#' weighted by the cell's posterior weight, and `summary()` reports that
#' mixture's exact mean, SD and quantiles. The `draws` of such a fit are
#' sampled from the same mixture; they reproduce each time point's marginal
#' but not the within-cell correlation between time points. The `time` column
#' of the summary holds the time values the field was fitted on.
#'
#' @param object A `tulpa_fit` object fitted with `temporal` argument
#' @param component Which component to extract for multi-scale models:
#'   `"all"` (default), `"trend"`, `"seasonal"`, or `"short_term"`.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `tulpa_temporal_posterior` object
#'
#' @examples
#' \donttest{
#' set.seed(131)
#' df <- data.frame(year = 1:40, x = rnorm(40))
#' df$count <- rpois(40, exp(1 + 0.2 * df$x))
#'
#' fit <- tulpa(
#'   count ~ x,
#'   data = df,
#'   family = "poisson",
#'   temporal = temporal_multiscale("year", trend = "rw2", seasonal = 12),
#'   mode = "exact",
#'   control = list(n_iter = 200L, n_warmup = 100L, seed = 1L)
#' )
#'
#' # Extract all temporal effects
#' temp_post <- temporal(fit)
#' summary(temp_post)
#' }
#'
#' @seealso [temporal_multiscale()], [temporal_rw1()]
#'
#' @export
temporal <- function(object, component = "all", summary = FALSE,
                     probs = c(0.025, 0.5, 0.975), ...) {
  # Overloaded: with a formula first argument (or a named `formula =` /
  # `structure =`), `temporal()` is the inline varying-coefficient field
  # constructor used in a tulpa() model formula -- temporal(formula = ~ 1 + x ||
  # time, structure = "rw1") -- mirroring spatial(). Otherwise it is the
  # temporal-posterior accessor on a fitted model (temporal(fit)).
  if (!missing(object) && inherits(object, "formula")) {
    return(.temporal_field_spec(object, ...))
  }
  if (missing(object)) {
    return(.temporal_field_spec(...))
  }
  UseMethod("temporal")
}


# The temporal field read back off the fit's own draws. The sampler names a
# multiscale block by its component (`trend[k]` / `seasonal[k]` /
# `short_term[k]`) and a single field `phi_temporal[k]`
# (src/sampler_model_data.h); neither carries the layout, which is why the
# validated spec is read alongside. NULL when any active component is absent,
# so a partial read surfaces as "draws not found" instead of a short list.
.temporal_draws_from_fit <- function(object, info) {
  if (!inherits(info, "tulpa_temporal_multiscale")) {
    return(.draws_by_prefix(object$draws, "phi_temporal"))
  }
  out <- lapply(info$components, function(cmp)
    .draws_by_prefix(object$draws, cmp))
  names(out) <- info$components
  if (!length(out) || any(vapply(out, is.null, logical(1)))) return(NULL)
  out
}


# A nested-Laplace fit carries no phi_temporal draws: the field is a block of
# each outer-grid cell's latent vector, and the cell's Laplace approximation
# is a Gaussian with that cell's mode (`object$modes`, one row per cell) and
# its marginal variance (`object$grid_field_var`, retained for the temporal
# field by `.nl_attach_grid_hessians()`). The posterior is their mixture over
# the grid weights. Reading the cell modes alone carried only the spread
# BETWEEN cells: sd ~0.05 against an exact ~0.13-0.48, and 95% intervals that
# covered the truth 57% of the time (gcol33/tulpa#904).
#
# `.nl_field_mixture()` returns the components (mu, var: n_cell x width; w);
# summary() reads them exactly through `.nl_gauss_mixture_summary()`, and the
# draws print / plot / a user's own derived quantity go through are sampled
# from the same mixture -- a cell by its weight, then each coordinate from
# that cell's Gaussian. The cell's cross-time covariance is not retained, so a
# draw has the right marginal at every time point but not the within-cell
# correlation between two of them.
.NL_TEMPORAL_DRAW_N <- 2000L

.nl_field_mixture <- function(object, cols) {
  M <- object$modes
  w <- object$weights
  if (!is.matrix(M) || is.null(w) || length(w) != nrow(M)) return(NULL)
  if (!length(cols) || max(cols) > ncol(M)) return(NULL)
  V <- object$grid_field_var
  vcols <- object$grid_field_var_cols
  var <- if (is.matrix(V) && nrow(V) == nrow(M) && all(cols %in% vcols)) {
    V[, match(cols, vcols), drop = FALSE]
  } else NULL
  w <- w / sum(w)
  keep <- is.finite(w) & w > 0 & is.finite(rowSums(M[, cols, drop = FALSE]))
  if (!is.null(var)) keep <- keep & is.finite(rowSums(var))
  if (!any(keep)) return(NULL)
  list(mu = M[keep, cols, drop = FALSE],
       var = if (is.null(var)) NULL else pmax(var[keep, , drop = FALSE], 0),
       w = w[keep] / sum(w[keep]))
}

.nl_field_mixture_draws <- function(mix) {
  idx <- sample.int(nrow(mix$mu), size = .NL_TEMPORAL_DRAW_N, replace = TRUE,
                    prob = mix$w)
  mu <- mix$mu[idx, , drop = FALSE]
  if (is.null(mix$var)) return(mu)
  mu + sqrt(mix$var[idx, , drop = FALSE]) *
    matrix(stats::rnorm(length(mu)), nrow(mu), ncol(mu))
}

# Columns of the `temporal =` field in a nested fit's latent vector, located
# through the fit's own block layout (`.nl_role_cols()`), never at a fixed
# offset: behind a spatial block the slice right after the fixed effects IS the
# spatial field (gcol33/tulpa#903). fit_st_nested() records its own layout.
.nl_temporal_field_cols <- function(object) {
  if (!is.null(object$field_cols$temporal)) return(object$field_cols$temporal)
  .nl_role_cols(object$prior, object$block_latent_offsets,
                if (is.matrix(object$modes)) ncol(object$modes), "temporal")
}

# `latent(temporal_ar2(...))` / `latent(temporal_ar(...))` build a `tgmrf`
# block tagged `tulpa_temporal_latent_block` (R/temporal_ar2.R) rather than
# filling `object$temporal`, so this is the second place a nested fit's field
# is found: the first such block in `object$blocks`, at the column the solve
# reported for it (`block_latent_offsets`) -- summing the earlier blocks'
# `n_latent` held only when every earlier block was itself a tgmrf. Returns
# NULL when the fit carries no such block.
.nl_temporal_latent_block <- function(object) {
  blocks <- object$blocks
  if (!is.list(blocks) || !length(blocks)) return(NULL)
  is_temporal <- vapply(blocks, inherits, logical(1),
                         what = "tulpa_temporal_latent_block")
  if (!any(is_temporal)) return(NULL)
  k <- which(is_temporal)[1L]
  blk <- blocks[[k]]
  off <- object$block_latent_offsets
  start <- if (length(off) == length(blocks) + 1L) off[k] else {
    (object$n_fixed %||% 0L) + if (k > 1L) sum(vapply(
      blocks[seq_len(k - 1L)], function(b) b$n_latent %||% 0L, numeric(1))) else 0
  }
  list(
    info = structure(
      list(n_times = blk$n_times %||% blk$n_latent, n_groups = 1L,
           type = blk$name %||% "ar", time_levels = NULL),
      class = "tulpa_temporal"
    ),
    cols = as.integer(start) + seq_len(as.integer(blk$n_latent %||% 0L))
  )
}


#' @rdname temporal
#' @export
temporal.tulpa_fit <- function(object, component = "all", summary = FALSE,
                                probs = c(0.025, 0.5, 0.975), ...) {

  latent_blk <- NULL
  if (is.null(object$temporal)) {
    latent_blk <- .nl_temporal_latent_block(object)
    if (is.null(latent_blk)) {
      stop("Model was not fitted with temporal effects.\n",
           "Pass `temporal = temporal_rw1() / temporal_rw2() / temporal_ar1() / ",
           "temporal_multiscale(...)` to tulpa().", call. = FALSE)
    }
  }

  temp_info <- if (!is.null(latent_blk)) latent_blk$info else object$temporal

  # A TVC spec rides the same `temporal =` slot, and carries varying
  # coefficients rather than a temporal field, so it has no phi_temporal to
  # summarize. Name the accessor that reads it instead of failing further down
  # on the absent draws.
  if (inherits(temp_info, "tulpa_tvc")) {
    stop("This fit carries temporally-varying coefficients, not a temporal ",
         "field. Use tvc() to read them.", call. = FALSE)
  }

  # Get temporal draws from model
  temp_draws <- object$.internal$temporal_draws
  if (is.null(temp_draws)) {
    temp_draws <- .temporal_draws_from_fit(object, temp_info)
  }

  # No phi_temporal draws: on a nested-Laplace fit the field is read off the
  # grid instead, as the per-cell Gaussian mixture of either the `temporal =`
  # block or the tagged `latent(temporal_ar*())` block located above.
  mixture <- NULL
  if (is.null(temp_draws) && !inherits(temp_info, "tulpa_temporal_multiscale")) {
    cols <- if (!is.null(latent_blk)) latent_blk$cols else .nl_temporal_field_cols(object)
    width <- (temp_info$n_times %||% 0L) * max(temp_info$n_groups %||% 1L, 1L)
    if (is.null(latent_blk) && length(cols) != width) cols <- NULL
    mixture <- if (length(cols)) .nl_field_mixture(object, cols)
    if (!is.null(mixture)) {
      if (is.null(mixture$var)) {
        warning("This nested-Laplace fit retained no per-cell variance for ",
                "the temporal field (refit with the default ",
                "`control$keep_grid_hessians = TRUE`), so its temporal() ",
                "intervals carry only the spread between grid cells and are ",
                "too narrow.", call. = FALSE)
      }
      temp_draws <- .nl_field_mixture_draws(mixture)
    }
  }

  if (is.null(temp_draws)) {
    stop("Temporal draws not found in model output", call. = FALSE)
  }

  # Handle multi-scale vs single-component
  if (inherits(temp_info, "tulpa_temporal_multiscale")) {
    available_components <- temp_info$components

    if (component != "all" && !(component %in% available_components)) {
      stop(sprintf("Component '%s' not in model. Available: %s",
                   component, paste(available_components, collapse = ", ")),
           call. = FALSE)
    }

    if (component != "all") {
      # Subset to requested component
      temp_draws <- temp_draws[[component]]
    }
  }

  result <- structure(
    list(
      draws = temp_draws,
      time_levels = temp_info$time_levels,
      n_times = temp_info$n_times,
      n_groups = temp_info$n_groups,
      n_draws = if (is.list(temp_draws)) dim(temp_draws[[1]])[1] else dim(temp_draws)[1],
      type = temp_info$type,
      components = if (inherits(temp_info, "tulpa_temporal_multiscale"))
        temp_info$components else temp_info$type,
      component_requested = component,
      # A nested fit's exact posterior, which summary() reads in place of the
      # sampled draws.
      mixture = mixture
    ),
    class = "tulpa_temporal_posterior"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}


#' Print method for tulpa_temporal_posterior
#'
#' @param x A tulpa_temporal_posterior object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing a summary of the temporal-effect posterior to the console.
#'
#' @export
print.tulpa_temporal_posterior <- function(x, ...) {
  cat("Temporal effect posterior\n")
  cat("=========================\n\n")

  if (x$type == "multiscale") {
    cat("Type: Multi-scale decomposition\n")
    cat("Components:", paste(x$components, collapse = ", "), "\n")
    if (x$component_requested != "all") {
      cat("Extracted:", x$component_requested, "\n")
    }
  } else {
    cat("Type:", toupper(x$type), "\n")
  }

  cat("Time points:", x$n_times, "\n")
  if (x$n_groups > 1) {
    cat("Groups:", x$n_groups, "\n")
  }
  cat("Posterior draws:", x$n_draws, "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for visualization\n")
  invisible(x)
}


#' Summary method for tulpa_temporal_posterior
#'
#' @param object A tulpa_temporal_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @return A `tulpa_temporal_summary` data frame with one row per time point
#'   (per component for multi-scale fits), holding the posterior mean, SD, and
#'   requested quantiles of the temporal effect.
#'
#' @export
summary.tulpa_temporal_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {

  # Handle multi-scale (list) vs single (matrix)
  if (is.list(object$draws) && !is.data.frame(object$draws)) {
    # Multi-scale: summarize each component
    results <- list()

    for (comp in names(object$draws)) {
      draws <- object$draws[[comp]]
      n_times <- ncol(draws)

      summaries <- data.frame(
        component = comp,
        time_idx = seq_len(n_times),
        time = if (!is.null(object$time_levels) && n_times == length(object$time_levels))
          object$time_levels else seq_len(n_times),
        mean = colMeans(draws),
        sd = apply(draws, 2, sd),
        t(apply(draws, 2, quantile, probs = probs))
      )
      names(summaries)[6:ncol(summaries)] <- paste0("q", probs * 100)
      rownames(summaries) <- NULL
      results[[comp]] <- summaries
    }

    result <- do.call(rbind, results)
  } else {
    # Single component
    draws <- object$draws
    n_times <- ncol(draws)
    # A nested fit's mixture is summarized exactly (moments by the law of total
    # variance, quantiles by inverting the mixture CDF) rather than off draws
    # sampled from it; without per-cell variances the mixture gives no SD, and
    # the draws (cell modes alone) are what is left to read.
    mx <- if (!is.null(object$mixture$var)) {
      .nl_gauss_mixture_summary(object$mixture$mu, object$mixture$var,
                                object$mixture$w, probs = probs)
    }
    stats_tbl <- if (!is.null(mx)) {
      data.frame(mean = mx$mean, sd = mx$sd, mx$quantiles)
    } else {
      data.frame(mean = colMeans(draws), sd = apply(draws, 2, sd),
                 t(apply(draws, 2, quantile, probs = probs)))
    }

    result <- data.frame(
      time_idx = seq_len(n_times),
      time = if (!is.null(object$time_levels)) object$time_levels else seq_len(n_times),
      stats_tbl
    )
    names(result)[5:ncol(result)] <- paste0("q", probs * 100)
    rownames(result) <- NULL
  }

  structure(
    result,
    n_draws = object$n_draws,
    class = c("tulpa_temporal_summary", "data.frame")
  )
}


#' Plot method for tulpa_temporal_posterior
#'
#' @param x A tulpa_temporal_posterior object
#' @param component Which component to plot (for multi-scale). Default: first.
#' @param type Plot type: "ribbon" (default) or "line"
#' @param ... Additional arguments passed to plotting functions
#'
#' @return A `ggplot` object when ggplot2 is installed; otherwise `NULL`
#'   invisibly, after drawing a base-graphics plot. Called for the side effect
#'   of plotting the temporal-effect posterior.
#'
#' @export
plot.tulpa_temporal_posterior <- function(x, component = NULL, type = "ribbon", ...) {

  # Get draws to plot
  if (is.list(x$draws) && !is.data.frame(x$draws)) {
    if (is.null(component)) {
      component <- names(x$draws)[1]
    }
    draws <- x$draws[[component]]
    title <- paste("Temporal effect:", component)
  } else {
    draws <- x$draws
    title <- paste("Temporal effect:", x$type)
  }

  n_times <- ncol(draws)
  times <- if (!is.null(x$time_levels) && n_times == length(x$time_levels)) {
    as.numeric(x$time_levels)
  } else {
    seq_len(n_times)
  }

  # Compute summaries
  means <- colMeans(draws)
  lower <- apply(draws, 2, quantile, probs = 0.025)
  upper <- apply(draws, 2, quantile, probs = 0.975)

  # Use ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      time = times,
      mean = means,
      lower = lower,
      upper = upper
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$time, y = .data$mean)) +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$lower, ymax = .data$upper),
        alpha = 0.3, fill = "steelblue"
      ) +
      ggplot2::geom_line(color = "steelblue", linewidth = 1) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      ggplot2::labs(
        title = title,
        x = "Time",
        y = "Effect"
      ) +
      theme_tulpa()

    return(p)
  }

  # Base R fallback
  ylim <- range(c(lower, upper))

  plot(times, means, type = "l", col = "steelblue", lwd = 2,
       ylim = ylim, xlab = "Time", ylab = "Effect",
       main = title, ...)

  polygon(c(times, rev(times)), c(lower, rev(upper)),
          col = adjustcolor("steelblue", alpha.f = 0.3), border = NA)

  abline(h = 0, lty = 2, col = "gray50")
  lines(times, means, col = "steelblue", lwd = 2)

  invisible(NULL)
}
