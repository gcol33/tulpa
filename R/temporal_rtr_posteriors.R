temporal_rtr <- function(temporal, restrict_to) {

  if (!inherits(temporal, "tulpa_temporal")) {
    stop("`temporal` must be a tulpa temporal specification", call. = FALSE)
  }

  if (!inherits(restrict_to, "formula")) {
    stop("`restrict_to` must be a formula", call. = FALSE)
  }

  # Store RTR information in the temporal object
  temporal$rtr <- TRUE
  temporal$rtr_formula <- restrict_to

  # Add RTR class for dispatch
  class(temporal) <- c("tulpa_rtr", class(temporal))

  temporal
}


#' Print method for tulpa_rtr (temporal)
#'
#' @param x A tulpa_rtr object
#' @param ... Passed to underlying print method
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the temporal specification and its restricted-temporal-regression
#'   modifier to the console.
#'
#' @export
print.tulpa_rtr <- function(x, ...) {
  # Print underlying temporal type
  NextMethod()

  cat("\nRestricted Temporal Regression (RTR):\n")
  cat("  Orthogonal to:", deparse(x$rtr_formula), "\n")
  cat("  (Temporal effect constrained to be orthogonal to covariate space)\n")

  invisible(x)
}


#' Validate RTR specification
#'
#' @param temporal tulpa_rtr object
#' @param data Data frame
#' @param formula Model formula (to extract design matrix)
#'
#' @return Updated temporal object with projection matrix
#' @keywords internal
validate_rtr <- function(temporal, data, formula) {
  if (is.null(temporal) || !inherits(temporal, "tulpa_rtr")) {
    return(temporal)
  }

  # Build design matrix for RTR covariates
  rtr_formula <- temporal$rtr_formula

  # Check if terms exist in data
  rtr_vars <- all.vars(rtr_formula)
  missing_vars <- setdiff(rtr_vars, names(data))
  if (length(missing_vars) > 0) {
    stop(sprintf("RTR variables not found in data: %s",
                 paste(missing_vars, collapse = ", ")), call. = FALSE)
  }

  # Build design matrix
  X_rtr <- model.matrix(rtr_formula, data = data)

  # Compute projection matrix (using QR for numerical stability)
  temporal$rtr_projection <- compute_rtr_projection(X_rtr)
  temporal$rtr_vars <- rtr_vars

  temporal
}


#' Compute RTR projection matrix
#'
#' @description
#' Compute the orthogonal projection matrix P_perp = I - P_X that projects
#' the temporal effect into the space orthogonal to the covariates.
#'
#' @param X Design matrix of covariates to orthogonalize against
#'
#' @return Projection matrix (n x n)
#' @keywords internal
compute_rtr_projection <- function(X) {
  .orthogonal_complement_projection(X, "RTR")
}


#' Apply RTR projection to temporal effect
#'
#' @description
#' Project temporal effect into the space orthogonal to covariates.
#' Called during posterior computation.
#'
#' @param f Temporal effect vector (length n)
#' @param P_perp Projection matrix from compute_rtr_projection
#'
#' @return Projected temporal effect (length n)
#' @keywords internal
apply_rtr_projection <- function(f, P_perp) {
  as.vector(P_perp %*% f)
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
#' \dontrun{
#' # Fit model with multi-scale temporal effects (not run - experimental)
#' set.seed(131)
#' df <- data.frame(
#'   year = 1:40,
#'   x = rnorm(40),
#'   count = rpois(40, lambda = 22),
#'   effort = rgamma(40, shape = 4.5, rate = 1)
#' )
#'
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   temporal = temporal_multiscale("year", trend = "rw2", seasonal = 12),
#'   iter = 200, warmup = 100, chains = 1
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


#' @rdname temporal
#' @export
temporal.tulpa_fit <- function(object, component = "all", summary = FALSE,
                                probs = c(0.025, 0.5, 0.975), ...) {

  # Check if model has temporal effects
  if (is.null(object$temporal)) {
    stop("Model was not fitted with temporal effects.\n",
         "Use `temporal` argument in tulpa() to specify temporal structure.",
         call. = FALSE)
  }

  temp_info <- object$temporal

  # Get temporal draws from model
  temp_draws <- object$.internal$temporal_draws

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
      component_requested = component
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

    result <- data.frame(
      time_idx = seq_len(n_times),
      time = if (!is.null(object$time_levels)) object$time_levels else seq_len(n_times),
      mean = colMeans(draws),
      sd = apply(draws, 2, sd),
      t(apply(draws, 2, quantile, probs = probs))
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
