#' Time-varying coefficient structure
#'
#' @description
#' Specify a time-varying coefficient (TVC): one or more fixed-effect
#' coefficients are allowed to evolve over time, with the evolution governed by
#' a temporal prior (`rw1`, `rw2`, `ar1`, or a GP).
#'
#' @param time_var Single character string naming the time variable in the data.
#' @param terms Which coefficients vary over time. A formula, an integer vector
#'   of design-matrix column indices, or a character vector of term names.
#'   Default `1` (the intercept).
#' @param structure Temporal prior governing how the coefficients evolve. One of
#'   `"rw1"`, `"rw2"`, `"ar1"`, or `"gp"`.
#' @param group_var Optional character string naming a grouping variable for
#'   group-specific time-varying coefficients.
#' @param shared Whether the effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares it; `FALSE` fits
#'   process-specific effects and emits a warning.
#'
#' @return A `tulpa_tvc` object.
#'
#' @seealso [temporal_rw1()], [temporal_rw2()], [temporal_ar1()] for the
#'   underlying temporal priors.
#'
#' @examples
#' # Intercept that drifts as a first-order random walk over year
#' temporal_tvc("year", structure = "rw1")
#'
#' @export
temporal_tvc <- function(time_var,
                         terms = 1,
                         structure = c("rw1", "rw2", "ar1", "gp"),
                         group_var = NULL,
                         shared = NULL) {

  structure_type <- match.arg(structure)

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  if (!is.null(group_var)) {
    if (!is.character(group_var) || length(group_var) != 1) {
      stop("`group_var` must be a single character string", call. = FALSE)
    }
  }

  # Parse terms specification
  if (inherits(terms, "formula")) {
    terms_spec <- list(type = "formula", formula = terms)
  } else if (is.numeric(terms)) {
    terms_spec <- list(type = "index", indices = as.integer(terms))
  } else if (is.character(terms)) {
    terms_spec <- list(type = "names", names = terms)
  } else {
    stop("`terms` must be a formula, integer vector, or character vector",
         call. = FALSE)
  }

  # Warning for non-shared TVCs
  if (isFALSE(shared)) {
    warning(
      "Non-shared TVCs (shared = FALSE) means effects are not shared across processes.\n",
      "Consider whether time-varying effects should be shared between\n",
      "processes if shared confounding structure is expected.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "tvc",
      time_var = time_var,
      group_var = group_var,
      terms_spec = terms_spec,
      structure = structure_type,
      shared = shared,
      # Filled in during validation
      n_times = NULL,
      n_groups = NULL,
      n_tvc = NULL,
      tvc_indices = NULL,
      tvc_names = NULL,
      time_index = NULL,
      group_index = NULL
    ),
    class = c("tulpa_tvc", "tulpa_temporal", "list")
  )
}


#' Print method for tulpa_tvc
#'
#' @param x A tulpa_tvc object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the temporally-varying-coefficient specification to the console.
#'
#' @export
print.tulpa_tvc <- function(x, ...) {
  cat("tulpa temporally-varying coefficients\n")
  cat("======================================\n\n")

  cat("Time variable:", x$time_var, "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  struct_name <- switch(x$structure,
    rw1 = "RW1 (first-order random walk)",
    rw2 = "RW2 (second-order random walk)",
    ar1 = "AR(1) (autoregressive)",
    gp = "GP (Gaussian process)"
  )
  cat("Structure:", struct_name, "\n")
  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_tvc)) {
    cat("\nTVC terms:", x$n_tvc, "\n")
    if (!is.null(x$tvc_names)) {
      cat("  ", paste(x$tvc_names, collapse = ", "), "\n")
    }
  } else {
    cat("\nTerms: ")
    if (x$terms_spec$type == "formula") {
      cat(deparse(x$terms_spec$formula), "\n")
    } else if (x$terms_spec$type == "index") {
      cat("columns ", paste(x$terms_spec$indices, collapse = ", "), "\n")
    } else {
      cat(paste(x$terms_spec$names, collapse = ", "), "\n")
    }
  }

  if (!is.null(x$n_times)) {
    cat("Time points:", x$n_times, "\n")
  }

  invisible(x)
}


#' Validate TVC specification against data and design matrix
#'
#' @param tvc tulpa_tvc object
#' @param data Data frame
#' @param X Design matrix (to resolve term names)
#'
#' @return Updated tulpa_tvc object with computed structure
#' @keywords internal
validate_tvc <- function(tvc, data, X) {
  if (is.null(tvc)) return(NULL)
  if (!inherits(tvc, "tulpa_tvc")) return(tvc)

  N <- nrow(data)
  p <- ncol(X)

  # Check time variable exists
  if (!(tvc$time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", tvc$time_var),
         call. = FALSE)
  }

  # Get time values and create indices
  time_vals <- data[[tvc$time_var]]
  if (is.factor(time_vals)) {
    time_factor <- time_vals
  } else {
    unique_times <- sort(unique(time_vals))
    time_factor <- factor(time_vals, levels = unique_times)
  }

  tvc$n_times <- nlevels(time_factor)
  tvc$time_index <- as.integer(time_factor)
  tvc$time_levels <- levels(time_factor)

  # Handle grouping
  if (!is.null(tvc$group_var)) {
    if (!(tvc$group_var %in% names(data))) {
      stop(sprintf("Group variable '%s' not found in data", tvc$group_var),
           call. = FALSE)
    }
    group_vals <- data[[tvc$group_var]]
    group_factor <- as.factor(group_vals)
    tvc$n_groups <- nlevels(group_factor)
    tvc$group_index <- as.integer(group_factor)
    tvc$group_levels <- levels(group_factor)
  } else {
    tvc$n_groups <- 1L
    tvc$group_index <- rep(1L, N)
  }

  # Resolve TVC terms against design matrix
  coef_names <- colnames(X)
  if (is.null(coef_names)) {
    coef_names <- paste0("V", seq_len(p))
  }

  if (tvc$terms_spec$type == "index") {
    tvc_indices <- tvc$terms_spec$indices
    if (any(tvc_indices < 1 | tvc_indices > p)) {
      stop(sprintf("TVC term indices must be between 1 and %d", p),
           call. = FALSE)
    }
    tvc_names <- coef_names[tvc_indices]

  } else if (tvc$terms_spec$type == "names") {
    tvc_names <- tvc$terms_spec$names
    tvc_indices <- match(tvc_names, coef_names)
    if (any(is.na(tvc_indices))) {
      missing <- tvc_names[is.na(tvc_indices)]
      stop(sprintf("TVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }

  } else if (tvc$terms_spec$type == "formula") {
    fmla <- tvc$terms_spec$formula
    tt <- terms(fmla)
    term_labels <- attr(tt, "term.labels")
    has_intercept <- attr(tt, "intercept") == 1

    tvc_names <- character(0)
    if (has_intercept) {
      tvc_names <- c(tvc_names, "(Intercept)")
    }
    tvc_names <- c(tvc_names, term_labels)

    tvc_indices <- match(tvc_names, coef_names)
    if (any(is.na(tvc_indices))) {
      missing <- tvc_names[is.na(tvc_indices)]
      stop(sprintf("TVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }
  }

  tvc$n_tvc <- length(tvc_indices)
  tvc$tvc_indices <- tvc_indices
  tvc$tvc_names <- tvc_names

  # Store design matrix subset for TVC terms
  tvc$X_tvc <- X[, tvc_indices, drop = FALSE]

  # Total TVC parameters = n_times * n_tvc * n_groups
  tvc$n_temporal_params <- tvc$n_times * tvc$n_tvc * tvc$n_groups

  tvc
}


#' Extract temporally-varying coefficients from a fitted model
#'
#' @description
#' Extract posterior distributions of temporally-varying coefficients (TVCs)
#' from a fitted tulpa model with TVC specification.
#'
#' @param object A `tulpa_fit` object fitted with `tvc` argument
#' @param terms Which TVC terms to extract. If NULL (default), extracts all.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `tulpa_tvc_posterior` object containing:
#' - `draws`: Array of posterior draws (draws x times x terms)
#' - `time_levels`: Time point labels
#' - `term_names`: Names of TVC terms
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data (not run - TVC experimental)
#' set.seed(160)
#' df <- data.frame(
#'   year = rep(2015:2024, each = 5),
#'   x = rnorm(50),
#'   count = rpois(50, lambda = 18),
#'   effort = rgamma(50, shape = 3.5, rate = 1)
#' )
#'
#' # Fit model with TVC
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpaRatio::tulpa_poisson_gamma(),
#'   tvc = temporal_tvc("year", terms = c(1, 2)),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Extract TVC posteriors
#' tvc_post <- tvc(fit)
#' summary(tvc_post)
#'
#' # Plot temporal evolution
#' plot(tvc_post, "x")
#' }
#'
#' @seealso [temporal_tvc()], [plot.tulpa_tvc_posterior()]
#'
#' @export
tvc <- function(object, terms = NULL, summary = FALSE,
                probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("tvc")
}


#' @rdname tvc
#' @export
tvc.tulpa_fit <- function(object, terms = NULL, summary = FALSE,
                           probs = c(0.025, 0.5, 0.975), ...) {
  .extract_varying_coef(
    object, terms, summary, probs,
    slot = "tvc",
    info_class = "tulpa_tvc",
    not_fitted_msg = paste0(
      "Model was not fitted with temporally-varying coefficients.\n",
      "Use `tvc` argument in tulpa() to specify TVCs."),
    draws_field = "tvc_draws",
    names_field = "tvc_names",
    build_result = function(info, draws, term_names) {
      structure(
        list(
          draws = draws,
          time_levels = info$time_levels,
          term_names = term_names,
          n_times = info$n_times,
          n_tvc = length(term_names),
          n_draws = dim(draws)[1],
          structure = info$structure
        ),
        class = "tulpa_tvc_posterior"
      )
    }
  )
}


#' Print method for tulpa_tvc_posterior
#'
#' @param x A tulpa_tvc_posterior object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing a summary of the temporally-varying-coefficient posterior to the
#'   console.
#'
#' @export
print.tulpa_tvc_posterior <- function(x, ...) {
  .print_varying_coef(
    x, "Temporally-varying",
    axis_label = "Time points", axis_value = x$n_times,
    meta_label = "Structure", meta_value = x$structure,
    viz = "temporal"
  )
}


#' Summary method for tulpa_tvc_posterior
#'
#' @param object A tulpa_tvc_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @return A `tulpa_tvc_summary` data frame with one row per time point and
#'   term, holding the posterior mean, SD, and requested quantiles of each
#'   temporally-varying coefficient.
#'
#' @export
summary.tulpa_tvc_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {
  .summary_varying_coef(
    object, probs,
    n_terms = object$n_tvc,
    lead_cols = function(j) data.frame(
      time_idx = seq_len(object$n_times),
      time = object$time_levels,
      term = object$term_names[j]
    ),
    summary_class = "tulpa_tvc_summary"
  )
}


#' Plot method for tulpa_tvc_posterior
#'
#' @param x A tulpa_tvc_posterior object
#' @param term Which term to plot (name or index). Default: first term.
#' @param type Plot type: "ribbon" (default) or "line"
#' @param ... Additional arguments passed to plotting functions
#'
#' @return A `ggplot` object when ggplot2 is installed; otherwise `NULL`
#'   invisibly, after drawing a base-graphics plot. Called for the side effect
#'   of plotting the selected temporally-varying coefficient.
#'
#' @export
plot.tulpa_tvc_posterior <- function(x, term = 1, type = "ribbon", ...) {

  if (is.character(term)) {
    term_idx <- match(term, x$term_names)
    if (is.na(term_idx)) {
      stop("Term not found: ", term, call. = FALSE)
    }
  } else {
    term_idx <- term
  }

  term_name <- x$term_names[term_idx]
  draws <- x$draws[, , term_idx]

  n_times <- x$n_times
  times <- as.numeric(x$time_levels)

  # Compute summaries
  means <- colMeans(draws)
  lower <- apply(draws, 2, quantile, probs = 0.025)
  upper <- apply(draws, 2, quantile, probs = 0.975)

  title <- paste("TVC:", term_name)

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
        y = "Time-Varying Effect"
      ) +
      theme_tulpa()

    return(p)
  }

  # Base R fallback
  ylim <- range(c(lower, upper))

  plot(times, means, type = "l", col = "steelblue", lwd = 2,
       ylim = ylim, xlab = "Time", ylab = "Time-Varying Effect",
       main = title, ...)

  polygon(c(times, rev(times)), c(lower, rev(upper)),
          col = adjustcolor("steelblue", alpha.f = 0.3), border = NA)

  abline(h = 0, lty = 2, col = "gray50")
  lines(times, means, col = "steelblue", lwd = 2)

  invisible(NULL)
}


# =============================================================================
# Restricted Temporal Regression (RTR)
# =============================================================================

#' Restricted Temporal Regression (RTR)
#'
#' @description
#' Apply Restricted Temporal Regression to mitigate temporal confounding.
#' RTR orthogonalizes the temporal effect to the covariate space, preventing
#' the temporal random effect from absorbing covariate information.
#'
#' This is important when covariates have temporal trends (e.g., increasing
#' over time) because the temporal random effect can "steal" variance from
#' these covariates, leading to biased coefficient estimates.
#'
#' @param temporal A temporal specification (`temporal_rw1`, `temporal_gp`, etc.)
#' @param restrict_to Formula specifying which covariates to orthogonalize
#'   against (e.g., `~ year_effect + trend`). The temporal effect will be
#'   constrained to be orthogonal to the column space of these covariates.
#'
#' @return A modified temporal specification with RTR enabled
#'
#' @details
#' The RTR approach (analogous to Restricted Spatial Regression) modifies the
#' temporal random effect to be orthogonal to the fixed effect design matrix:
#'
#' \deqn{f_{RTR}(t) = (I - P_X) f(t)}
#'
#' where \eqn{P_X = X(X'X)^{-1}X'} is the projection matrix onto the column
#' space of X.
#'
#' **When to use RTR:**
#' - Covariates have temporal trends (e.g., increasing/decreasing over time)
#' - Interested in causal interpretation of covariate effects
#' - Coefficients appear attenuated toward zero
#' - Time and covariates are correlated
#'
#' **When NOT to use RTR:**
#' - Covariates are temporally uncorrelated (random sampling over time)
#' - Temporal effect is the primary quantity of interest
#' - Prediction is the main goal (not causal inference)
#'
#' @examples
#' # Create RTR temporal structure
#' rtr <- temporal_rtr(
#'   temporal_rw2("year"),
#'   restrict_to = ~ temperature + precipitation
#' )
#' print(rtr)
#'
#' \dontrun{
#' # Generate data with temporal confounding (not run - RTR experimental)
#' set.seed(170)
#' years <- 2000:2023
#' n_per_year <- 4
#' df <- data.frame(
#'   year = rep(years, each = n_per_year),
#'   # Temperature increasing over time (confounded with year)
#'   temperature = rep(seq(15, 18, length.out = length(years)), each = n_per_year) +
#'                 rnorm(length(years) * n_per_year, 0, 0.5),
#'   count = rpois(length(years) * n_per_year, lambda = 20),
#'   effort = rgamma(length(years) * n_per_year, shape = 4, rate = 1)
#' )
#'
#' # Standard RW2 (may have temporal confounding)
#' fit1 <- tulpa(
#'   count | effort ~ temperature,
#'   data = df,
#'   temporal = temporal_rw2("year"),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # RTR to protect temperature coefficient
#' fit2 <- tulpa(
#'   count | effort ~ temperature,
#'   data = df,
#'   temporal = temporal_rtr(
#'     temporal_rw2("year"),
#'     restrict_to = ~ temperature
#'   ),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Compare coefficient estimates
#' summary(fit1)  # May be attenuated
#' summary(fit2)  # Protected from temporal confounding
#' }
#'
#' @references
#' Hanks, E. M., Schliep, E. M., Hooten, M. B., & Hoeting, J. A. (2015).
#' Restricted spatial regression in practice: geostatistical models, confounding,
#' and robustness under model misspecification. Environmetrics, 26(4), 243-254.
#'
#' @seealso [temporal_rw1()], [temporal_rw2()], [temporal_gp()],
#'   [spatial_rsr()] for the spatial analog
#'
#' @export
