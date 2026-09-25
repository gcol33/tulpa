#' Time-varying coefficient structure
#'
#' @description
#' Specify a time-varying coefficient (TVC): one or more fixed-effect
#' coefficients are allowed to evolve over time, with the evolution governed by
#' a temporal prior (`rw1`, `rw2`, `ar1` or `gp`).
#'
#' `rw1`, `rw2` and `ar1` read the time index as a position on a grid, so
#' consecutive instants are one step apart whatever the data says. `gp` is the
#' continuous-time structure: the coefficient is a Gaussian process over the
#' distinct time VALUES, which is what irregular spacing needs. It is distinct
#' from [temporal_gp()], a GP over time entering the linear predictor additively
#' (`eta_i += f(t_i)`); here the GP IS a coefficient (`eta_i += x_i w(t_i)`).
#'
#' @param time_var Single character string naming the time variable in the data.
#'   `structure = "gp"` needs it numeric: a factor states an ordering with no
#'   spacing for the kernel to measure.
#' @param terms Which coefficients vary over time. A formula, an integer vector
#'   of design-matrix column indices, or a character vector of term names.
#'   Default `1` (the intercept).
#' @param structure Temporal prior governing how the coefficients evolve. One of
#'   `"rw1"`, `"rw2"`, `"ar1"` or `"gp"`.
#' @param cov,nu,period Covariance kernel for `structure = "gp"`, ignored
#'   otherwise. `cov` is one of `"exponential"` (the default), `"matern"`,
#'   `"gaussian"` or `"periodic"`; `nu` is the Matern smoothness, closed-form at
#'   `0.5`, `1.5` and `2.5` only (`0.5` IS the exponential kernel); `period` is
#'   the periodic kernel's period. Exponential and Matern `nu = 0.5` evaluate in
#'   `O(T)` through the Ornstein-Uhlenbeck factorization; the rest take a dense
#'   `T x T` Cholesky per coefficient per gradient evaluation.
#' @param group_var Optional character string naming a grouping variable for
#'   group-specific time-varying coefficients.
#' @param shared Whether the effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares it; `FALSE` fits
#'   process-specific effects and emits a warning.
#' @param sigma_prior_U,sigma_prior_alpha Penalized-complexity prior on each
#'   varying coefficient's marginal standard deviation, calibrated so that
#'   `P(sigma > sigma_prior_U) = sigma_prior_alpha`. Defaults to
#'   `P(sigma > 1) = 0.01`. `sigma_prior_U` must be positive and
#'   `sigma_prior_alpha` must lie in `(0, 1)`. Read on every structure: it is
#'   the same anchor pair whether the field samples a log-precision (`rw1` /
#'   `rw2` / `ar1`) or a log-variance (`gp`).
#' @param scale_coords Logical, `structure = "gp"` only: standardize the time
#'   values before fitting (default `TRUE`), which puts the lengthscale on the
#'   same universal support [temporal_gp()] uses. `period` is stated in the raw
#'   time units and makes the same trip.
#'
#' @return A `tulpa_tvc` object.
#'
#' @seealso [temporal_rw1()], [temporal_rw2()], [temporal_ar1()] for the
#'   underlying temporal priors; [temporal_gp()] for a GP over time that is not
#'   a varying coefficient.
#'
#' @examples
#' # Intercept that drifts as a first-order random walk over year
#' temporal_tvc("year", structure = "rw1")
#'
#' # A slope evolving as a continuous-time GP over irregularly-spaced visits
#' temporal_tvc("day", terms = ~ x - 1, structure = "gp", cov = "matern")
#'
#' @export
temporal_tvc <- function(time_var,
                         terms = 1,
                         structure = c("rw1", "rw2", "ar1", "gp"),
                         cov = c("exponential", "matern", "gaussian", "periodic"),
                         nu = 1.5,
                         period = NULL,
                         group_var = NULL,
                         shared = NULL,
                         sigma_prior_U = 1,
                         sigma_prior_alpha = 0.01,
                         scale_coords = TRUE) {

  structure_type <- match.arg(structure)
  .check_pc_anchors(sigma_prior_U, sigma_prior_alpha,
                    "sigma_prior_U", "sigma_prior_alpha", "temporal_tvc()")
  cov_type <- match.arg(cov)
  # The kernel a continuous-time temporal GP is built from, validated once for
  # both doors that offer one: the Matern smoothnesses with a closed form and
  # the periodic kernel's period (gcol33/tulpa#288 was those choices being
  # accepted and then silently run as exponential).
  if (identical(structure_type, "gp")) {
    .check_temporal_gp_kernel(cov_type, nu, period, "temporal_tvc()")
  }

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

  if (isFALSE(shared)) .warn_nonshared("TVCs", "time-varying effects")

  structure(
    list(
      type = "tvc",
      time_var = time_var,
      group_var = group_var,
      terms_spec = terms_spec,
      structure = structure_type,
      shared = shared,
      sigma_prior_U = as.numeric(sigma_prior_U),
      sigma_prior_alpha = as.numeric(sigma_prior_alpha),
      # GP kernel; NULL on every other structure, which reads the time index as
      # a grid position and has no kernel.
      cov = if (identical(structure_type, "gp")) cov_type else NULL,
      nu = if (identical(structure_type, "gp") && cov_type == "matern") nu else NULL,
      period = if (identical(structure_type, "gp") && cov_type == "periodic")
                 period else NULL,
      scale_coords = isTRUE(scale_coords),
      # Filled in during validation
      time_values = NULL,
      time_scale = NULL,
      period_scaled = NULL,
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
    gp = "GP (Gaussian process over the distinct times)"
  )
  cat("Structure:", struct_name, "\n")
  if (identical(x$structure, "gp")) {
    cov_str <- x$cov
    if (identical(x$cov, "matern")) cov_str <- paste0("matern (nu = ", x$nu, ")")
    if (identical(x$cov, "periodic")) {
      cov_str <- paste0("periodic (period = ", x$period, ")")
    }
    cat("Covariance:", cov_str, "\n")
  }
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
  .check_time_complete(time_vals, tvc$time_var)
  unique_times <- NULL
  if (is.factor(time_vals)) {
    time_factor <- time_vals
  } else {
    unique_times <- sort(unique(time_vals))
    time_factor <- factor(time_vals, levels = unique_times)
  }

  tvc$n_times <- nlevels(time_factor)
  tvc$time_index <- as.integer(time_factor)
  tvc$time_levels <- levels(time_factor)

  # A GP-evolving coefficient is a continuous-time field over the distinct
  # instants, so it needs WHERE they sit and not only their order --
  # `rw1` / `rw2` / `ar1` read the index as a position on a grid and carry
  # nothing here. A factor time variable states an ordering and no spacing, so
  # it has no lag for a kernel to measure (gcol33/tulpa#847).
  if (identical(tvc$structure, "gp")) {
    if (is.null(unique_times)) {
      stop("`temporal_tvc(structure = \"gp\")` needs a numeric time variable: ",
           "the coefficient evolves as a continuous-time GP, and a factor '",
           tvc$time_var, "' states an ordering with no spacing for the kernel ",
           "to measure. Use a numeric time, or \"rw1\" / \"rw2\" / \"ar1\", ",
           "which read the index as a position on a grid.", call. = FALSE)
    }
    tvals <- as.numeric(unique_times)
    # Scaled the way temporal_gp() scales its own: the lengthscale then lives on
    # one support whatever units the time variable is measured in. `period` is a
    # LAG stated in the raw units and makes the same trip; centring cancels in a
    # lag, the divisor does not (gcol33/tulpa#687).
    tvc$time_scale <- 1
    if (isTRUE(tvc$scale_coords) && length(tvals) > 1L) {
      obs_num <- as.numeric(time_vals)   # Date / POSIXt reach numeric here too
      s <- stats::sd(obs_num)
      if (is.finite(s) && s > 0) {
        tvc$time_scale <- s
        tvals <- (tvals - mean(obs_num)) / s
      }
    }
    tvc$time_values <- tvals
    tvc$period_scaled <- if (is.null(tvc$period)) NULL else
      as.numeric(tvc$period) / tvc$time_scale
  }

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
    tvc_indices <- .resolve_varying_coef_columns(
      tvc$terms_spec$names, has_intercept = FALSE, data, coef_names, "TVC")
    tvc_names <- coef_names[tvc_indices]

  } else if (tvc$terms_spec$type == "formula") {
    tt <- terms(tvc$terms_spec$formula)
    tvc_indices <- .resolve_varying_coef_columns(
      attr(tt, "term.labels"), has_intercept = attr(tt, "intercept") == 1,
      data, coef_names, "TVC")
    tvc_names <- coef_names[tvc_indices]
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
#' \donttest{
#' set.seed(160)
#' n_t <- 10L; reps <- 5L
#' walk <- cumsum(rnorm(n_t, 0, 0.35)); walk <- walk - mean(walk)
#' year <- rep(seq_len(n_t), each = reps)
#' df <- data.frame(year = year, x = rnorm(length(year)))
#' df$count <- rpois(nrow(df), exp(0.3 + (0.5 + walk[year]) * df$x))
#'
#' # The slope on `x` walks in time; TVC is exact-mode only.
#' fit <- tulpa(
#'   count ~ x,
#'   data = df,
#'   family = "poisson",
#'   temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
#'   mode = "exact",
#'   control = list(n_iter = 200L, n_warmup = 100L, seed = 1L)
#' )
#'
#' tvc_post <- tvc(fit)
#' summary(tvc_post)
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
  # `tulpa_tvc_posterior` carries a [draws, time, term] field and no group
  # axis, while a grouped fit's flat layout is (group, term, time). Reshaping
  # it into the ungrouped shape would report the first group as though it were
  # the whole field, so a grouped fit is refused here rather than collapsed.
  info <- .varying_coef_info(object, c("tvc", "temporal"), "tulpa_tvc")
  if (!is.null(info) && is.null(object$.internal$tvc_draws) &&
      isTRUE((info$n_groups %||% 1L) > 1L)) {
    stop("tvc() cannot summarize a grouped TVC field (group_var = \"",
         info$group_var, "\", ", info$n_groups, " groups): the posterior it ",
         "returns has no group dimension. Read the `tvc_w[...]` draws ",
         "directly, laid out group-major over (term, time).", call. = FALSE)
  }
  .extract_varying_coef(
    object, terms, summary, probs,
    slot = "tvc",
    info_class = "tulpa_tvc",
    not_fitted_msg = paste0(
      "Model was not fitted with temporally-varying coefficients.\n",
      "Pass `temporal = temporal_tvc(...)` to tulpa() with `mode = \"exact\"`."),
    draws_field = "tvc_draws",
    names_field = "tvc_names",
    field_slot = "temporal",
    draws_prefix = "tvc_w",
    n_units_field = "n_times",
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
