# Inline temporal varying-coefficient fields: the temporal mirror of the
# spatial() bar API (gcol33/tulpa#91). temporal(formula = ~ 1 + x || time,
# structure = "rw1") declares a smooth temporal level (the intercept column) and
# a temporally varying slope on each covariate column (eta_i += x_i * f(time_i)),
# routed through the same single-arm joint nested-Laplace machinery as spatial().
# The constructor is reached through the overloaded temporal() generic (see
# R/temporal_rtr_posteriors.R), which dispatches a formula first argument here.

# Build a tulpa_temporal_field from the bar formula. Mirrors spatial(): the bar
# RHS names the time index (the temporal analogue of the graph node); the LHS
# expands to one temporal field per design column. `structure` selects the
# temporal GMRF (rw1 / rw2 / ar1), the analogue of spatial()'s graph / proper.
.temporal_field_spec <- function(formula, structure = c("rw1", "rw2", "ar1"),
                                 shared = NULL, by = NULL) {

  if (!is.null(by)) {
    stop("`by =` (replicated temporal field -- one independent field per ",
         "factor level) is not implemented yet. Fit one temporal() field per ",
         "level by hand, or wait for the replicated feature.", call. = FALSE)
  }

  structure <- match.arg(structure)

  if (missing(formula) || !inherits(formula, "formula")) {
    stop("temporal() field form needs a one-sided formula with a grouping bar, ",
         "e.g. ~ 1 + x || time.", call. = FALSE)
  }
  if (length(formula) != 2L) {
    stop("temporal() `formula` must be one-sided (no left-hand side), e.g. ",
         "~ 1 + x || time.", call. = FALSE)
  }

  rhs <- formula[[2]]
  is_bar <- is.call(rhs) &&
    (identical(rhs[[1]], as.name("|")) || identical(rhs[[1]], as.name("||")))
  if (!is_bar) {
    stop("temporal() `formula` must carry a grouping bar naming the time ",
         "index, e.g. ~ 1 + x || time. Use the double bar || for independent ",
         "fields.", call. = FALSE)
  }

  specs <- parse_bar_term(rhs)
  if (length(specs) != 1L) {
    stop("Nested grouping is not supported in temporal(). The grouping ",
         "variable must be a single time index. Use ",
         "temporal(~ ... || time) and add ordinary random effects such as ",
         "(1 | site) separately.", call. = FALSE)
  }
  spec1 <- specs[[1L]]

  if (!is.null(spec1$group_expr) || length(spec1$group_vars) != 1L) {
    stop("temporal() grouping must be a single bare time-index column ",
         "(e.g. || time), not an interaction or expression.", call. = FALSE)
  }

  if (isTRUE(spec1$correlated)) {
    stop("Correlated temporal varying coefficients (fields sharing a ",
         "covariance) are not implemented yet; use the double bar || for ",
         "independent fields, e.g. ~ 1 + x || time. A single | is the temporal ",
         "counterpart of the spatial MCAR model.", call. = FALSE)
  }

  if (!isTRUE(spec1$has_intercept) && length(spec1$slope_terms) == 0L) {
    stop("temporal() `formula` has no terms; include an intercept (1) and/or ",
         "slopes, e.g. ~ 1 + x || time.", call. = FALSE)
  }

  structure(
    list(
      type          = "temporal_field",
      structure     = structure,
      group_var     = spec1$group_vars[[1L]],
      lhs           = rhs[[2L]],
      has_intercept = isTRUE(spec1$has_intercept),
      slope_terms   = spec1$slope_terms,
      correlated    = FALSE,
      shared        = shared,
      formula       = formula,
      env           = environment(formula)
    ),
    class = c("tulpa_temporal_field", "list")
  )
}


# Expand a temporal-field spec + data into a list of temporal GMRF blocks in the
# format tulpa_nested_laplace_joint() consumes: one block per design column,
# carrying the per-arm temporal_idx and (for non-intercept columns) the per-arm
# svc_weight (the temporally varying slope). The time-index column is mapped to
# 1..n_times by its sorted unique levels, matching the TVC convention.
.temporal_field_blocks <- function(spec, data) {
  tvals <- data[[spec$group_var]]
  if (is.null(tvals)) {
    stop("temporal() time-index column '", spec$group_var,
         "' was not found in the data.", call. = FALSE)
  }
  levels_t <- sort(unique(tvals))
  tfac <- factor(tvals, levels = levels_t)
  idx <- as.integer(tfac)
  n_times <- nlevels(tfac)
  cols <- .bar_field_columns(spec, data, fname = "temporal")
  struct <- spec$structure

  lapply(cols, function(col) {
    blk <- list(
      type         = struct,
      name         = paste(spec$group_var, col$name, sep = "."),
      n_times      = as.integer(n_times),
      temporal_idx = list(as.integer(idx))
    )
    if (struct == "rw1") blk$cyclic <- FALSE
    if (!col$is_intercept) {
      blk$svc_weight <- list(as.numeric(col$weight))
    }
    blk
  })
}


# Front-door fit for inline temporal() varying-coefficient fields. Mirrors
# .tulpa_fit_spatial_field: the bar expands to independent temporal GMRF blocks
# (rw1 / rw2 / ar1; slope columns carry a per-row svc_weight), fit as a one-arm
# joint nested-Laplace model through the shared .bar_field_fit_core.
.tulpa_fit_temporal_field <- function(parsed, bundle, data, family, mode, phi,
                                      sigma_re, n_trials, control, formula,
                                      call = NULL) {
  mode_lc <- tolower(mode %||% "auto")
  if (!mode_lc %in% c("auto", "laplace", "nested_laplace")) {
    stop("Inline temporal() varying-coefficient fields are fit by nested ",
         "Laplace; `mode` must be 'auto' or 'laplace' (got '", mode, "').",
         call. = FALSE)
  }
  if (!is.null(parsed$temporal_var)) {
    stop("Cannot combine an inline temporal(formula = ) field with a ",
         "temporal(col) term. Use one or the other.", call. = FALSE)
  }

  specs <- parsed$temporal_field_blocks
  blocks <- unlist(lapply(specs, .temporal_field_blocks, data = data),
                   recursive = FALSE)
  block_names <- vapply(blocks, function(b) b$name, character(1))

  core <- .bar_field_fit_core(
    blocks = blocks, block_names = block_names, bundle = bundle,
    parsed = parsed, family = family, phi = phi, n_trials = n_trials,
    sigma_re = sigma_re, control = control, arm_spatial_idx = NULL)
  jfit <- core$jfit

  # The second outer axis of an AR1 field is the AR1 correlation rho; rw1 / rw2
  # have only the precision axis (rho stays NULL).
  temporal_field_hypers <- lapply(core$hypers, function(h)
    list(name = h$name, structure = h$structure,
         sigma = h$sigma, rho = h$rho))
  names(temporal_field_hypers) <- block_names

  layout <- .tulpa_param_layout(bundle)
  fit <- jfit
  fit$draws <- core$beta_draws
  fit$draws_kind <- "iid"
  fit$means <- colMeans(core$beta_draws)
  fit$param_names <- colnames(bundle$X)
  fit$temporal_fields <- core$fields
  fit$temporal_field_names <- block_names
  fit$temporal_field_hypers <- temporal_field_hypers
  fit$inference_mode <- "laplace"
  fit$inference_tier <- 2L
  fit$backend <- "temporal_field_nested_laplace"
  fit$selection_reason <-
    "inline temporal() varying-coefficient field(s); nested Laplace over the temporal precisions"
  fit$formula <- formula
  fit$family <- family
  fit$call <- call
  fit$n_fixed <- layout$n_fixed
  fit$fixed_names <- layout$fixed_names
  fit$re_layout <- layout$re_layout
  fit$N <- bundle$n_obs
  fit$model_matrix <- bundle$X
  class(fit) <- c("tulpa_temporal_field_fit", "tulpa_fit", oldClass(fit))
  fit
}


#' @export
print.tulpa_temporal_field_fit <- function(x, ...) {
  cat("tulpa fit: inline temporal() varying-coefficient field(s)\n")
  cat("  N =", x$N, " fixed effects =", length(x$means), "\n\n")
  cat("Fixed effects (posterior mean):\n")
  print(round(x$means, 4))
  hy <- x$temporal_field_hypers
  if (length(hy)) {
    cat("\nTemporal fields:\n")
    for (h in hy) {
      cat(sprintf("  %s [%s]\n", h$name, toupper(h$structure)))
      if (!is.null(h$sigma)) {
        cat(sprintf("    sigma  %.3f  (95%% CI %.3f, %.3f)\n",
                    h$sigma[2L], h$sigma[1L], h$sigma[3L]))
      }
      if (!is.null(h$rho)) {
        cat(sprintf("    rho    %.3f  (95%% CI %.3f, %.3f)\n",
                    h$rho[2L], h$rho[1L], h$rho[3L]))
      }
    }
  }
  invisible(x)
}


#' @export
print.tulpa_temporal_field <- function(x, ...) {
  cat("tulpa temporal varying-coefficient field\n")
  cat("========================================\n\n")
  cat("Structure:", toupper(x$structure), "\n")
  cat("Time index:", x$group_var, "\n")
  cat("Fields:", if (x$correlated) "correlated" else "independent",
      "(|| -> separate precision per coefficient)\n")
  labs <- .bar_field_term_labels(x)
  cat("Expands to", length(labs),
      "temporal field(s) (one per design-matrix column):\n")
  for (lab in labs) {
    cat("  ", x$group_var, ".", lab, "\n", sep = "")
  }
  invisible(x)
}
