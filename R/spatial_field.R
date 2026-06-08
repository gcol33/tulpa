#' Areal spatially varying coefficient field
#'
#' @description
#' Declare one or more areal (CAR / Besag) fields over a graph from an
#' \pkg{lme4}-style bar formula. The bar's left-hand side lists the
#' coefficients that vary smoothly over the graph; the right-hand side names
#' the graph-node index. Each coefficient becomes an independent CAR field,
#' entering the linear predictor scaled by that coefficient's per-observation
#' design value:
#'
#' \deqn{\eta_i = \ldots + \sum_c X_{ic}\, z^{(c)}_{g_i},}
#'
#' where \eqn{g_i} is the graph node of observation \eqn{i}, \eqn{z^{(c)}} is
#' the CAR field for design column \eqn{c}, and \eqn{X_{ic}} is that column's
#' value at observation \eqn{i}. The intercept column is all ones, so
#' `~ 1 || cell` is the ordinary spatial intercept field; a covariate column
#' (e.g. `time`) gives a spatially varying slope on that covariate (a
#' per-region trend).
#'
#' Use it inline in a [tulpa()] model formula, the same way a random-effect
#' bar is written:
#'
#' \preformatted{
#' y ~ time + spatial(graph = adj, formula = ~ 1 + time || cell) + (1 | site)
#' }
#'
#' @param graph Symmetric adjacency matrix of the spatial graph
#'   (`[n_node x n_node]`, dense or sparse). One CAR field is defined over its
#'   nodes per coefficient.
#' @param formula One-sided formula carrying a grouping bar, e.g.
#'   `~ 1 + time || cell`. The left-hand side is expanded with
#'   [stats::model.matrix()] into one CAR field per column (`1` = intercept
#'   field; a covariate = spatially varying slope on it; `0 +` drops the
#'   intercept). The right-hand side must be a single bare column naming the
#'   graph node. Only the double bar `||` (independent fields) is supported;
#'   the single bar `|` would request correlated fields (a multivariate CAR)
#'   and is not yet implemented.
#' @param proper Logical; `FALSE` (default) builds intrinsic CAR (ICAR /
#'   Besag) fields with the sum-to-zero constraint (`rho` fixed at 1). `TRUE`
#'   builds proper CAR fields, each with its own precision `Q = D - rho_car W`
#'   and the spatial autocorrelation `rho_car` estimated from the data (one
#'   `(sigma, rho_car)` pair per field). `summary()` and `print()` report the
#'   per-field `rho_car`. Independent (`||`) only; correlated proper CAR (a
#'   single `|` with `proper = TRUE`) is a separate model.
#' @param shared Optional shared-effect handle, passed through to the field
#'   blocks (see the model docs). Default `NULL` (shared).
#' @param by Reserved. A future replicated-CAR argument that would build an
#'   independent copy of the whole field for each level of a factor. Not yet
#'   implemented; passing it errors.
#'
#' @return A `tulpa_spatial_field` object describing the field(s). It is
#'   expanded into one CAR block per design column at fit time, when the data
#'   is available.
#'
#' @details
#' Each field is independent (its own precision), matching `INLA`'s two
#' separate `f(cell, model = "besag")` and `f(cell.slope, time,
#' model = "besag")` fields. Nesting (`a / b`), interaction, or expression
#' grouping is rejected: the grouping must be a single graph node. Add
#' ordinary nested random effects, e.g. `(1 | site)`, as separate terms.
#'
#' @seealso [spatial_car()] for a single areal field passed via the
#'   `spatial =` argument, [spatial_svc()] for the coordinate-based
#'   (Gaussian-process) spatially varying coefficient.
#'
#' @examples
#' # Chain graph over 10 cells
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) adj[i, i + 1] <- adj[i + 1, i] <- 1
#'
#' # Spatial intercept plus a spatially varying time slope
#' f <- spatial(graph = adj, formula = ~ 1 + time || cell)
#' print(f)
#'
#' @export
spatial <- function(graph, formula, proper = FALSE, shared = NULL,
                    by = NULL) {

  if (!is.null(by)) {
    stop("`by =` (replicated CAR -- one independent field per factor level) ",
         "is not implemented yet. Fit one spatial() field per level by hand, ",
         "or wait for the replicated-CAR feature.", call. = FALSE)
  }

  if (missing(graph) || (!is.matrix(graph) && !inherits(graph, "Matrix"))) {
    stop("`graph` must be an adjacency matrix (the spatial graph).",
         call. = FALSE)
  }
  if (nrow(graph) != ncol(graph)) {
    stop("`graph` must be square.", call. = FALSE)
  }
  if (!isSymmetric(unname(as.matrix(graph)))) {
    stop("`graph` must be symmetric.", call. = FALSE)
  }

  if (missing(formula) || !inherits(formula, "formula")) {
    stop("`formula` must be a one-sided formula with a grouping bar, e.g. ",
         "~ 1 + time || cell.", call. = FALSE)
  }
  if (length(formula) != 2L) {
    stop("`formula` must be one-sided (no left-hand side), e.g. ",
         "~ 1 + time || cell.", call. = FALSE)
  }

  rhs <- formula[[2]]
  is_bar <- is.call(rhs) &&
    (identical(rhs[[1]], as.name("|")) || identical(rhs[[1]], as.name("||")))
  if (!is_bar) {
    stop("spatial() `formula` must carry a grouping bar naming the graph ",
         "node, e.g. ~ 1 + time || cell. Use the double bar || for ",
         "independent fields.", call. = FALSE)
  }

  specs <- parse_bar_term(rhs)
  if (length(specs) != 1L) {
    stop("Nested grouping is not supported in spatial(). The grouping ",
         "variable must be a single graph-node index. Use ",
         "spatial(graph, ~ ... || cell) for the CAR field and add ordinary ",
         "random effects such as (1 | site) separately.", call. = FALSE)
  }
  spec1 <- specs[[1L]]

  if (!is.null(spec1$group_expr) || length(spec1$group_vars) != 1L) {
    stop("spatial() grouping must be a single bare graph-node column ",
         "(e.g. || cell), not an interaction or expression.", call. = FALSE)
  }

  if (isTRUE(spec1$correlated)) {
    stop("Correlated areal varying coefficients (a multivariate CAR) are not ",
         "implemented yet; use the double bar || for independent fields, e.g. ",
         "~ 1 + time || cell. A single | would couple the fields through a ",
         "shared covariance, which is a separate model.", call. = FALSE)
  }

  if (!isTRUE(spec1$has_intercept) && length(spec1$slope_terms) == 0L) {
    stop("spatial() `formula` has no terms; include an intercept (1) and/or ",
         "slopes, e.g. ~ 1 + time || cell.", call. = FALSE)
  }

  structure(
    list(
      type          = "car_field",
      adjacency     = graph,
      group_var     = spec1$group_vars[[1L]],
      lhs           = rhs[[2L]],
      has_intercept = isTRUE(spec1$has_intercept),
      slope_terms   = spec1$slope_terms,
      correlated    = FALSE,
      proper        = isTRUE(proper),
      shared        = shared,
      n_spatial     = nrow(graph),
      formula       = formula,
      env           = environment(formula)
    ),
    class = c("tulpa_spatial_field", "list")
  )
}


# Conceptual field labels from the bar LHS, without needing the data. Factor
# covariates expand to several model.matrix columns at fit time; this lists the
# model terms (one entry per term), which is what print shows.
.spatial_field_term_labels <- function(spec) {
  lhs_form <- stats::as.formula(call("~", spec$lhs), env = spec$env)
  tt <- stats::terms(lhs_form)
  labs <- attr(tt, "term.labels")
  c(if (attr(tt, "intercept") == 1L) "Intercept", labs)
}


# Expand the bar LHS against the data into one descriptor per design-matrix
# column: name, the per-observation weight (the column itself), and whether it
# is the intercept (all-ones) column. This is the single source of the
# "one field per design column, weighted by it" rule -- the intercept is just
# the ones column, so there is no separate weighted vs unweighted path.
.spatial_field_columns <- function(spec, data) {
  lhs_form <- stats::as.formula(call("~", spec$lhs), env = spec$env)
  N <- nrow(data)
  tt <- stats::terms(lhs_form)
  has_int <- attr(tt, "intercept") == 1L

  if (length(all.vars(lhs_form)) == 0L) {
    if (!has_int) {
      stop("spatial() `formula` has no terms.", call. = FALSE)
    }
    return(list(list(name = "Intercept", weight = rep(1.0, N),
                     is_intercept = TRUE)))
  }

  mf <- stats::model.frame(lhs_form, data = data, na.action = stats::na.pass)
  mm <- stats::model.matrix(tt, mf)
  if (nrow(mm) != N) {
    stop("spatial() design expansion produced ", nrow(mm), " row(s) but data ",
         "has ", N, "; check for NA in the spatial() covariate(s).",
         call. = FALSE)
  }
  cols <- colnames(mm)
  lapply(seq_along(cols), function(j) {
    nm <- cols[[j]]
    is_int <- identical(nm, "(Intercept)")
    if (anyNA(mm[, j])) {
      stop("spatial() covariate column '", nm, "' has NA value(s).",
           call. = FALSE)
    }
    list(name = if (is_int) "Intercept" else nm,
         weight = as.numeric(mm[, j]),
         is_intercept = is_int)
  })
}


# Expand a spatial-field spec + data into a list of areal CAR blocks in the
# format tulpa_nested_laplace_joint() consumes: one block per design column,
# each carrying the per-arm spatial_idx and (for non-intercept columns) the
# per-arm svc_weight. Built for a single response arm; the per-arm lists are
# length 1. `proper = TRUE` builds `car_proper` blocks carrying the
# eigenvalue-derived `rho_bounds` (so the registry derives a (tau, rho_car)
# outer grid and rho_car is estimated per field); `proper = FALSE` builds
# intrinsic `icar` blocks (rho fixed at 1).
.spatial_field_blocks <- function(spec, data) {
  adj <- as.matrix(spec$adjacency)
  csr <- adjacency_to_csr_tulpa(adj)
  n_units <- nrow(adj)
  idx <- .resolve_unit_index(data[[spec$group_var]], spec$group_var, n_units)
  cols <- .spatial_field_columns(spec, data)
  proper <- isTRUE(spec$proper)
  rho_bounds <- if (proper) compute_car_rho_bounds(adj) else NULL

  lapply(cols, function(col) {
    blk <- list(
      type            = if (proper) "car_proper" else "icar",
      name            = paste(spec$group_var, col$name, sep = "."),
      n_spatial_units = as.integer(n_units),
      adj_row_ptr     = as.integer(csr$row_ptr),
      adj_col_idx     = as.integer(csr$col_idx),
      n_neighbors     = as.integer(csr$n_neighbors),
      spatial_idx     = list(as.integer(idx))
    )
    if (proper) blk$rho_bounds <- as.numeric(rho_bounds)
    if (!col$is_intercept) {
      blk$svc_weight <- list(as.numeric(col$weight))
    }
    blk
  })
}


# Front-door fit for inline spatial() varying-coefficient fields. The bar
# expands to a list of independent CAR blocks (one per design column, the
# slope columns carrying a per-row svc_weight); a single response is fit as a
# one-arm joint nested-Laplace model, which is the engine path that threads
# svc_weight. Returns a tulpa_fit.
.tulpa_fit_spatial_field <- function(parsed, bundle, data, family, mode, phi,
                                     sigma_re, n_trials, control, formula,
                                     call = NULL) {
  mode_lc <- tolower(mode %||% "auto")
  if (!mode_lc %in% c("auto", "laplace", "nested_laplace")) {
    stop("Inline spatial() varying-coefficient fields are fit by nested ",
         "Laplace; `mode` must be 'auto' or 'laplace' (got '", mode, "').",
         call. = FALSE)
  }
  if (!is.null(parsed$spatial_var)) {
    stop("Cannot combine an inline spatial(graph = , formula = ) field with a ",
         "spatial(col) term. Use one or the other.", call. = FALSE)
  }

  specs <- parsed$spatial_field_blocks
  N <- bundle$n_obs

  # Single random intercept (1 | g) is conditioned on at sigma_re, as on the
  # other front-door nested paths; richer RE structure is out of scope here.
  re <- bundle$re_terms %||% list()
  re_idx <- rep(0L, N)
  n_re_groups <- 0L
  sigma_re_scalar <- 1.0
  if (length(re) > 0L) {
    if (length(re) > 1L || (re[[1]]$n_coefs %||% 1L) != 1L) {
      stop("Inline spatial() fields support at most one random-intercept term ",
           "(1 | g) alongside the spatial field(s). Model richer random ",
           "effects separately.", call. = FALSE)
    }
    re_idx <- as.integer(re[[1]]$group_idx)
    n_re_groups <- as.integer(re[[1]]$n_groups)
    sigma_re_scalar <- if (is.null(sigma_re)) 1.0 else sigma_re[1]
  }

  blocks <- unlist(lapply(specs, .spatial_field_blocks, data = data),
                   recursive = FALSE)
  block_names <- vapply(blocks, function(b) b$name, character(1))

  first_idx <- blocks[[1L]]$spatial_idx[[1L]]
  arm <- list(
    y           = bundle$y,
    n_trials    = n_trials %||% rep(1L, N),
    X           = bundle$X,
    re_idx      = re_idx,
    n_re_groups = n_re_groups,
    sigma_re    = sigma_re_scalar,
    spatial_idx = first_idx,
    family      = family,
    phi         = phi
  )

  ctrl <- control %||% list()
  ctrl$store_Q <- TRUE
  ctrl$progress <- ctrl$progress %||% FALSE
  # Two or more fields make the tensor outer grid blow up multiplicatively;
  # CCD is the engine's recommended path for k >= 2 outer axes (same posterior,
  # polynomial node count). A single field keeps the driver's default grid.
  if (length(blocks) >= 2L) {
    ctrl$integration <- ctrl$integration %||% "ccd"
  }
  n_draws <- as.integer(ctrl$n_draws %||% 1000L)
  ctrl$n_draws <- NULL

  # The tensor outer grid's cell count is an intended property of the field
  # count (shown by print.tulpa_spatial_field); muffle only that specific
  # grid-size notice -- CCD cannot engage below three axes, so its advice does
  # not apply to the common intercept-plus-one-slope case. Other warnings pass.
  jfit <- withCallingHandlers(
    tulpa_nested_laplace_joint(
      responses = stats::setNames(list(arm), parsed$response %||% "y"),
      prior     = blocks,
      copy      = NULL,
      control   = ctrl
    ),
    warning = function(w) {
      if (grepl("multi-block grid has", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )

  al <- jfit$arm_layout
  p <- ncol(bundle$X)
  beta_start <- if (!is.null(al$beta_start)) al$beta_start[1L] else 0L
  field_starts <- al$field_starts

  # Weighted-mode posterior mean of each CAR field (sum_k w_k m_k restricted to
  # the field's latent block) -- the deterministic field summary.
  w <- jfit$weights
  spatial_fields <- lapply(seq_along(blocks), function(b) {
    n_u <- blocks[[b]]$n_spatial_units
    cols <- field_starts[b] + seq_len(n_u)
    list(name = block_names[b],
         mean = as.numeric(crossprod(w, jfit$modes[, cols, drop = FALSE])))
  })
  names(spatial_fields) <- block_names

  # Per-field hyperparameter posteriors, marginalized over the outer grid: the
  # field amplitude sigma = 1/sqrt(tau) and (for proper CAR) the spatial
  # autocorrelation rho_car. Each is a derived quantity evaluated per grid cell
  # then weighted-quantiled (never a plug-in of the modal hyperparameter). The
  # outer-grid axes are named `b<block>.tau` / `b<block>.rho`.
  tg <- jfit$theta_grid
  spatial_field_hypers <- lapply(seq_along(blocks), function(b) {
    prefix  <- paste0("b", b, ".")
    tau_col <- paste0(prefix, "tau")
    rho_col <- paste0(prefix, "rho")
    has_col <- function(nm) !is.null(tg) && nm %in% colnames(tg)
    sigma <- if (has_col(tau_col))
      .nl_wtd_quantile(1 / sqrt(tg[, tau_col]), w, c(0.025, 0.5, 0.975)) else NULL
    rho_car <- if (has_col(rho_col))
      .nl_wtd_quantile(tg[, rho_col], w, c(0.025, 0.5, 0.975)) else NULL
    list(name = block_names[b], structure = blocks[[b]]$type,
         sigma = sigma, rho_car = rho_car)
  })
  names(spatial_field_hypers) <- block_names

  # Fixed-effect posterior draws (for coef/summary). Mixture-sample the latent
  # then slice the per-arm beta block.
  beta_cols <- beta_start + seq_len(p)
  draws_full <- tulpa_posterior_draws(jfit, idx = NULL, n = n_draws)
  beta_draws <- draws_full[, beta_cols, drop = FALSE]
  colnames(beta_draws) <- colnames(bundle$X)

  layout <- .tulpa_param_layout(bundle)
  fit <- jfit
  fit$draws <- beta_draws
  fit$draws_kind <- "iid"
  fit$means <- colMeans(beta_draws)
  fit$param_names <- colnames(bundle$X)
  fit$spatial_fields <- spatial_fields
  fit$spatial_field_names <- block_names
  fit$spatial_field_hypers <- spatial_field_hypers
  fit$inference_mode <- "laplace"
  fit$inference_tier <- 2L
  fit$backend <- "spatial_field_nested_laplace"
  fit$selection_reason <-
    "inline spatial() varying-coefficient field(s); nested Laplace over the CAR precisions"
  fit$formula <- formula
  fit$family <- family
  fit$call <- call
  fit$n_fixed <- layout$n_fixed
  fit$fixed_names <- layout$fixed_names
  fit$re_layout <- layout$re_layout
  fit$N <- N
  fit$model_matrix <- bundle$X
  class(fit) <- c("tulpa_spatial_field_fit", "tulpa_fit", oldClass(fit))
  fit
}


#' @export
print.tulpa_spatial_field_fit <- function(x, ...) {
  cat("tulpa fit: inline spatial() varying-coefficient field(s)\n")
  cat("  N =", x$N, " fixed effects =", length(x$means), "\n\n")
  cat("Fixed effects (posterior mean):\n")
  print(round(x$means, 4))
  hy <- x$spatial_field_hypers
  if (length(hy)) {
    cat("\nSpatial fields:\n")
    for (h in hy) {
      struct <- if (identical(h$structure, "car_proper")) "proper CAR" else "ICAR"
      cat(sprintf("  %s [%s]\n", h$name, struct))
      if (!is.null(h$sigma)) {
        cat(sprintf("    sigma    %.3f  (95%% CI %.3f, %.3f)\n",
                    h$sigma[2L], h$sigma[1L], h$sigma[3L]))
      }
      if (!is.null(h$rho_car)) {
        cat(sprintf("    rho_car  %.3f  (95%% CI %.3f, %.3f)\n",
                    h$rho_car[2L], h$rho_car[1L], h$rho_car[3L]))
      }
    }
  }
  invisible(x)
}


#' @export
print.tulpa_spatial_field <- function(x, ...) {
  cat("tulpa areal varying-coefficient field\n")
  cat("=====================================\n\n")
  cat("Structure:", if (x$proper) "proper CAR" else "ICAR (Besag)", "\n")
  cat("Graph nodes:", x$n_spatial, "\n")
  cat("Graph node index:", x$group_var, "\n")
  cat("Fields:", if (x$correlated) "correlated" else "independent",
      "(|| -> separate precision per coefficient)\n")
  labs <- .spatial_field_term_labels(x)
  cat("Expands to", length(labs), "CAR field(s) (one per design-matrix column):\n")
  for (lab in labs) {
    cat("  ", x$group_var, ".", lab, "\n", sep = "")
  }
  invisible(x)
}
