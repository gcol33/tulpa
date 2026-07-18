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
#'   graph node. The double bar `||` builds independent fields (each its own
#'   precision); a single bar `|` builds correlated fields -- a separable
#'   multivariate CAR where the per-cell coefficient vector shares a
#'   cross-covariance `Sigma` (covariance `Sigma (x) Q^-1`), so the
#'   intercept-slope correlation `rho` is shared across the graph.
#' @param proper Logical; `FALSE` (default) builds intrinsic CAR (ICAR /
#'   Besag) fields with the sum-to-zero constraint (`rho` fixed at 1). `TRUE`
#'   builds proper CAR fields, each with its own precision `Q = D - rho_car W`
#'   and the spatial autocorrelation `rho_car` estimated from the data (one
#'   `(sigma, rho_car)` pair per field). `summary()` and `print()` report the
#'   per-field `rho_car`. Independent (`||`) only; correlated proper CAR (a
#'   single `|` with `proper = TRUE`) is a separate model.
#' @param shared Optional shared-effect handle, passed through to the field
#'   blocks (see the model docs). Default `NULL` (shared).
#' @param by Optional replicated-CAR factor: a bare column name (or a string)
#'   naming a factor in the model data. With `L` levels it builds one
#'   independent copy of the whole field per level -- the field over the
#'   block-diagonal Kronecker graph `I_L (x) Q` (`L` disjoint copies of the
#'   graph) -- with the hyperparameters shared across levels (one `Sigma` for
#'   `|`; one `(sigma[, rho_car])` for `||`). This is `INLA`'s `replicate =` /
#'   `mgcv`'s `s(cell, by = ...)` generalised to the varying-coefficient bar,
#'   and is orthogonal to the bar character: `|` / `||` sets the covariance
#'   among the coefficient columns within a field, while `by` sets how many
#'   independent replicates of the whole field exist. Default `NULL` (one
#'   field). Correlated proper CAR (`|` with `proper = TRUE`) stays out of scope
#'   with or without `by`.
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

  # Capture `by` as a column name without evaluating it: in a model formula the
  # spatial() call is evaluated in the formula environment (not the data), so a
  # bare `by = habitat` must be deparsed to a name and resolved against the data
  # at fit time, exactly as the bar's group_var is.
  by_var <- NULL
  if (!missing(by)) {
    by_expr <- substitute(by)
    if (!is.null(by_expr)) {
      by_var <- if (is.character(by_expr)) as.character(by_expr)
                else deparse(by_expr)
    }
  }

  if (missing(graph)) {
    stop("`graph` must be an adjacency matrix (the spatial graph).",
         call. = FALSE)
  }
  graph <- .validate_adjacency_arg(graph, "graph")
  if (isFALSE(shared)) .warn_nonshared("spatial effects")

  if (missing(formula) || !inherits(formula, "formula")) {
    stop("`formula` must be a one-sided formula with a grouping bar, e.g. ",
         "~ 1 + time || cell.", call. = FALSE)
  }
  if (length(formula) != 2L) {
    stop("`formula` must be one-sided (no left-hand side), e.g. ",
         "~ 1 + time || cell.", call. = FALSE)
  }

  rhs <- formula[[2]]
  if (!.is_bar_call(rhs)) {
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

  if (isTRUE(spec1$correlated) && isTRUE(proper)) {
    stop("Correlated proper CAR (a single | with proper = TRUE) is not in ",
         "scope. Use `|` with the default intrinsic CAR (proper = FALSE) for a ",
         "separable MCAR, or `||` with proper = TRUE for independent proper ",
         "fields.", call. = FALSE)
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
      correlated    = isTRUE(spec1$correlated),
      proper        = isTRUE(proper),
      shared        = shared,
      by_var        = by_var,
      n_spatial     = nrow(graph),
      formula       = formula,
      env           = environment(formula)
    ),
    class = c("tulpa_spatial_field", "list")
  )
}


# Conceptual field labels from the bar LHS, without needing the data. Factor
# covariates expand to several model.matrix columns at fit time; this lists the
# model terms (one entry per term), which is what print shows. Shared by the
# spatial and temporal inline-field constructors (the bar grammar is identical).
.bar_field_term_labels <- function(spec) {
  lhs_form <- stats::as.formula(call("~", spec$lhs), env = spec$env)
  tt <- stats::terms(lhs_form)
  labs <- attr(tt, "term.labels")
  c(if (attr(tt, "intercept") == 1L) "Intercept", labs)
}


# Expand the bar LHS against the data into one descriptor per design-matrix
# column: name, the per-observation weight (the column itself), and whether it
# is the intercept (all-ones) column. This is the single source of the
# "one field per design column, weighted by it" rule -- the intercept is just
# the ones column, so there is no separate weighted vs unweighted path. Shared
# by spatial() and temporal(); `fname` only flavours the error messages.
.bar_field_columns <- function(spec, data, fname = "spatial") {
  lhs_form <- stats::as.formula(call("~", spec$lhs), env = spec$env)
  N <- nrow(data)
  tt <- stats::terms(lhs_form)
  has_int <- attr(tt, "intercept") == 1L

  if (length(all.vars(lhs_form)) == 0L) {
    if (!has_int) {
      stop(fname, "() `formula` has no terms.", call. = FALSE)
    }
    return(list(list(name = "Intercept", weight = rep(1.0, N),
                     is_intercept = TRUE)))
  }

  mf <- stats::model.frame(lhs_form, data = data, na.action = stats::na.pass)
  mm <- stats::model.matrix(tt, mf)
  if (nrow(mm) != N) {
    stop(fname, "() design expansion produced ", nrow(mm), " row(s) but data ",
         "has ", N, "; check for NA in the ", fname, "() covariate(s).",
         call. = FALSE)
  }
  cols <- colnames(mm)
  lapply(seq_along(cols), function(j) {
    nm <- cols[[j]]
    is_int <- identical(nm, "(Intercept)")
    if (anyNA(mm[, j])) {
      stop(fname, "() covariate column '", nm, "' has NA value(s).",
           call. = FALSE)
    }
    list(name = if (is_int) "Intercept" else nm,
         weight = as.numeric(mm[, j]),
         is_intercept = is_int)
  })
}


#' Expand a varying-coefficient bar into per-column field specs
#'
#' @description
#' Expand the left-hand side of an \pkg{lme4}-style varying-coefficient bar
#' (`~ 1 + w || node`) against a data frame into one descriptor per design
#' matrix column. This is the same expansion [spatial()] and the inline
#' `temporal()` field constructor use internally to turn a bar into one CAR /
#' temporal field per design column, exposed so a downstream package can reuse
#' the one implementation rather than re-parsing the bar grammar.
#'
#' The bar's right-hand side names the node index (the graph node for a spatial
#' field, the time index for a temporal field); it is not expanded but is
#' returned as the `node` attribute. The left-hand side is expanded with
#' [stats::model.matrix()]: the intercept column (`1`) is the unweighted
#' (all-ones) field, and each covariate column is a varying coefficient whose
#' per-observation weight is that column's design value (`0 +` drops the
#' intercept).
#'
#' @param formula A one-sided formula carrying a single grouping bar, e.g.
#'   `~ 1 + w || cell`. Use [tulpa_is_spatial_bar()] to test first. The
#'   right-hand side must be a single bare column naming the node index;
#'   nesting (`a / b`), interaction (`a:b`), or expression grouping is rejected.
#' @param data A data frame whose columns the bar left-hand side and the node
#'   index refer to. The per-column weight vectors are evaluated against it, so
#'   they have `nrow(data)` entries.
#'
#' @return A list with one element per design-matrix column, each a list:
#'   \describe{
#'     \item{`column_name`}{character; `"Intercept"` for the intercept column,
#'       otherwise the [model.matrix()] column name (e.g. `"w"`).}
#'     \item{`weight`}{a numeric vector of length `nrow(data)` for a covariate
#'       column (the per-observation design value scaling that field), or `NULL`
#'       for the intercept column (which is the all-ones, unweighted field).}
#'     \item{`is_intercept`}{logical; `TRUE` for the intercept column.}
#'   }
#'   The list carries two attributes: `node` (character, the node-index column
#'   named by the bar right-hand side) and `correlated` (logical, `TRUE` for a
#'   single `|`, `FALSE` for a double `||`).
#'
#' @seealso [tulpa_is_spatial_bar()] for the recognizer, [spatial()] for the
#'   inline areal field constructor that consumes this expansion.
#'
#' @examples
#' d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))
#'
#' # Intercept plus a varying slope on w
#' specs <- tulpa_bar_field_specs(~ 1 + w || cell, d)
#' length(specs)                 # 2
#' specs[[1]]$column_name        # "Intercept"
#' is.null(specs[[1]]$weight)    # TRUE (unweighted field)
#' specs[[2]]$column_name        # "w"
#' identical(specs[[2]]$weight, d$w)  # TRUE
#' attr(specs, "node")           # "cell"
#'
#' @export
tulpa_bar_field_specs <- function(formula, data) {
  if (!inherits(formula, "formula") || length(formula) != 2L) {
    stop("`formula` must be a one-sided formula carrying a grouping bar, e.g. ",
         "~ 1 + w || cell.", call. = FALSE)
  }
  rhs <- formula[[2L]]
  if (!.is_bar_call(rhs)) {
    stop("`formula` must carry a grouping bar naming the node index, e.g. ",
         "~ 1 + w || cell.", call. = FALSE)
  }

  specs <- parse_bar_term(rhs)
  if (length(specs) != 1L) {
    stop("Nested grouping is not supported. The grouping variable must be a ",
         "single node-index column, e.g. ~ ... || cell.", call. = FALSE)
  }
  spec1 <- specs[[1L]]
  if (!is.null(spec1$group_expr) || length(spec1$group_vars) != 1L) {
    stop("The bar grouping must be a single bare node-index column (e.g. ",
         "|| cell), not an interaction or expression.", call. = FALSE)
  }

  spec <- list(lhs = rhs[[2L]], env = environment(formula))
  cols <- .bar_field_columns(spec, data, fname = "tulpa_bar_field_specs")

  out <- lapply(cols, function(col) {
    list(column_name  = col$name,
         weight       = if (col$is_intercept) NULL else col$weight,
         is_intercept = col$is_intercept)
  })
  attr(out, "node") <- spec1$group_vars[[1L]]
  attr(out, "correlated") <- isTRUE(spec1$correlated)
  out
}


# Replicate a graph + per-observation node index across the levels of a `by`
# factor: build the block-diagonal Kronecker adjacency I_L (x) Q (L disjoint
# copies of the graph) and offset each observation's node into its level's copy
# (`node + (level - 1) * n_nodes`). Because the result is one graph with one
# precision, the L replicate fields are independent (no edges across levels) and
# share the hyperparameters -- the outer integration grid stays one axis. The
# single source of the replication remap, shared by .spatial_field_blocks() and
# the exported tulpa_bar_field_replicate(). L == 1 returns the graph untouched,
# so a single-level `by` is byte-identical to the no-`by` field.
.bar_field_replicate <- function(adjacency, node_idx, by_values) {
  f <- factor(by_values)
  if (anyNA(f)) {
    stop("spatial() `by` column has missing value(s).", call. = FALSE)
  }
  L <- nlevels(f)
  n <- nrow(adjacency)
  level <- as.integer(f)
  adj_rep <- if (L == 1L) {
    adjacency
  } else {
    as.matrix(Matrix::bdiag(
      rep(list(Matrix::Matrix(adjacency, sparse = TRUE)), L)))
  }
  list(adjacency = adj_rep,
       index     = node_idx + (level - 1L) * n,
       n_levels  = L,
       n_nodes   = n,
       levels    = levels(f))
}


#' Replicate an areal graph across the levels of a factor (replicated CAR)
#'
#' @description
#' Build the block-diagonal Kronecker graph `I_L (x) Q` and the level-offset
#' node index for a replicated areal field: one independent copy of the graph
#' per level of a `by` factor, all sharing one precision. This is the
#' graph-side counterpart to [tulpa_bar_field_specs()] -- that helper expands
#' the coefficient columns and is graph-agnostic, while replication needs the
#' graph, so it is a sibling rather than a new argument. A downstream package
#' composes the two (column expansion x replication) from the one
#' implementation rather than re-deriving the Kronecker remap.
#'
#' @param adjacency Symmetric adjacency matrix of the base graph
#'   (`[n_node x n_node]`, dense or sparse).
#' @param node Integer vector of 1-based graph-node indices, one per
#'   observation (the resolved bar right-hand side).
#' @param by A vector of the same length as `node` giving each observation's
#'   replication level; coerced to a factor. With `L` distinct levels the field
#'   is replicated `L` times.
#'
#' @return A list:
#'   \describe{
#'     \item{`adjacency`}{the `[L*n_node x L*n_node]` block-diagonal Kronecker
#'       adjacency `I_L (x) Q` (the base graph for `L == 1`).}
#'     \item{`index`}{integer vector, one per observation: the node index
#'       offset into its level's copy (`node + (level - 1) * n_node`).}
#'     \item{`n_levels`}{the number of replication levels `L`.}
#'     \item{`n_nodes`}{the base graph node count `n_node`.}
#'     \item{`levels`}{the factor levels of `by`, in replicate order.}
#'   }
#'
#' @seealso [tulpa_bar_field_specs()] for the coefficient-column expansion,
#'   [spatial()] for the inline areal field constructor whose `by =` argument
#'   this powers.
#'
#' @examples
#' adj <- matrix(0, 4, 4)
#' for (i in 1:3) adj[i, i + 1] <- adj[i + 1, i] <- 1
#' node <- rep(1:4, times = 2)
#' lev  <- rep(c("a", "b"), each = 4)
#' rep_info <- tulpa_bar_field_replicate(adj, node, lev)
#' dim(rep_info$adjacency)   # 8 x 8 (I_2 (x) Q)
#' rep_info$index            # level b nodes offset by 4
#'
#' @export
tulpa_bar_field_replicate <- function(adjacency, node, by) {
  if ((!is.matrix(adjacency) && !inherits(adjacency, "Matrix"))) {
    stop("`adjacency` must be an adjacency matrix.", call. = FALSE)
  }
  adj <- as.matrix(adjacency)
  if (nrow(adj) != ncol(adj)) {
    stop("`adjacency` must be square.", call. = FALSE)
  }
  node <- as.integer(node)
  if (length(node) != length(by)) {
    stop("`node` and `by` must have the same length (one entry per ",
         "observation).", call. = FALSE)
  }
  if (anyNA(node) || min(node) < 1L || max(node) > nrow(adj)) {
    stop("`node` must be 1-based indices into the adjacency graph (1..",
         nrow(adj), ").", call. = FALSE)
  }
  .bar_field_replicate(adj, node, by)
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
  base_adj <- as.matrix(spec$adjacency)
  n_units <- nrow(base_adj)
  idx <- .resolve_unit_index(data[[spec$group_var]], spec$group_var, n_units)
  proper <- isTRUE(spec$proper)
  # rho_car bounds come from the base graph's eigenvalue interval of D^-1 W;
  # block-diagonal replication leaves the spectrum unchanged (L identical
  # blocks), so derive them once from the base graph rather than the larger
  # replicated adjacency.
  rho_bounds <- if (proper) compute_car_rho_bounds(base_adj) else NULL

  # Replicated CAR: build the field over I_L (x) Q, offsetting each observation's
  # node into its `by`-level copy. One precision is shared across levels (it is
  # one graph), so the outer grid stays one axis.
  adj <- base_adj
  n_comp <- 1L
  if (!is.null(spec$by_var)) {
    if (is.null(data[[spec$by_var]])) {
      stop("spatial() `by` column '", spec$by_var,
           "' not found in the data.", call. = FALSE)
    }
    rep_info <- .bar_field_replicate(base_adj, idx, data[[spec$by_var]])
    adj <- rep_info$adjacency
    idx <- rep_info$index
    n_units <- rep_info$n_levels * rep_info$n_nodes
    n_comp <- rep_info$n_levels
  }

  csr <- adjacency_to_csr_tulpa(adj)
  cols <- .bar_field_columns(spec, data, fname = "spatial")

  # Correlated (single | ): one coupled separable-MCAR block over the p design
  # columns sharing Sigma (x) Q^-1. The p design columns become the fields; each
  # carries its column as the per-row design weight (the intercept's is ones).
  if (isTRUE(spec$correlated)) {
    field_names <- vapply(cols, function(col)
      paste(spec$group_var, col$name, sep = "."), character(1))
    blk <- list(
      type            = "mcar",
      name            = spec$group_var,
      n_spatial_units = as.integer(n_units),
      n_fields        = length(cols),
      adj_row_ptr     = as.integer(csr$row_ptr),
      adj_col_idx     = as.integer(csr$col_idx),
      n_neighbors     = as.integer(csr$n_neighbors),
      spatial_idx     = list(as.integer(idx)),
      field_weight    = lapply(cols, function(col) list(as.numeric(col$weight))),
      field_names     = field_names
    )
    # Replicated MCAR: L equal-size components per field (the engine pins each
    # per component and uses (n - L) in the Sigma normalizer).
    if (n_comp > 1L) blk$n_components <- as.integer(n_comp)
    return(list(blk))
  }

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
    # Intrinsic ICAR replication is component-aware in the engine; proper CAR is
    # full rank, so it needs no per-component treatment (correct over the
    # block-diagonal graph as is).
    if (!proper && n_comp > 1L) blk$n_components <- as.integer(n_comp)
    if (!col$is_intercept) {
      blk$svc_weight <- list(as.numeric(col$weight))
    }
    blk
  })
}


# Shared fit engine for an inline bar field (spatial or temporal). Given the
# already-built block list (icar / car_proper / rw1 / rw2 / ar1, each optionally
# carrying a per-row svc_weight), fit the one-arm joint nested-Laplace model and
# return the pieces both wrappers assemble into a tulpa_fit: the joint fit, the
# per-field weighted-mode posterior means, the per-field hyperparameter
# summaries, and the fixed-effect draws. The hyperparameter summaries are
# generic over the outer-grid axes: sigma = 1/sqrt(tau) from the `b<b>.tau`
# axis, and (when present) the second axis `b<b>.rho` -- which the caller labels
# rho_car (CAR) or the AR1 rho (temporal). Each is a derived quantity evaluated
# per grid cell then weighted-quantiled, never a plug-in of the modal value.
.bar_field_fit_core <- function(blocks, block_names, bundle, parsed, family,
                                phi, n_trials, sigma_re, control,
                                arm_spatial_idx = NULL) {
  N <- bundle$n_obs

  # Single random intercept (1 | g) is conditioned on at sigma_re, as on the
  # other front-door nested paths; richer RE structure is out of scope here.
  re <- bundle$re_terms %||% list()
  re_idx <- rep(0L, N)
  n_re_groups <- 0L
  sigma_re_scalar <- 1.0
  if (length(re) > 0L) {
    if (length(re) > 1L || (re[[1]]$n_coefs %||% 1L) != 1L) {
      stop("Inline bar fields support at most one random-intercept term ",
           "(1 | g) alongside the field(s). Model richer random effects ",
           "separately.", call. = FALSE)
    }
    re_idx <- as.integer(re[[1]]$group_idx)
    n_re_groups <- as.integer(re[[1]]$n_groups)
    sigma_re_scalar <- if (is.null(sigma_re)) 1.0 else sigma_re[1]
  }

  arm <- list(
    y           = bundle$y,
    n_trials    = n_trials %||% rep(1L, N),
    X           = bundle$X,
    re_idx      = re_idx,
    n_re_groups = n_re_groups,
    sigma_re    = sigma_re_scalar,
    family      = family,
    phi         = phi
  )
  # Spatial fields pin the arm's spatial_idx (legacy-kernel compatibility);
  # temporal fields carry their index on the block and leave the arm placeholder.
  if (!is.null(arm_spatial_idx)) arm$spatial_idx <- arm_spatial_idx

  # `control` reaching here has passed tulpa()'s union whitelist, which is wider
  # than the joint driver's set; n_draws is a field-fit knob read below and not
  # forwarded, so capture it, then subset the rest to what the joint driver
  # accepts (matching every other backend route in .tulpa_fitter_args()).
  ctrl_in <- control %||% list()
  n_draws <- as.integer(ctrl_in$n_draws %||% 1000L)
  ctrl <- .control_subset(ctrl_in, .CONTROL_KEYS$nested_laplace_joint)
  ctrl$store_Q <- TRUE
  ctrl$progress <- ctrl$progress %||% FALSE
  # Two or more fields make the tensor outer grid blow up multiplicatively;
  # CCD is the engine's recommended path for k >= 2 outer axes (same posterior,
  # polynomial node count). A single field keeps the driver's default grid.
  # A coupled MCAR block is a single block but carries p(p+1)/2 >= 3 outer axes
  # (the log-Cholesky coordinates of Sigma), so it requests CCD too: the joint
  # mode-centred CCD then places the Sigma grid at the marginal-likelihood mode
  # (the fix for a sharp likelihood landing between fixed nodes) and scales to
  # general p, declining back to the fixed log-Cholesky tensor only when the
  # outer curvature is ill-conditioned (a weakly-identified cross-correlation).
  is_mcar_block <- length(blocks) == 1L && identical(blocks[[1L]]$type, "mcar")
  if (length(blocks) >= 2L || is_mcar_block) {
    ctrl$integration <- ctrl$integration %||% "ccd"
  }

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
  # Per-block latent offsets. `block_start` is aligned with `blocks` for every
  # block type; the layout's `field_starts` registers only areal types (icar /
  # car_proper / bym2 / mcar), so temporal blocks are absent from it.
  field_starts <- al$block_start
  w <- jfit$weights

  # Weighted-mode posterior mean of each field (sum_k w_k m_k restricted to the
  # field's latent block).
  fields <- lapply(seq_along(blocks), function(b) {
    n_u <- blocks[[b]]$n_spatial_units %||% blocks[[b]]$n_times
    cols <- field_starts[b] + seq_len(n_u)
    list(name = block_names[b],
         mean = as.numeric(crossprod(w, jfit$modes[, cols, drop = FALSE])))
  })
  names(fields) <- block_names

  tg <- jfit$theta_grid
  hypers <- lapply(seq_along(blocks), function(b) {
    prefix  <- paste0("b", b, ".")
    has_col <- function(nm) !is.null(tg) && nm %in% colnames(tg)
    sigma <- if (has_col(paste0(prefix, "tau")))
      .nl_wtd_quantile(1 / sqrt(tg[, paste0(prefix, "tau")]), w,
                       c(0.025, 0.5, 0.975)) else NULL
    rho <- if (has_col(paste0(prefix, "rho")))
      .nl_wtd_quantile(tg[, paste0(prefix, "rho")], w,
                       c(0.025, 0.5, 0.975)) else NULL
    list(name = block_names[b], structure = blocks[[b]]$type,
         sigma = sigma, rho = rho)
  })
  names(hypers) <- block_names

  # Fixed-effect posterior draws (for coef/summary). Mixture-sample the latent
  # then slice the per-arm beta block.
  beta_cols <- beta_start + seq_len(p)
  draws_full <- tulpa_posterior_draws(jfit, idx = NULL, n = n_draws)
  beta_draws <- draws_full[, beta_cols, drop = FALSE]
  colnames(beta_draws) <- colnames(bundle$X)

  list(jfit = jfit, fields = fields, hypers = hypers, beta_draws = beta_draws)
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
  blocks <- unlist(lapply(specs, .spatial_field_blocks, data = data),
                   recursive = FALSE)
  is_mcar <- length(blocks) == 1L && identical(blocks[[1L]]$type, "mcar")
  block_names <- if (is_mcar) blocks[[1L]]$field_names
                 else vapply(blocks, function(b) b$name, character(1))

  core <- .bar_field_fit_core(
    blocks = blocks,
    block_names = if (is_mcar) blocks[[1L]]$name else block_names,
    bundle = bundle, parsed = parsed, family = family, phi = phi,
    n_trials = n_trials, sigma_re = sigma_re, control = control,
    arm_spatial_idx = blocks[[1L]]$spatial_idx[[1L]])
  jfit <- core$jfit

  if (is_mcar) {
    mb <- blocks[[1L]]
    p_fields <- mb$n_fields
    n_units  <- mb$n_spatial_units
    w  <- jfit$weights
    tg <- jfit$theta_grid
    fs <- jfit$arm_layout$field_starts[1L]
    # p field posterior means (weighted mode), one per design column, sliced from
    # the coupled p * n_units latent block (field-major layout).
    spatial_fields <- lapply(seq_len(p_fields), function(a) {
      cols <- fs + (a - 1L) * n_units + seq_len(n_units)
      list(name = mb$field_names[a],
           mean = as.numeric(crossprod(w, jfit$modes[, cols, drop = FALSE])))
    })
    names(spatial_fields) <- mb$field_names
    # Sigma derived quantities, marginalized over the log-Cholesky outer grid:
    # reconstruct Sigma per grid cell, derive (sigma_a, rho_ab), weighted-
    # quantile each (the marginalize-derived-quantities rule). Reuses the
    # RE-covariance log-Cholesky -> Sigma and derived-matrix helpers.
    m <- p_fields * (p_fields + 1L) / 2L
    axis_nm <- character(m); tt <- 1L
    for (j in seq_len(p_fields)) for (i in j:p_fields) {
      axis_nm[tt] <- sprintf("b1.L%d%d", i, j); tt <- tt + 1L
    }
    Sig_list <- lapply(seq_len(nrow(tg)), function(k) {
      L <- .re_logchol_to_L(as.numeric(tg[k, axis_nm]), p_fields)
      L %*% t(L)
    })
    D <- .re_cov_derived_matrix(Sig_list, p_fields, full = TRUE)
    mcar_summary <- lapply(colnames(D), function(cn)
      list(name = cn, q = .nl_wtd_quantile(D[, cn], w, c(0.025, 0.5, 0.975))))
    names(mcar_summary) <- colnames(D)
    spatial_field_hypers <- NULL
  } else {
    spatial_fields <- core$fields
    # The second outer axis of a CAR field is the spatial autocorrelation rho_car.
    spatial_field_hypers <- lapply(core$hypers, function(h)
      list(name = h$name, structure = h$structure,
           sigma = h$sigma, rho_car = h$rho))
    names(spatial_field_hypers) <- block_names
    mcar_summary <- NULL
  }

  layout <- .tulpa_param_layout(bundle)
  fit <- jfit
  fit$draws <- core$beta_draws
  fit$draws_kind <- "iid"
  fit$means <- colMeans(core$beta_draws)
  fit$param_names <- colnames(bundle$X)
  fit$spatial_fields <- spatial_fields
  fit$spatial_field_names <- block_names
  fit$spatial_field_hypers <- spatial_field_hypers
  fit$mcar_summary <- mcar_summary
  fit$correlated <- is_mcar
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
  fit$N <- bundle$n_obs
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
  if (isTRUE(x$correlated) && length(x$mcar_summary)) {
    cat("\nCorrelated areal fields (separable MCAR):\n")
    cat("  ", paste(x$spatial_field_names, collapse = ", "), "\n", sep = "")
    cat("  Cross-covariance Sigma (marginalized over the grid):\n")
    for (h in x$mcar_summary) {
      cat(sprintf("    %-8s %.3f  (95%% CI %.3f, %.3f)\n",
                  h$name, h$q[2L], h$q[1L], h$q[3L]))
    }
  } else {
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
  if (!is.null(x$by_var)) {
    cat("Replicated by:", x$by_var,
        "(one independent copy of the field per level, shared hyperparameters)\n")
  }
  cat("Fields:", if (x$correlated) "correlated" else "independent",
      "(|| -> separate precision per coefficient)\n")
  labs <- .bar_field_term_labels(x)
  cat("Expands to", length(labs), "CAR field(s) (one per design-matrix column):\n")
  for (lab in labs) {
    cat("  ", x$group_var, ".", lab, "\n", sep = "")
  }
  invisible(x)
}
