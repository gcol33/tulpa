#' CAR / ICAR spatial structure
#'
#' Constructs a conditional autoregressive spatial random effect from an
#' adjacency matrix. With `proper = FALSE` (the default) this is the
#' improper CAR / ICAR (`type = "car"`); with `proper = TRUE` it returns
#' the same object as [spatial_car_proper()].
#'
#' @param adjacency Symmetric adjacency matrix (`[n_units x n_units]`).
#' @param level Either `"group"` (one effect per level of `group_var`) or
#'   `"obs"` (one effect per row of the data; `nrow(data)` must equal
#'   `nrow(adjacency)`).
#' @param group_var Name of the grouping variable in the data; required
#'   when `level = "group"`.
#' @param proper If `TRUE`, use proper CAR (`type = "car_proper"`); else
#'   ICAR (`type = "car"`).
#' @param shared Optional shared-effect handle (see model docs).
#' @param parameterization `"standard"` (default) or `"collapsed"` (deprecated).
#'
#' @return A `tulpa_spatial` object with `type = "car"` (or
#'   `"car_proper"` when `proper = TRUE`).
#'
#' @seealso [spatial_car_proper()], [spatial_bym2()].
#' @examples
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) adj[i, i + 1] <- adj[i + 1, i] <- 1
#' spatial_car(adj, level = "group", group_var = "region")
#' @export
spatial_car <- function(adjacency, level = c("group", "obs"),
                        group_var = NULL, proper = FALSE, shared = NULL,
                        parameterization = c("standard", "collapsed")) {

  level <- match.arg(level)
  parameterization <- match.arg(parameterization)

  adjacency <- .validate_adjacency_arg(adjacency, "adjacency")

  # Check for group_var if level = "group"
  if (level == "group" && is.null(group_var)) {
    stop("`group_var` is required when level = 'group'", call. = FALSE)
  }

  # Collapsed parameterization is deprecated (archived 2026-03-10)
  if (parameterization == "collapsed") {
    warning(
      "Collapsed parameterization is deprecated and will be removed in a future version.\n",
      "Collapsed + HMC creates poor posterior geometry (88x slower than standard).\n",
      "Use standard parameterization instead. A Gibbs backend for large spatial ",
      "models is planned.",
      call. = FALSE
    )
  }

  # Collapsed not supported with proper CAR
  if (parameterization == "collapsed" && proper) {
    stop("Collapsed parameterization is not supported with proper CAR", call. = FALSE)
  }

  # Compute eigenvalue bounds for proper CAR (needed for valid rho range)
  rho_bounds <- NULL
  if (proper) {
    rho_bounds <- compute_car_rho_bounds(adjacency)
  }

  if (isFALSE(shared)) .warn_nonshared("spatial effects")

  structure(
    list(
      type = if (proper) "car_proper" else "car",
      adjacency = adjacency,
      level = level,
      group_var = group_var,
      proper = proper,
      rho_bounds = rho_bounds,
      shared = shared,
      parameterization = parameterization,
      n_spatial = nrow(adjacency)
    ),
    class = c("tulpa_spatial", "list")
  )
}


# The nested-Laplace path (.spatial_spec_to_nl_prior(), .NL_FRONTDOOR_AREAL)
# already treats type = "car" as the intrinsic ICAR field -- spatial_car()'s
# own doc calls the untyped default "the improper CAR / ICAR". The Polya-Gamma
# Gibbs dispatch (dispatch_gibbs_spatial()) and auto's Gibbs-eligibility check
# (auto_select_mode()) matched only the literal "icar", so a spatial_car()
# spec was refused by mode = "gibbs" and routed to nested Laplace under auto
# where the identical bare list `type = "icar"` reached Gibbs (gcol33/tulpa#819).
# One alias, read by both.
#' @keywords internal
.areal_gibbs_type <- function(spatial_type) {
  if (identical(spatial_type, "car")) "icar" else spatial_type
}


#' Proper CAR spatial structure
#'
#' @description
#' Convenience wrapper for `spatial_car(..., proper = TRUE)`. Creates a
#' proper conditional autoregressive (CAR) spatial random effect with the
#' autocorrelation parameter rho estimated from the data.
#'
#' Use this when you want spatial autocorrelation to be a parameter of the
#' model rather than fixed at 1 (as in ICAR). rho ~= 0 collapses to IID,
#' rho ~= 1 approaches ICAR.
#'
#' @inheritParams spatial_car
#'
#' @return A `tulpa_spatial` object with `type = "car_proper"`.
#'
#' @seealso [spatial_car()] for ICAR (rho fixed at 1),
#'   [spatial_bym2()] for the BYM2 decomposition.
#'
#' @examples
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) adj[i, i+1] <- adj[i+1, i] <- 1
#' spec <- spatial_car_proper(adj, level = "group", group_var = "site")
#' print(spec)
#'
#' @export
spatial_car_proper <- function(adjacency,
                               level = c("group", "obs"),
                               group_var = NULL,
                               shared = NULL) {
  spatial_car(
    adjacency = adjacency,
    level = match.arg(level),
    group_var = group_var,
    proper = TRUE,
    shared = shared,
    parameterization = "standard"
  )
}


# Resolve a user-supplied group_var vector to 1-based indices into the rows
# of an adjacency matrix. Empty cells (cells with no data) are allowed:
# the ICAR / CAR / BYM2 Q-matrix is well-defined on every node of the graph
# regardless of whether the data touches that node. This matches INLA's
# `f(cell, model = "besag", graph = g)` semantics.
#
# Resolution rules:
#   * integer / numeric values  -> interpreted as 1-based row indices into
#                                   the adjacency. Values are validated
#                                   to lie in [1, n_spatial_units].
#   * character / factor values + rownames(adjacency) set
#                              -> match by name.
#   * character / factor values, no rownames, every label a whole number
#                              -> read as the numeric node index the label
#                                  spells ("10" is node 10), never as a sort
#                                  position ("10" sorting before "2").
#   * factor values, no rownames, other labels
#                              -> the level order is the node order, and the
#                                  number of levels must equal n_spatial_units
#                                  (otherwise the levels reindex the cells and
#                                  lose the user's cell identity).
#   * character values, no rownames, other labels
#                              -> refused. A character vector states no order,
#                                  so the only order available is the sort
#                                  order of the labels ("r1", "r10", "r11",
#                                  "r2", ...), which is not the graph's; it
#                                  scrambled the field silently
#                                  (gcol33/tulpa#900).
#
# This is the one resolver for an areal unit column at every door -- tulpa()'s
# sampler / Laplace / nested routes and the inline spatial() field alike.
.resolve_spatial_idx <- function(values, n_spatial_units, adjacency,
                                 group_var = "group") {
  if (length(values) == 0L) return(integer(0))
  if (anyNA(values)) {
    stop("Spatial group variable '", group_var,
         "' contains NA values; cannot resolve to adjacency indices.",
         call. = FALSE)
  }

  rn <- rownames(adjacency)

  if (is.numeric(values) || is.integer(values)) {
    idx <- as.integer(values)
    if (anyNA(idx) || any(idx != values)) {
      stop("Spatial group variable '", group_var,
           "' must contain whole-number indices when numeric.",
           call. = FALSE)
    }
    bad <- idx[idx < 1L | idx > n_spatial_units]
    if (length(bad) > 0L) {
      stop("Spatial group variable '", group_var,
           "' has values outside [1, n_spatial_units = ",
           n_spatial_units, "]: ",
           paste(unique(bad), collapse = ", "), ". ",
           "Integer `group_var` values are treated as 1-based row indices ",
           "into the adjacency matrix.", call. = FALSE)
    }
    return(idx)
  }

  if (is.character(values) || is.factor(values)) {
    char_vals <- as.character(values)
    if (!is.null(rn)) {
      idx <- match(char_vals, rn)
      missing_vals <- unique(char_vals[is.na(idx)])
      if (length(missing_vals) > 0L) {
        stop("Spatial group variable '", group_var,
             "' has values not found in rownames(adjacency): ",
             paste(missing_vals, collapse = ", "), ".", call. = FALSE)
      }
      return(idx)
    }
    if (all(grepl("^[[:space:]]*[0-9]+[[:space:]]*$", char_vals))) {
      return(.resolve_spatial_idx(as.integer(char_vals), n_spatial_units,
                                  adjacency, group_var))
    }
    if (is.character(values)) {
      stop("Spatial group variable '", group_var, "' holds character labels ",
           "but the adjacency carries no rownames, so nothing says which ",
           "label is which node (sorting the labels gives \"r1\", \"r10\", ",
           "\"r2\", ..., not the graph's order). Set rownames(adjacency) to ",
           "the labels, pass 1-based integer node indices, or pass a factor ",
           "whose levels are in node order.", call. = FALSE)
    }
    f <- values
    if (nlevels(f) != n_spatial_units) {
      stop("Spatial group variable '", group_var,
           "' has ", nlevels(f), " unique values but adjacency has ",
           n_spatial_units, " cells. ",
           "To use a sparse subset of cells, either pass `group_var` as ",
           "integer 1-based indices into the adjacency, or attach ",
           "`rownames(adjacency)` and use matching character / factor ",
           "values in the data.", call. = FALSE)
    }
    return(as.integer(f))
  }

  stop("Spatial group variable '", group_var,
       "' must be integer, numeric, character, or factor.", call. = FALSE)
}


#' Compute valid bounds for rho in proper CAR
#'
#' @description
#' For proper CAR, rho must be in the range (1/lambda_min, 1/lambda_max) where lambda are the
#' eigenvalues of D^(-1)W. In practice, we typically restrict to (0, 1) for
#' interpretability (positive spatial autocorrelation).
#'
#' @param adjacency Adjacency matrix
#'
#' @return Named vector with `lower` and `upper` bounds for rho
#' @keywords internal
compute_car_rho_bounds <- function(adjacency) {
  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  n <- nrow(adj)
  n_neighbors <- rowSums(adj)

  # Check for isolated nodes (no neighbors)
  if (any(n_neighbors == 0)) {
    warning("Adjacency matrix contains isolated nodes (no neighbors).\n",
            "These will be treated as independent.", call. = FALSE)
    # Remove isolated nodes for eigenvalue computation
    keep <- n_neighbors > 0
    adj_sub <- adj[keep, keep]
    n_neighbors_sub <- n_neighbors[keep]

    if (sum(keep) < 2) {
      # Not enough connected nodes
      return(c(lower = 0, upper = 1))
    }

    D_inv <- diag(1 / n_neighbors_sub)
    D_inv_W <- D_inv %*% adj_sub
  } else {
    D_inv <- diag(1 / n_neighbors)
    D_inv_W <- D_inv %*% adj
  }

  # Compute eigenvalues
  eig <- eigen(D_inv_W, symmetric = FALSE, only.values = TRUE)$values
  eig_real <- Re(eig)

  # Theoretical bounds: 1/lambda_min < rho < 1/lambda_max
  lambda_min <- min(eig_real)
  lambda_max <- max(eig_real)

  # Theoretical interval is (1/lambda_min, 1/lambda_max); restrict to (0, 1)
  # for positive-autocorrelation interpretability. For a connected graph
  # D^-1 W is row-stochastic (lambda_max = 1, lambda_min < 0), so this resolves
  # to (0, 1); a non-row-stochastic graph can give a tighter upper bound.
  lower <- max(0, 1 / lambda_min)
  upper <- min(1, 1 / lambda_max)

  # Ensure valid range
  if (lower >= upper) {
    lower <- 0
    upper <- 1
  }

  c(lower = lower, upper = upper)
}

#' BYM2 spatial structure
#'
#' @description
#' Specify a Besag-York-Mollie 2 (BYM2) spatial random effect.
#' BYM2 decomposes the spatial effect into a structured (ICAR) component
#' and an unstructured (IID) component, with a mixing parameter controlling
#' the proportion of variance attributable to spatial structure.
#'
#' BYM2 is preferred over plain CAR when you want to:
#' - Distinguish structured vs unstructured spatial variation
#' - Have an interpretable spatial fraction parameter
#' - Use the scaling from Riebler et al. (2016)
#'
#' @inheritParams spatial_car
#' @param scale_factor Scaling factor for the ICAR component. If NULL
#'   (default), computed from the adjacency matrix following Riebler et al.
#'   On a graph with several connected components each component is scaled
#'   to unit generalized variance separately and an isolated node (island)
#'   gets unit variance on its structured part, following Freni-Sterrantino
#'   et al. (2018). A supplied value is one scale for the whole graph.
#'
#' @return A `tulpa_spatial` object
#'
#' @references
#' Riebler, A., Sorbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
#' Bayesian spatial model for disease mapping that accounts for scaling.
#' Statistical Methods in Medical Research, 25(4), 1145-1165.
#'
#' Freni-Sterrantino, A., Ventrucci, M., & Rue, H. (2018). A note on intrinsic
#' conditional autoregressive models for disconnected graphs. Spatial and
#' Spatio-temporal Epidemiology, 26, 25-34.
#'
#' @examples
#' # Create adjacency matrix for 10 regions (chain structure)
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) {
#'   adj[i, i+1] <- adj[i+1, i] <- 1
#' }
#'
#' # Create BYM2 spatial structure
#' bym2 <- spatial_bym2(adj, level = "group", group_var = "region")
#' print(bym2)
#'
#' \donttest{
#' # Disease mapping with BYM2 spatial smoothing
#' set.seed(456)
#' n_regions <- 10
#' epi_data <- data.frame(
#'   region = factor(rep(1:n_regions, each = 4)),
#'   age = rnorm(n_regions * 4, 50, 10)
#' )
#' epi_data$cases <- rbinom(nrow(epi_data), size = 100, prob = 0.15)
#'
#' fit <- tulpa(
#'   cases ~ age + spatial(region),
#'   spatial = spatial_bym2(adj, level = "group", group_var = "region"),
#'   data = epi_data,
#'   family = "binomial",
#'   n_trials = rep(100L, nrow(epi_data)),
#'   mode = "laplace"
#' )
#' summary(fit)
#' }
#'
#' @export
spatial_bym2 <- function(adjacency, level = c("group", "obs"),
                         group_var = NULL, shared = NULL,
                         scale_factor = NULL,
                         parameterization = c("standard", "collapsed")) {

  level <- match.arg(level)
  parameterization <- match.arg(parameterization)

  adjacency <- .validate_adjacency_arg(adjacency, "adjacency")

  if (level == "group" && is.null(group_var)) {
    stop("`group_var` is required when level = 'group'", call. = FALSE)
  }

  # Collapsed parameterization is deprecated (archived 2026-03-10)
  if (parameterization == "collapsed") {
    warning(
      "Collapsed parameterization is deprecated and will be removed in a future version.\n",
      "Collapsed + HMC creates poor posterior geometry (14x slower than standard).\n",
      "Use standard parameterization instead. A Gibbs backend for large spatial ",
      "models is planned.",
      call. = FALSE
    )
  }

  # Compute scale factor if not provided. Per connected component, with the
  # component-to-component remainder carried as `node_prec`
  # (.bym2_component_scaling); a supplied scale_factor is the caller's single
  # scale for the whole graph and carries none.
  node_prec <- NULL
  if (is.null(scale_factor)) {
    sc <- .bym2_component_scaling(adjacency)
    scale_factor <- sc$scale_factor
    node_prec <- sc$node_prec
  } else if (!is.numeric(scale_factor) || length(scale_factor) != 1L ||
             !is.finite(scale_factor) || scale_factor <= 0) {
    stop("`scale_factor` must be a single positive number.", call. = FALSE)
  }

  if (isFALSE(shared)) .warn_nonshared("spatial effects")

  structure(
    list(
      type = "bym2",
      adjacency = adjacency,
      level = level,
      group_var = group_var,
      shared = shared,
      parameterization = parameterization,
      n_spatial = nrow(adjacency),
      scale_factor = scale_factor,
      node_prec = node_prec
    ),
    class = c("tulpa_spatial", "list")
  )
}

#' Compute BYM2 scaling factor
#'
#' @description
#' Compute the scaling factor for BYM2 following Riebler et al. (2016).
#' This makes the spatial fraction parameter interpretable. On a disconnected
#' graph each connected component is scaled separately and an isolated node
#' (an island) gets unit variance (Freni-Sterrantino et al. 2018); the value
#' returned is then the reference scale of the largest component, and
#' `.bym2_component_scaling()` carries the per-node remainder.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Scaling factor (scalar)
#' @keywords internal
compute_bym2_scale <- function(adjacency) {
  .bym2_component_scaling(adjacency)$scale_factor
}

# Per-component BYM2 scaling (Freni-Sterrantino, Ventrucci & Rue 2018, the
# INLA `scale.model` convention for a disconnected graph; gcol33/tulpa#902).
#
# Riebler et al. (2016) scale the structured field so its generalised variance
# -- the geometric mean of the marginal variances diag(Q^+) -- is one. Over a
# whole disconnected graph that mean mixes pieces of different geometry, and on
# an isolated node diag(Q^+) is 0, so log(0) made the scale infinite and every
# backend then failed its own way (NA coefficients, a frozen sampler). Scaled
# per component instead: component c of size >= 2 gets
# s_c = 1 / sqrt(gv_c), gv_c the geometric mean of diag(L_c^+) over its own
# nodes, and an island gets s_c = 1, i.e. unit variance on its structured part.
#
# The engine's BYM2 enters the linear predictor through ONE scalar,
# sigma * sqrt(rho) * scale_factor * phi, with phi's own ICAR prior at unit
# precision. So the scalar is the reference scale s_ref (the largest component's,
# which is the whole-graph value on a connected map) and each component's
# remaining factor rides on phi's prior as a precision multiplier,
# node_prec_i = (s_ref / s_c)^2 for node i of component c: s_ref * phi then has
# precision L_c / s_c^2 on component c and variance 1 on an island. `node_prec`
# is NULL when every multiplier is 1 (a connected graph), which keeps every
# kernel on its unweighted path.
#' @keywords internal
.bym2_component_scaling <- function(adjacency) {
  adj <- as.matrix(adjacency)
  diag(adj) <- 0
  n <- nrow(adj)
  comps <- .graph_components(adj)
  s_c <- vapply(comps, function(cc) {
    if (length(cc) < 2L) return(1)
    Wc <- adj[cc, cc, drop = FALSE]
    Lc <- diag(rowSums(Wc), length(cc)) - Wc
    # L^+ = (L + 11'/m)^{-1} - 11'/m on a connected component of size m.
    m  <- length(cc)
    gv_diag <- diag(chol2inv(chol(Lc + 1 / m))) - 1 / m
    1 / sqrt(exp(mean(log(gv_diag))))
  }, numeric(1))
  sizes <- lengths(comps)
  s_ref <- if (any(sizes >= 2L)) s_c[which.max(sizes)] else 1
  node_prec <- numeric(n)
  for (k in seq_along(comps)) node_prec[comps[[k]]] <- (s_ref / s_c[k])^2
  if (isTRUE(all.equal(node_prec, rep(1, n), tolerance = 1e-12))) {
    node_prec <- NULL
  }
  list(scale_factor = s_ref, node_prec = node_prec, component_scale = s_c,
       components = comps)
}

#' Print method for tulpa_spatial
#'
#' @param x A tulpa_spatial object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the spatial specification to the console.
#'
#' @export
print.tulpa_spatial <- function(x, ...) {
  # SPDE specs carry a different field set (Matern mesh) and print separately.
  # A single method handles every tulpa_spatial type here, so no other file
  # redefines print.tulpa_spatial (which would shadow the areal branch below).
  if (!is.null(x$type) && x$type == "spde") {
    cat("tulpa_spatial: SPDE (Matern, nu =", x$nu, ")\n")
    cat("  Mesh nodes:", x$n_mesh, "\n")
    if (!is.null(x$mesh)) {
      cat("  Triangles: ", x$mesh$n_triangles, "\n")
    }
    if (!is.null(x$obs_coords)) {
      cat("  Observations:", nrow(x$obs_coords), "\n")
    }
    cat("  Prior range: P(range <", x$prior_range[1], ") =", x$prior_range[2], "\n")
    cat("  Prior sigma: P(sigma >", x$prior_sigma[1], ") =", x$prior_sigma[2], "\n")
    return(invisible(x))
  }

  cat("tulpa spatial specification\n")
  cat("===========================\n\n")

  # Format type name
  type_name <- switch(x$type,
    car = "ICAR (Intrinsic CAR)",
    car_proper = "Proper CAR",
    bym2 = "BYM2",
    toupper(x$type)
  )
  cat("Type:", type_name, "\n")
  cat("Level:", x$level, "\n")
  cat("Spatial units:", x$n_spatial, "\n")
  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  if (x$type == "car_proper" && !is.null(x$rho_bounds)) {
    cat("Rho bounds: [", round(x$rho_bounds["lower"], 4), ", ",
        round(x$rho_bounds["upper"], 4), "]\n", sep = "")
    cat("  (spatial autocorrelation parameter, estimated from data)\n")
  }

  if (x$type == "car") {
    cat("  (rho fixed at 1, sum-to-zero constraint applied)\n")
  }

  if (x$type == "bym2") {
    cat("Scale factor:", round(x$scale_factor, 4), "\n")
    if (!is.null(x$node_prec)) {
      cat("  (per connected component: ",
          length(unique(round(x$node_prec, 10))), " distinct scale(s) over ",
          .graph_n_components(x$adjacency), " components; islands at unit ",
          "variance)\n", sep = "")
    }
  }

  invisible(x)
}

#' Check if adjacency matrix is connected
#'
#' @description
#' Check if the spatial graph defined by the adjacency matrix is fully
#' connected. A disconnected graph can cause identifiability issues.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Logical; TRUE if connected
#' @keywords internal
is_connected <- function(adjacency) {
  n <- nrow(adjacency)
  if (n == 0) return(TRUE)

  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  # BFS to check connectivity
  visited <- logical(n)
  queue <- 1L
  visited[1] <- TRUE

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]

    neighbors <- which(adj[current, ] > 0)
    new_neighbors <- neighbors[!visited[neighbors]]

    visited[new_neighbors] <- TRUE
    queue <- c(queue, new_neighbors)
  }

  all(visited)
}
