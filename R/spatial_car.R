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
#   * character / factor values, no rownames
#                              -> fall back to as.integer(as.factor(.)),
#                                  but require the number of unique values
#                                  to equal n_spatial_units (otherwise the
#                                  factor levels reindex the cells and lose
#                                  the user's cell identity).
#
# The fallback restriction is the original (pre-issue-#25) behavior, kept
# for backward compatibility with tests that pass factors covering every
# cell. Users hitting the empty-cell case should switch to integer 1-based
# indices or attach `rownames(adjacency)`.
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
    f <- as.factor(values)
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

  # For positive autocorrelation, restrict to (0, 1)
  # The upper bound is typically 1/lambda_max ~= 1 for connected graphs
  lower <- max(0, 1 / lambda_max)
  upper <- min(1, 1 / lambda_min)

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
#'
#' @return A `tulpa_spatial` object
#'
#' @references
#' Riebler, A., Sorbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
#' Bayesian spatial model for disease mapping that accounts for scaling.
#' Statistical Methods in Medical Research, 25(4), 1145-1165.
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

  # Compute scale factor if not provided
  if (is.null(scale_factor)) {
    scale_factor <- compute_bym2_scale(adjacency)
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
      scale_factor = scale_factor
    ),
    class = c("tulpa_spatial", "list")
  )
}

#' Compute BYM2 scaling factor
#'
#' @description
#' Compute the scaling factor for BYM2 following Riebler et al. (2016).
#' This makes the spatial fraction parameter interpretable.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Scaling factor (scalar)
#' @keywords internal
compute_bym2_scale <- function(adjacency) {
  # Build precision matrix Q for ICAR
  n <- nrow(adjacency)
  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  # Number of neighbors for each area
  n_neighbors <- rowSums(adj)

  # ICAR precision matrix: Q_ii = n_neighbors[i], Q_ij = -1 if neighbors
  Q <- diag(n_neighbors) - adj

  # Compute generalized inverse (Q is rank-deficient)
  # Use eigendecomposition
  eig <- eigen(Q, symmetric = TRUE)

  # Remove the zero eigenvalue (rank deficiency)
  non_zero <- abs(eig$values) > 1e-10
  lambda <- eig$values[non_zero]

  # Geometric mean of non-zero eigenvalues
  # This is the scaling factor following INLA convention
  scale <- exp(mean(log(lambda)))

  scale
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
  # Single method for every tulpa_spatial type: sourcing spatial_rsr_spde.R
  # after this file used to redefine print.tulpa_spatial and shadow the areal
  # branch below (its NextMethod() fell through to print.default).
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
