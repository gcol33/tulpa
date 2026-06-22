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

  # Validate adjacency matrix
  if (inherits(adjacency, "tulpa_adjacency")) {
    adjacency <- adjacency$adjacency
  }
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix", call. = FALSE)
  }

  if (nrow(adjacency) != ncol(adjacency)) {
    stop("`adjacency` must be square", call. = FALSE)
  }

  # Check symmetry
  if (!isSymmetric(unname(as.matrix(adjacency)))) {
    stop("`adjacency` must be symmetric", call. = FALSE)
  }

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

  # Compute eigenvalue bounds for proper CAR (needed for valid ρ range)
  rho_bounds <- NULL
  if (proper) {
    rho_bounds <- compute_car_rho_bounds(adjacency)
  }

  # Warning for non-shared spatial effects
  if (isFALSE(shared)) {
    warning(
      "Non-shared spatial effects (shared = FALSE) means effects are not shared across processes.\n",
      "Consider whether spatial effects should be shared between\n",
      "processes if shared confounding structure is expected.",
      call. = FALSE
    )
  }

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

  # Theoretical bounds: 1/λ_min < ρ < 1/λ_max
  lambda_min <- min(eig_real)
  lambda_max <- max(eig_real)

  # For positive autocorrelation, restrict to (0, 1)
  # The upper bound is typically 1/λ_max ≈ 1 for connected graphs
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
#' \dontrun{
#' # Generate synthetic epidemiological data (not run - requires laplace backend)
#' set.seed(456)
#' n_regions <- 10
#' epi_data <- data.frame(
#'   region = factor(1:n_regions),
#'   age = rnorm(n_regions, 50, 10),
#'   cases = rbinom(n_regions, size = 100, prob = 0.15),
#'   population = rep(100L, n_regions)
#' )
#'
#' # Disease mapping with BYM2 spatial smoothing
#' fit <- tulpa(
#'   cases | population ~ age,
#'   spatial = spatial_bym2(adj, level = "group", group_var = "region"),
#'   data = epi_data,
#'   family = tulpa_binomial(),
#'   backend = "laplace"
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

  # Validate adjacency matrix
  if (inherits(adjacency, "tulpa_adjacency")) {
    adjacency <- adjacency$adjacency
  }
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix", call. = FALSE)
  }

  if (nrow(adjacency) != ncol(adjacency)) {
    stop("`adjacency` must be square", call. = FALSE)
  }

  if (!isSymmetric(unname(as.matrix(adjacency)))) {
    stop("`adjacency` must be symmetric", call. = FALSE)
  }

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

  # Warning for non-shared spatial effects
  if (isFALSE(shared)) {
    warning(
      "Non-shared spatial effects (shared = FALSE) means effects are not shared across processes.\n",
      "Consider whether spatial effects should be shared between\n",
      "processes if shared confounding structure is expected.",
      call. = FALSE
    )
  }

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
#' @export
print.tulpa_spatial <- function(x, ...) {
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

#' Gaussian Process spatial structure
#'
#' @description
#' Specify a Gaussian Process (GP) spatial random effect using coordinate-based
#' distances. Uses Nearest Neighbor Gaussian Process (NNGP) approximation for
#' computational efficiency with large datasets (scales to millions of observations).
#'
#' This provides a continuous spatial effect that captures smooth spatial
#' variation, unlike CAR/BYM2 which require discrete areal units.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param cov Covariance function: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nu Smoothness parameter for Matern covariance. Common values:
#'   - 0.5: Equivalent to exponential (rough, once mean-square differentiable)
#'   - 1.5: Once differentiable (moderate smoothness)
#'   - 2.5: Twice differentiable (smooth)
#'   Ignored for non-Matern covariance functions.
#' @param nn Number of nearest neighbors for NNGP approximation. Default 15.
#'   Larger values give better approximation but slower computation.
#' @param solver Linear algebra solver for GP computations:
#'   - `"auto"` (default): Automatically select based on problem size.
#'     Uses Cholesky for N < 2000, CG for larger problems. Uses GPU if
#'     available and N > 5000.
#'   - `"cholesky"`: Direct Cholesky decomposition. Exact but O(N*k^3).
#'     Best for smaller datasets or when high precision is critical.
#'   - `"cg"`: Conjugate Gradient iterative solver. O(N*k^2*iter).
#'     Better for large N (> 2000) with good preconditioning.
#'   - `"pcg"`: Preconditioned CG with diagonal preconditioner.
#'     Faster convergence than CG for ill-conditioned systems.
#'   - `"gpu"`: GPU-accelerated batched Cholesky using CUDA (if available).
#'     Requires tulpa to be compiled with GPU support. Falls back to PCG
#'     if GPU is unavailable. Best for large datasets (N > 5000) with
#'     CUDA-capable GPU. GPU support is detected automatically at fit time.
#' @param cg_tol Convergence tolerance for CG/PCG solvers. Default 1e-6.
#'   Smaller values give more accurate solutions but slower convergence.
#' @param cg_maxiter Maximum iterations for CG/PCG. Default 100.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   all processes. Set to FALSE for process-specific spatial
#'   effects (triggers warning about potential confounding).
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing distances.
#' @param parameterization GP parameterization: `"centered"` (default) samples
#'   GP effects directly; `"noncentered"` uses z ~ N(0,1) with scaling;
#'   `"collapsed"` (deprecated) marginalizes via inner Laplace.
#'
#' @return A `tulpa_gp` object
#'
#' @details
#' The GP spatial model adds a spatially-correlated random effect to the
#' linear predictor:
#'
#' \deqn{\eta(s) = X\beta + w(s)}
#'
#' where \eqn{w(s)} follows a Gaussian process:
#' \deqn{w(s) \sim GP(0, \sigma^2 C(d; \phi))}
#'
#' The correlation function \eqn{C(d; \phi)} depends on distance \eqn{d} and
#' range parameter \eqn{\phi}:
#'
#' - **Exponential**: \eqn{C(d) = \exp(-d/\phi)}
#' - **Matern**: \eqn{C(d) = \frac{2^{1-\nu}}{\Gamma(\nu)} (\sqrt{2\nu} d/\phi)^\nu K_\nu(\sqrt{2\nu} d/\phi)}
#' - **Gaussian**: \eqn{C(d) = \exp(-(d/\phi)^2)}
#' - **Spherical**: \eqn{C(d) = 1 - 1.5(d/\phi) + 0.5(d/\phi)^3} for \eqn{d < \phi}
#'
#' **NNGP approximation**: For computational tractability, we use the
#' Nearest Neighbor Gaussian Process (Datta et al., 2016), which conditions
#' each location on its k nearest neighbors. This reduces complexity from
#' O(n^3) to O(n*k^2), enabling models with millions of observations.
#'
#' @examples
#' # Create GP spatial structure
#' gp <- spatial_gp(~ lon + lat)
#' print(gp)
#'
#' \donttest{
#' # Generate synthetic spatial data
#' set.seed(789)
#' n <- 50
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n),
#'   count = rpois(n, 25),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Continuous spatial effect with exponential covariance
#' fit <- tulpa(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_gp(~ lon + lat),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#' summary(fit)
#'
#' # Smoother spatial field with Matern covariance
#' fit2 <- tulpa(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_gp(~ lon + lat, cov = "matern", nu = 1.5),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#' }
#'
#' @references
#' Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016).
#' Hierarchical nearest-neighbor Gaussian process models for large
#' geostatistical datasets. Journal of the American Statistical Association,
#' 111(514), 800-812.
#'
#' @seealso [spatial_car()], [spatial_bym2()] for areal spatial effects,
#'   [spatial_svc()] for spatially-varying coefficients,
#'   [spatial_multiscale()] for multi-scale spatial effects
#'
#' @export
