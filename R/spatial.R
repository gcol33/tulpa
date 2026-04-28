#' Spatial structure specifications for tulpa
#'
#' @description
#' Functions to specify spatial random effects for tulpa models.
#' Spatial effects are shared between processes by default,
#' which helps prevent bias from spatially-structured
#' unmeasured confounders.
#'
#' @name tulpa_spatial
NULL

#' CAR spatial structure
#'
#' @description
#' Specify a conditional autoregressive (CAR) spatial random effect.
#' Supports both intrinsic CAR (ICAR) and proper CAR variants.
#'
#' @param adjacency Adjacency matrix (sparse or dense). A symmetric matrix
#'   where entry (i,j) is 1 if areas i and j are neighbors, 0 otherwise.
#' @param level Level at which spatial structure applies:
#'   - `"group"`: Spatial effect at the grouping variable level (e.g., sites).
#'     Requires `group_var` to be specified.
#'   - `"obs"`: Spatial effect at the observation level.
#' @param group_var Name of the grouping variable in data (required if
#'   `level = "group"`).
#' @param proper Logical; if FALSE (default), uses Intrinsic CAR (ICAR) with
#'   ρ = 1 fixed. If TRUE, uses proper CAR with ρ estimated. See Details.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   all process linear predictors identically.
#' @param parameterization Spatial parameterization: `"standard"` (default)
#'   samples spatial effects directly; `"collapsed"` (deprecated) marginalizes
#'   spatial effects via inner Laplace approximation.
#'
#' @return A `tulpa_spatial` object
#'
#' @details
#' The CAR model specifies that:
#'
#' \deqn{\phi_i | \phi_{-i} \sim N\left(\rho \frac{\sum_{j \sim i} \phi_j}{n_i},
#'   \frac{\sigma^2}{n_i}\right)}
#'
#' where \eqn{n_i} is the number of neighbors of area i and \eqn{j \sim i}
#' denotes that j is a neighbor of i.
#'
#' This leads to a precision matrix:
#' \deqn{Q = \tau (D - \rho W)}
#'
#' where D is the diagonal matrix of neighbor counts and W is the adjacency
#' matrix.
#'
#' **ICAR (proper = FALSE, default)**
#'
#' - Sets ρ = 1 (fixed)
#' - Improper prior (rank-deficient Q)
#' - Requires sum-to-zero constraint for identifiability
#' - Simpler: one fewer parameter to estimate
#' - Standard choice for disease mapping
#' - Equivalent to RW1 on a graph
#'
#' **Proper CAR (proper = TRUE)**
#'
#' - Estimates ρ ∈ (0, 1) from data
#' - Proper prior (integrates to 1)
#' - ρ measures spatial autocorrelation strength
#' - ρ → 0: approaches independence (IID)
#' - ρ → 1: approaches ICAR
#' - More flexible but additional parameter to estimate
#' - Prior: ρ ~ Beta(1, 1) (uniform on (0, 1))
#'
#' @examples
#' # Create adjacency matrix for 10 regions (chain structure)
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) {
#'   adj[i, i+1] <- adj[i+1, i] <- 1
#' }
#'
#' # ICAR (default, ρ = 1 fixed)
#' icar <- spatial_car(adj, level = "group", group_var = "site")
#' print(icar)
#'
#' # Proper CAR (ρ estimated)
#' proper_car <- spatial_car(adj, level = "group", group_var = "site",
#'                           proper = TRUE)
#' print(proper_car)
#'
#' \donttest{
#' # Generate synthetic data with spatial structure
#' set.seed(123)
#' n_sites <- 10
#' n_per_site <- 5
#' df <- data.frame(
#'   site = rep(1:n_sites, each = n_per_site),
#'   x = rnorm(n_sites * n_per_site),
#'   count = rpois(n_sites * n_per_site, 20),
#'   effort = rgamma(n_sites * n_per_site, shape = 5, rate = 1)
#' )
#'
#' # ICAR model (standard disease mapping approach)
#' fit_icar <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_car(adj, level = "group", group_var = "site"),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Proper CAR model (estimate spatial autocorrelation)
#' fit_car <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_car(adj, level = "group", group_var = "site",
#'                         proper = TRUE),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Extract ρ from proper CAR
#' summary(fit_car)  # Shows rho_spatial parameter
#' }
#'
#' @seealso [spatial_bym2()] for decomposed spatial + IID effects,
#'   [spatial_gp()] for continuous spatial effects
#'
#' @export
spatial_car <- function(adjacency, level = c("group", "obs"),
                        group_var = NULL, proper = FALSE, shared = NULL,
                        parameterization = c("standard", "collapsed")) {

  level <- match.arg(level)
  parameterization <- match.arg(parameterization)

  # Validate adjacency matrix
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
#' autocorrelation parameter ρ estimated from the data.
#'
#' Use this when you want spatial autocorrelation to be a parameter of the
#' model rather than fixed at 1 (as in ICAR). ρ ≈ 0 collapses to IID,
#' ρ ≈ 1 approaches ICAR.
#'
#' @inheritParams spatial_car
#'
#' @return A `tulpa_spatial` object with `type = "car_proper"`.
#'
#' @seealso [spatial_car()] for ICAR (ρ fixed at 1),
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
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

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
#'     CUDA-capable GPU. Use [gpu_available()] to check support.
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
spatial_gp <- function(coords,
                       cov = c("exponential", "matern", "gaussian", "spherical"),
                       nu = 1.5,
                       nn = 15,
                       solver = c("auto", "cholesky", "cg", "pcg", "gpu"),
                       cg_tol = 1e-6,
                       cg_maxiter = 100,
                       shared = NULL,
                       scale_coords = TRUE,
                       parameterization = c("centered", "noncentered", "collapsed")) {

  cov <- match.arg(cov)
  solver <- match.arg(solver)
  parameterization <- match.arg(parameterization)

  # Collapsed parameterization is deprecated (archived 2026-03-10)
  if (parameterization == "collapsed") {
    warning(
      "Collapsed parameterization is deprecated and will be removed in a future version.\n",
      "Collapsed + HMC creates poor posterior geometry.\n",
      "Use 'centered' or 'noncentered' instead.",
      call. = FALSE
    )
  }

  # Check GPU availability if requested

if (solver == "gpu" && !gpu_available()) {
    warning("GPU solver requested but GPU support is not available. ",
            "Falling back to PCG solver. ",
            "To enable GPU support, reinstall tulpa with CUDA.",
            call. = FALSE)
    solver <- "pcg"
  }

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Validate nu for Matern
  if (cov == "matern") {
    if (!is.numeric(nu) || length(nu) != 1 || nu <= 0) {
      stop("`nu` must be a positive number for Matern covariance", call. = FALSE)
    }
  }

  # Validate nn
  if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
    stop("`nn` must be a positive integer", call. = FALSE)
  }
  nn <- as.integer(nn)

  # Validate CG parameters
  if (!is.numeric(cg_tol) || length(cg_tol) != 1 || cg_tol <= 0) {
    stop("`cg_tol` must be a positive number", call. = FALSE)
  }
  if (!is.numeric(cg_maxiter) || length(cg_maxiter) != 1 || cg_maxiter < 1) {
    stop("`cg_maxiter` must be a positive integer", call. = FALSE)
  }
  cg_maxiter <- as.integer(cg_maxiter)

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
      type = "gp",
      coord_vars = coord_vars,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      nn = nn,
      solver = solver,
      cg_tol = cg_tol,
      cg_maxiter = cg_maxiter,
      shared = shared,
      scale_coords = scale_coords,
      parameterization = parameterization,
      # Filled in during validation
      n_obs = NULL,
      n_spatial = NULL,
      coords_matrix = NULL,
      neighbor_info = NULL
    ),
    class = c("tulpa_gp", "tulpa_spatial", "list")
  )
}


#' Print method for tulpa_gp
#'
#' @param x A tulpa_gp object
#' @param ... Ignored
#'
#' @export
print.tulpa_gp <- function(x, ...) {
  cat("tulpa Gaussian Process spatial specification\n")
  cat("=============================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cov_str <- x$cov
  if (x$cov == "matern" && !is.null(x$nu)) {
    cov_str <- sprintf("%s (nu = %.1f)", x$cov, x$nu)
  }
  cat("Covariance:", cov_str, "\n")
  cat("Neighbors (NNGP):", x$nn, "\n")

  # Solver info
  solver_str <- switch(x$solver,
    auto = "auto (Cholesky<2k, PCG<5k, GPU/CG for larger)",
    cholesky = "Cholesky (exact, O(N*k^3))",
    cg = sprintf("CG (iterative, tol=%.0e, maxiter=%d)", x$cg_tol, x$cg_maxiter),
    pcg = sprintf("PCG (preconditioned, tol=%.0e, maxiter=%d)", x$cg_tol, x$cg_maxiter),
    gpu = "GPU (CUDA/OpenCL batched Cholesky)",
    x$solver
  )
  cat("Solver:", solver_str, "\n")

  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Hilbert Space Gaussian Process (HSGP) spatial structure
#'
#' @description
#' Specify a Hilbert Space Gaussian Process approximation for spatial effects.
#' HSGP approximates a GP using Laplacian eigenfunctions, providing O(N*M^2)
#' complexity with analytical gradients instead of O(N*k^2) with numerical
#' gradients for NNGP.
#'
#' This gives approximately 50x speedup over standard NNGP while maintaining
#' high accuracy for smooth spatial fields.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param m Number of basis functions per dimension. Total basis functions
#'   will be m^2. Default 6. Higher values give better approximation but
#'   slower computation. Recommended range: 5-15.
#' @param c Boundary factor controlling domain extension beyond data range.
#'   Default 1.5. The domain is extended to \eqn{(-cL, cL)} where L is half the
#'   data range. Larger values improve approximation at boundaries.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   all processes. Set to FALSE for process-specific spatial
#'   effects (triggers warning about potential confounding).
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing basis functions.
#'
#' @return A `tulpa_hsgp` object
#'
#' @details
#' HSGP approximates a GP as:
#'
#' \deqn{f(x) = \sum_{j=1}^{M^2} \phi_j(x) \sqrt{S(\lambda_j)} \beta_j}
#'
#' where:
#' - \eqn{\phi_j(x)} are Laplacian eigenfunctions (products of sines)
#' - \eqn{S(\lambda_j)} is the spectral density of the squared exponential kernel
#' - \eqn{\beta_j \sim N(0, 1)} are basis coefficients
#'
#' The hyperparameters are:
#' - \eqn{\sigma^2}: marginal variance (PC prior: P(sigma > 1) = 0.01)
#' - \eqn{\ell}: lengthscale (LogNormal(0, 1) prior)
#'
#' **Advantages over NNGP**:
#' - Analytical gradients enable ~50x speedup
#' - Simple parameter interpretation
#' - Works well for smooth spatial fields
#'
#' **Limitations**:
#' - Assumes squared exponential (smooth) covariance
#' - Less accurate for rough fields (Matern with low nu)
#' - Approximation quality depends on m and c choices
#'
#' @examples
#' # Create HSGP spatial structure
#' hsgp <- spatial_hsgp(~ lon + lat)
#' print(hsgp)
#'
#' \donttest{
#' # Generate synthetic spatial data
#' set.seed(42)
#' n <- 100
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   x = rnorm(n),
#'   count = rpois(n, 25),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Fast spatial effect with HSGP (much faster than spatial_gp)
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_hsgp(~ lon + lat, m = 6),
#'   backend = "hmc",
#'   iter = 500,
#'   warmup = 250,
#'   chains = 2
#' )
#' summary(fit)
#' }
#'
#' @references
#' Riutort-Mayol, G., Bürkner, P. C., Andersen, M. R., Solin, A., & Vehtari, A.
#' (2023). Practical Hilbert space approximate Bayesian Gaussian processes for
#' probabilistic programming. Statistics and Computing, 33(1), 17.
#'
#' @seealso [spatial_gp()] for NNGP-based GP, [spatial_car()] for areal effects
#'
#' @export
spatial_hsgp <- function(coords,
                         m = 6,
                         c = 1.5,
                         shared = NULL,
                         scale_coords = TRUE) {

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Validate m
  if (!is.numeric(m) || length(m) != 1 || m < 3 || m > 50) {
    stop("`m` must be an integer between 3 and 50", call. = FALSE)
  }
  m <- as.integer(m)

  # Validate c
  if (!is.numeric(c) || length(c) != 1 || c < 1) {
    stop("`c` (boundary factor) must be >= 1", call. = FALSE)
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
      type = "hsgp",
      coord_vars = coord_vars,
      m = m,
      c = c,
      shared = shared,
      scale_coords = scale_coords,
      # Filled in during validation
      n_obs = NULL,
      coords_matrix = NULL
    ),
    class = c("tulpa_hsgp", "tulpa_spatial", "list")
  )
}


#' Print method for tulpa_hsgp
#'
#' @param x A tulpa_hsgp object
#' @param ... Ignored
#'
#' @export
print.tulpa_hsgp <- function(x, ...) {
  cat("tulpa Hilbert Space GP (HSGP) spatial specification\n")
  cat("=====================================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Basis functions:", x$m, "per dim (", x$m^2, "total )\n")
  cat("Boundary factor:", x$c, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate HSGP spatial structure
#'
#' @param spatial A tulpa_hsgp object
#' @param data The data frame
#'
#' @return A validated tulpa_hsgp object with coords_matrix filled in
#'
#' @keywords internal
validate_hsgp <- function(spatial, data) {
  # Check coordinate columns exist
  missing_cols <- setdiff(spatial$coord_vars, names(data))
  if (length(missing_cols) > 0) {
    stop("Coordinate columns not found in data: ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  # Extract coordinates
  coords <- as.matrix(data[, spatial$coord_vars, drop = FALSE])

  # Check for missing values
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale if requested
  if (spatial$scale_coords) {
    coords <- scale(coords)
  }

  # Store validated data
  spatial$n_obs <- nrow(coords)
  spatial$coords_matrix <- coords

  spatial
}


#' Validate HSGP-MSGP (multi-scale GP with HSGP approximation)
#' @noRd
validate_hsgp_multiscale <- function(spatial, data) {
  # Check coordinate columns exist
  missing_cols <- setdiff(spatial$coord_vars, names(data))
  if (length(missing_cols) > 0) {
    stop("Coordinate columns not found in data: ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  # Extract coordinates
  coords <- as.matrix(data[, spatial$coord_vars, drop = FALSE])

  # Check for missing values
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale if requested
  if (isTRUE(spatial$scale_coords)) {
    coords <- scale(coords)
  }

  list(
    coords_matrix = coords,
    n_obs = nrow(coords)
  )
}


#' Multi-Scale Gaussian Process spatial structure
#'
#' @description
#' Specify a multi-scale spatial random effect that decomposes spatial
#' variation into local (fine-scale) and regional (broad-scale) components.
#' Each scale has its own range and variance parameters.
#'
#' This is particularly useful for large datasets (>100k observations) where
#' spatial patterns exist at multiple scales.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param scales Character vector specifying scale names. Default: `c("local", "regional")`.
#' @param range_local Prior range for local scale as `c(lower, upper)` in
#'   coordinate units. Default: `c(0.01, 1)` (after scaling).
#' @param range_regional Prior range for regional scale as `c(lower, upper)`.
#'   Default: `c(1, 10)` (after scaling).
#' @param cov Covariance function: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nu Smoothness parameter for Matern covariance.
#' @param nn_local Number of nearest neighbors for local scale. Default 10.
#' @param nn_regional Number of nearest neighbors for regional scale. Default 30.
#' @param shared Logical; if TRUE (default), spatial effects enter both
#'   all processes.
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing distances.
#' @param approx Approximation method: `"nngp"` (default) for Nearest Neighbor
#'   GP; `"hsgp"` for Hilbert Space GP (faster for smooth fields).
#' @param m Number of HSGP basis functions per dimension (default 6). Only
#'   used when `approx = "hsgp"`. Total basis functions will be m^2.
#' @param c_boundary Boundary factor for HSGP domain extension (default 1.5).
#'   Only used when `approx = "hsgp"`.
#' @param sampler Sampling strategy: `"auto"` (default) selects based on model;
#'   `"noncentered"` uses z ~ N(0,1); `"centered"` samples effects directly;
#'   `"interweaved"` alternates between parameterizations.
#'
#' @return A `tulpa_multiscale` object
#'
#' @details
#' The multi-scale model decomposes spatial variation additively:
#'
#' \deqn{\eta(s) = X\beta + w_{local}(s) + w_{regional}(s)}
#'
#' where each component follows an independent Gaussian process:
#' \deqn{w_{local}(s) \sim GP(0, \sigma^2_{local} C(\phi_{local}))}
#' \deqn{w_{regional}(s) \sim GP(0, \sigma^2_{regional} C(\phi_{regional}))}
#'
#' **Identifiability**: With sufficient data (>500 locations), the two scales
#' are typically well-identified when prior ranges are non-overlapping.
#' PC priors on variance components help prevent overfitting.
#'
#' **Computational cost**: Approximately 1.5-2x the cost of single-scale GP,
#' as two NNGP likelihoods must be evaluated.
#'
#' @examples
#' # Create multi-scale spatial structure
#' ms <- spatial_multiscale(
#'   ~ lon + lat,
#'   range_local = c(0.1, 0.5),
#'   range_regional = c(1, 5)
#' )
#' print(ms)
#'
#' \dontrun{
#' # Generate synthetic spatial data (not run - multiscale not fully supported)
#' set.seed(101)
#' n <- 60
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n),
#'   count = rpois(n, 30),
#'   effort = rgamma(n, shape = 5, rate = 1)
#' )
#'
#' # Multi-scale spatial with local and regional components
#' fit <- tulpa(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_multiscale(
#'     ~ lon + lat,
#'     range_local = c(0.1, 0.5),
#'     range_regional = c(1, 5)
#'   ),
#'   iter = 200, warmup = 100, chains = 1
#' )
#' summary(fit)
#' }
#'
#' @seealso [spatial_gp()] for single-scale GP, [temporal_multiscale()] for
#'   multi-scale temporal effects
#'
#' @export
spatial_multiscale <- function(coords,
                               scales = c("local", "regional"),
                               approx = c("nngp", "hsgp"),
                               m = 6L,
                               c_boundary = 1.5,
                               range_local = c(0.01, 1),
                               range_regional = c(1, 10),
                               cov = c("exponential", "matern", "gaussian", "spherical"),
                               nu = 1.5,
                               nn_local = 10,
                               nn_regional = 30,
                               shared = NULL,
                               scale_coords = TRUE,
                               sampler = c("auto", "noncentered", "centered",
                                          "interweaved", "adaptive", "riemannian", "lbfgs")) {

  approx <- match.arg(approx)
  cov <- match.arg(cov)
  sampler <- match.arg(sampler)

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Validate scales
  if (length(scales) != 2) {
    stop("Currently only 2 scales are supported", call. = FALSE)
  }

  # Validate range specifications
  if (length(range_local) != 2 || range_local[1] >= range_local[2]) {
    stop("`range_local` must be c(lower, upper) with lower < upper", call. = FALSE)
  }
  if (length(range_regional) != 2 || range_regional[1] >= range_regional[2]) {
    stop("`range_regional` must be c(lower, upper) with lower < upper", call. = FALSE)
  }

  # Check range separation
  if (range_local[2] > range_regional[1]) {
    warning(
      "Local and regional range priors overlap.\n",
      "This may cause identifiability issues. Consider separating the ranges:\n",
      sprintf("  Local: [%.2f, %.2f], Regional: [%.2f, %.2f]",
              range_local[1], range_local[2], range_regional[1], range_regional[2]),
      call. = FALSE
    )
  }

  # Validate nn
  if (!is.numeric(nn_local) || nn_local < 1) {
    stop("`nn_local` must be a positive integer", call. = FALSE)
  }
  if (!is.numeric(nn_regional) || nn_regional < 1) {
    stop("`nn_regional` must be a positive integer", call. = FALSE)
  }

  # Warning for non-shared
  if (isFALSE(shared)) {
    warning(
      "Non-shared multi-scale spatial effects (shared = FALSE) means effects are not shared across processes.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "multiscale",
      approx = approx,
      m = as.integer(m),
      c_boundary = c_boundary,
      coord_vars = coord_vars,
      scales = scales,
      range_local = range_local,
      range_regional = range_regional,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      nn_local = as.integer(nn_local),
      nn_regional = as.integer(nn_regional),
      shared = shared,
      scale_coords = scale_coords,
      sampler = sampler,
      # Filled in during validation
      n_obs = NULL,
      n_spatial = NULL,
      coords_matrix = NULL,
      neighbor_info_local = NULL,
      neighbor_info_regional = NULL
    ),
    class = c("tulpa_multiscale", "tulpa_spatial", "list")
  )
}


#' Print method for tulpa_multiscale
#'
#' @param x A tulpa_multiscale object
#' @param ... Ignored
#'
#' @export
print.tulpa_multiscale <- function(x, ...) {
  cat("tulpa Multi-Scale spatial specification\n")
  cat("========================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Scales:", paste(x$scales, collapse = " + "), "\n\n")

  cat("Local scale:\n")
  cat("  Range prior: [", x$range_local[1], ", ", x$range_local[2], "]\n", sep = "")
  cat("  Neighbors:", x$nn_local, "\n")

  cat("\nRegional scale:\n")
  cat("  Range prior: [", x$range_regional[1], ", ", x$range_regional[2], "]\n", sep = "")
  cat("  Neighbors:", x$nn_regional, "\n")

  cov_str <- x$cov
  if (x$cov == "matern" && !is.null(x$nu)) {
    cov_str <- sprintf("%s (nu = %.1f)", x$cov, x$nu)
  }
  cat("\nCovariance:", cov_str, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate GP spatial specification against data
#'
#' @param gp tulpa_gp or tulpa_multiscale object
#' @param data Data frame
#'
#' @return Updated spatial object with computed neighbor structure
#' @keywords internal
validate_gp <- function(gp, data) {
  if (is.null(gp)) return(NULL)
  if (!inherits(gp, c("tulpa_gp", "tulpa_multiscale"))) return(gp)

  N <- nrow(data)

 # Check coordinate columns exist
  for (cv in gp$coord_vars) {
    if (!(cv %in% names(data))) {
      stop(sprintf("Coordinate variable '%s' not found in data", cv),
           call. = FALSE)
    }
  }

  # Extract coordinates
  coords <- cbind(
    data[[gp$coord_vars[1]]],
    data[[gp$coord_vars[2]]]
  )

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale coordinates if requested
  if (gp$scale_coords) {
    coords <- scale(coords)
  }

  # Detect unique coordinates (NNGP requires unique locations)
  coord_key <- paste(coords[, 1], coords[, 2], sep = ",")
  unique_keys <- unique(coord_key)
  obs_to_loc <- match(coord_key, unique_keys)
  unique_coords <- coords[match(unique_keys, coord_key), , drop = FALSE]
  n_unique <- nrow(unique_coords)

  if (n_unique < N) {
    message(sprintf("GP: %d unique locations from %d observations", n_unique, N))
  }

  gp$n_obs <- n_unique
  gp$n_spatial <- n_unique
  gp$n_unique <- n_unique
  gp$obs_to_loc <- as.integer(obs_to_loc)
  gp$unique_coords <- unique_coords
  gp$coords_matrix <- coords  # Keep full coords for reference

  # Compute neighbors on unique coordinates only
  # Store clamped nn values back so prepare_gp_for_hmc passes correct sizes to C++
  if (inherits(gp, "tulpa_gp")) {
    # Single-scale GP
    nn <- min(gp$nn, n_unique - 1)
    gp$nn <- nn
    gp$neighbor_info <- compute_nngp_neighbors(unique_coords, nn)

  } else if (inherits(gp, "tulpa_multiscale")) {
    # Multi-scale: separate neighbor structures for each scale
    nn_local <- min(gp$nn_local, n_unique - 1)
    nn_regional <- min(gp$nn_regional, n_unique - 1)
    gp$nn_local <- nn_local
    gp$nn_regional <- nn_regional

    gp$neighbor_info_local <- compute_nngp_neighbors(unique_coords, nn_local)
    gp$neighbor_info_regional <- compute_nngp_neighbors(unique_coords, nn_regional)
  }

  gp
}


#' Spatially-Varying Coefficients (SVC)
#'
#' @description
#' Specify spatially-varying coefficients for tulpa models. SVCs allow
#' regression coefficients to vary smoothly across space using a Gaussian
#' process prior. This captures local effects that may differ from global
#' relationships.
#'
#' Uses Nearest Neighbor Gaussian Process (NNGP) approximation for
#' computational efficiency with large datasets.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param terms Which coefficients should vary spatially. Options:
#'   - Integer vector: Column indices of design matrix (1 = intercept)
#'   - Character vector: Coefficient names (e.g., `"(Intercept)"`, `"depth"`)
#'   - Formula: `~ 1 + depth` for intercept and depth
#' @param cov Covariance function: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nn Number of nearest neighbors for NNGP approximation. Default 15.
#'   Larger values give better approximation but slower computation.
#' @param shared Logical; if TRUE (default), SVC effects enter both
#'   all processes. Set to FALSE for process-specific SVCs
#'   (triggers warning about potential confounding).
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing distances.
#' @param approx Approximation method: `"nngp"` (default) for Nearest Neighbor
#'   GP; `"hsgp"` for Hilbert Space GP (30-40x faster).
#' @param m Number of HSGP basis functions per dimension (default 6). Only
#'   used when `approx = "hsgp"`. Total basis functions will be m^2.
#' @param c_boundary Boundary factor for HSGP domain extension (default 1.5).
#'   Only used when `approx = "hsgp"`.
#'
#' @return A `tulpa_svc` object
#'
#' @details
#' The SVC model extends the linear predictor:
#'
#' \deqn{\eta(s) = X\beta + \tilde{X}(s)w(s)}
#'
#' where:
#' - \eqn{\beta} are global (non-spatial) coefficients
#' - \eqn{\tilde{X}(s)} is the subset of covariates with SVCs

#' - \eqn{w(s)} are spatially-varying adjustments at location s
#'
#' Each SVC follows an independent Gaussian process:
#' \deqn{w_j(s) \sim GP(0, \sigma^2_j \cdot C(\phi_j))}
#'
#' where \eqn{C(\phi)} is the correlation function with range parameter
#' \eqn{\phi}.
#'
#' **NNGP approximation**: For computational tractability, we use the
#' Nearest Neighbor Gaussian Process (Datta et al., 2016), which conditions
#' each location on its k nearest neighbors. This reduces complexity from
#' O(n^3) to O(n*k^2), enabling models with thousands of locations.
#'
#' **Interpretation**: A positive SVC for depth at location s means the
#' depth effect is stronger at s than the global average. The spatial
#' variance \eqn{\sigma^2_j} quantifies how much the effect varies across
#' space.
#'
#' @examples
#' # Create SVC specification
#' svc_spec <- spatial_svc(~ lon + lat, terms = 1)
#' print(svc_spec)
#'
#' \dontrun{
#' # Generate synthetic spatial data (not run - SVC not fully supported)
#' set.seed(202)
#' n <- 40
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n),
#'   count = rpois(n, 20),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Spatially-varying intercept (random spatial field)
#' fit <- tulpa(
#'   count | effort ~ depth,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   svc = spatial_svc(~ lon + lat, terms = 1),
#'   iter = 200, warmup = 100, chains = 1
#' )
#' summary(fit)
#'
#' # Extract and plot the spatially-varying coefficients
#' svc_effects <- svc(fit)
#' plot(svc_effects, "depth")
#' }
#'
#' @references
#' Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (
#' 2016). Hierarchical nearest-neighbor Gaussian process models for large
#' geostatistical datasets. Journal of the American Statistical Association,
#' 111(514), 800-812.
#'
#' @seealso [spatial_car()], [spatial_bym2()] for areal spatial effects,
#'   [svc()] for extracting SVC posteriors
#'
#' @export
spatial_svc <- function(coords,
                        terms = 1,
                        cov = c("exponential", "matern", "gaussian", "spherical"),
                        nn = 15,
                        shared = NULL,
                        scale_coords = TRUE,
                        approx = c("nngp", "hsgp"),
                        m = 6,
                        c_boundary = 1.5) {

  cov <- match.arg(cov)
  approx <- match.arg(approx)

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Parse terms specification
  if (inherits(terms, "formula")) {
    # Will be resolved against design matrix later
    terms_spec <- list(type = "formula", formula = terms)
  } else if (is.numeric(terms)) {
    terms_spec <- list(type = "index", indices = as.integer(terms))
  } else if (is.character(terms)) {
    terms_spec <- list(type = "names", names = terms)
  } else {
    stop("`terms` must be a formula, integer vector, or character vector",
         call. = FALSE)
  }

  # Validate nn (only relevant for NNGP)
  if (approx == "nngp") {
    if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
      stop("`nn` must be a positive integer", call. = FALSE)
    }
    nn <- as.integer(nn)
  }

  # Validate HSGP parameters
  if (approx == "hsgp") {
    if (!is.numeric(m) || length(m) != 1 || m < 3 || m > 50) {
      stop("`m` must be an integer between 3 and 50", call. = FALSE)
    }
    m <- as.integer(m)
    if (!is.numeric(c_boundary) || length(c_boundary) != 1 || c_boundary < 1.0) {
      stop("`c_boundary` must be a number >= 1.0", call. = FALSE)
    }
  }

  # Warning for non-shared SVCs
  if (isFALSE(shared)) {
    warning(
      "Non-shared SVCs (shared = FALSE) means effects are not shared across processes.\n",
      "Consider whether spatially-varying effects should be shared between\n",
      "processes if shared confounding structure is expected.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "svc",
      coord_vars = coord_vars,
      terms_spec = terms_spec,
      cov = cov,
      nn = nn,
      shared = shared,
      scale_coords = scale_coords,
      approx = approx,
      m = m,
      c_boundary = c_boundary,
      # Filled in during validation
      n_obs = NULL,
      n_svc = NULL,
      svc_indices = NULL,
      svc_names = NULL,
      coords_matrix = NULL,
      neighbor_info = NULL
    ),
    class = c("tulpa_svc", "tulpa_spatial", "list")
  )
}


#' Print method for tulpa_svc
#'
#' @param x A tulpa_svc object
#' @param ... Ignored
#'
#' @export
print.tulpa_svc <- function(x, ...) {
  cat("tulpa spatially-varying coefficients\n")
  cat("=====================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Covariance:", x$cov, "\n")
  cat("Neighbors (NNGP):", x$nn, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_svc)) {
    cat("\nSVC terms:", x$n_svc, "\n")
    if (!is.null(x$svc_names)) {
      cat("  ", paste(x$svc_names, collapse = ", "), "\n")
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

  if (!is.null(x$n_obs)) {
    cat("Observations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate SVC specification against data and design matrix
#'
#' @param svc tulpa_svc object
#' @param data Data frame
#' @param X Design matrix (to resolve term names)
#'
#' @return Updated tulpa_svc object with computed neighbor structure
#' @keywords internal
validate_svc <- function(svc, data, X) {
  if (is.null(svc)) return(NULL)
  if (!inherits(svc, "tulpa_svc")) return(svc)  # Other spatial types

  N <- nrow(data)
  p <- ncol(X)

  # Check coordinate columns exist
  for (cv in svc$coord_vars) {
    if (!(cv %in% names(data))) {
      stop(sprintf("Coordinate variable '%s' not found in data", cv),
           call. = FALSE)
    }
  }

  # Extract coordinates
  coords <- cbind(
    data[[svc$coord_vars[1]]],
    data[[svc$coord_vars[2]]]
  )

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale coordinates if requested
  if (svc$scale_coords) {
    coords <- scale(coords)
  }

  # Resolve SVC terms
  coef_names <- colnames(X)
  if (is.null(coef_names)) {
    coef_names <- paste0("V", seq_len(p))
  }

  if (svc$terms_spec$type == "index") {
    svc_indices <- svc$terms_spec$indices
    if (any(svc_indices < 1 | svc_indices > p)) {
      stop(sprintf("SVC term indices must be between 1 and %d", p),
           call. = FALSE)
    }
    svc_names <- coef_names[svc_indices]

  } else if (svc$terms_spec$type == "names") {
    svc_names <- svc$terms_spec$names
    svc_indices <- match(svc_names, coef_names)
    if (any(is.na(svc_indices))) {
      missing <- svc_names[is.na(svc_indices)]
      stop(sprintf("SVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }

  } else if (svc$terms_spec$type == "formula") {
    # Parse formula to get terms
    fmla <- svc$terms_spec$formula
    tt <- terms(fmla)
    term_labels <- attr(tt, "term.labels")
    has_intercept <- attr(tt, "intercept") == 1

    svc_names <- character(0)
    if (has_intercept) {
      svc_names <- c(svc_names, "(Intercept)")
    }
    svc_names <- c(svc_names, term_labels)

    svc_indices <- match(svc_names, coef_names)
    if (any(is.na(svc_indices))) {
      missing <- svc_names[is.na(svc_indices)]
      stop(sprintf("SVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }
  }

  # Compute nearest neighbors (NNGP only)
  approx <- svc$approx %||% "nngp"
  if (approx == "nngp") {
    nn <- min(svc$nn, N - 1)
    neighbor_info <- compute_nngp_neighbors(coords, nn)
  } else {
    neighbor_info <- NULL
  }

  # Update SVC object
  svc$n_obs <- N
  svc$n_svc <- length(svc_indices)
  svc$svc_indices <- svc_indices
  svc$svc_names <- svc_names
  svc$coords_matrix <- coords
  svc$neighbor_info <- neighbor_info

  # Set spatial parameters for parameter layout
  if (approx == "hsgp") {
    m <- svc$m %||% 6L
    svc$n_spatial <- length(svc_indices) * as.integer(m)^2  # m^2 basis per SVC term
  } else {
    svc$n_spatial <- svc$n_obs * svc$n_svc  # N effects per SVC term
  }

  svc
}


#' Compute nearest neighbors for NNGP
#'
#' @description
#' Compute the k nearest neighbors for each observation using Euclidean
#' distance. Returns in a format suitable for the NNGP likelihood.
#'
#' @param coords N x 2 matrix of coordinates
#' @param k Number of nearest neighbors
#'
#' @return List with:
#'   - `nn_idx`: N x k matrix of neighbor indices (0 for obs with fewer neighbors)
#'   - `nn_dist`: N x k matrix of distances to neighbors
#'   - `nn_order`: Ordering of observations for NNGP (by coordinate)
#'
#' @keywords internal
compute_nngp_neighbors <- function(coords, k) {
  N <- nrow(coords)

  # Order observations (improves NNGP conditioning)
  # Use maximum-minimum distance ordering for better numerical properties
  order_idx <- order(coords[, 1], coords[, 2])

  # Reorder coordinates
  coords_ordered <- coords[order_idx, , drop = FALSE]

  # Compute neighbors for each observation
  nn_idx <- matrix(0L, nrow = N, ncol = k)
  nn_dist <- matrix(Inf, nrow = N, ncol = k)

  # Phase 1.3: Precompute pairwise distances among neighbors
  nn_neighbor_dist <- array(0, dim = c(N, k, k))

  for (i in 2:N) {
    # Only consider previous observations (in ordering) as potential neighbors
    n_candidates <- min(i - 1, k)

    if (n_candidates > 0) {
      # Compute distances to all previous observations
      dists <- sqrt(
        (coords_ordered[1:(i-1), 1] - coords_ordered[i, 1])^2 +
        (coords_ordered[1:(i-1), 2] - coords_ordered[i, 2])^2
      )

      # Find k nearest
      if (length(dists) <= k) {
        nn_order <- order(dists)
        nn_idx[i, seq_len(length(dists))] <- nn_order
        nn_dist[i, seq_len(length(dists))] <- dists[nn_order]
      } else {
        nn_order <- order(dists)[1:k]
        nn_idx[i, ] <- nn_order
        nn_dist[i, ] <- dists[nn_order]
      }

      # Phase 1.3: Compute pairwise distances among neighbors
      n_neighbors <- sum(nn_idx[i, ] > 0)
      if (n_neighbors > 1) {
        neighbor_indices <- nn_idx[i, 1:n_neighbors]
        neighbor_coords <- coords_ordered[neighbor_indices, , drop = FALSE]
        for (j1 in 1:n_neighbors) {
          for (j2 in 1:n_neighbors) {
            if (j1 == j2) {
              nn_neighbor_dist[i, j1, j2] <- 0
            } else {
              nn_neighbor_dist[i, j1, j2] <- sqrt(
                (neighbor_coords[j1, 1] - neighbor_coords[j2, 1])^2 +
                (neighbor_coords[j1, 2] - neighbor_coords[j2, 2])^2
              )
            }
          }
        }
      }
    }
  }

  list(
    nn_idx = nn_idx,
    nn_dist = nn_dist,
    nn_neighbor_dist = nn_neighbor_dist,  # Phase 1.3: cached pairwise distances
    nn_order = order_idx,
    nn_order_inv = order(order_idx),  # Inverse permutation
    k = k
  )
}


#' Extract spatially-varying coefficients from a fitted model
#'
#' @description
#' Extract posterior distributions of spatially-varying coefficients (SVCs)
#' from a fitted tulpa model with SVC specification.
#'
#' @param object A `tulpa_fit` object fitted with `svc` argument
#' @param terms Which SVC terms to extract. If NULL (default), extracts all.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `tulpa_svc_posterior` object containing:
#' - `draws`: Array of posterior draws (draws x locations x terms)
#' - `coords`: Coordinate matrix
#' - `term_names`: Names of SVC terms
#'
#' @examples
#' \dontrun{
#' # Generate synthetic spatial data (not run - SVC not fully supported)
#' set.seed(303)
#' n <- 40
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   count = rpois(n, 20),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Fit model with SVC
#' fit <- tulpa(
#'   count | effort ~ depth,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   svc = spatial_svc(~ lon + lat, terms = c(1, 2)),
#'   iter = 200, warmup = 100, chains = 1
#' )
#'
#' # Extract SVC posteriors
#' svc_post <- svc(fit)
#' summary(svc_post)
#' }
#'
#' @seealso [spatial_svc()], [plot.tulpa_svc_posterior()]
#'
#' @export
svc <- function(object, terms = NULL, summary = FALSE,
                probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("svc")
}


#' @rdname svc
#' @export
svc.tulpa_fit <- function(object, terms = NULL, summary = FALSE,
                           probs = c(0.025, 0.5, 0.975), ...) {

  # Check if model has SVCs
  if (is.null(object$svc) || !inherits(object$svc, "tulpa_svc")) {
    stop("Model was not fitted with spatially-varying coefficients.\n",
         "Use `svc` argument in tulpa() to specify SVCs.", call. = FALSE)
  }

  svc_info <- object$svc
  n_obs <- svc_info$n_obs
  n_svc <- svc_info$n_svc
  svc_names <- svc_info$svc_names

  # Get SVC draws from model
  svc_draws <- object$.internal$svc_draws

  if (is.null(svc_draws)) {
    stop("SVC draws not found in model output", call. = FALSE)
  }

  # Subset terms if requested
  if (!is.null(terms)) {
    if (is.numeric(terms)) {
      term_idx <- terms
    } else if (is.character(terms)) {
      term_idx <- match(terms, svc_names)
      if (any(is.na(term_idx))) {
        stop("Terms not found: ", paste(terms[is.na(term_idx)], collapse = ", "),
             call. = FALSE)
      }
    } else {
      stop("`terms` must be numeric or character", call. = FALSE)
    }
    svc_draws <- svc_draws[, , term_idx, drop = FALSE]
    svc_names <- svc_names[term_idx]
    n_svc <- length(term_idx)
  }

  result <- structure(
    list(
      draws = svc_draws,
      coords = svc_info$coords_matrix,
      term_names = svc_names,
      n_obs = n_obs,
      n_svc = n_svc,
      n_draws = dim(svc_draws)[1],
      cov = svc_info$cov
    ),
    class = "tulpa_svc_posterior"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}


#' Summary method for tulpa_svc_posterior
#'
#' @param object A tulpa_svc_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @export
summary.tulpa_svc_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {

  n_obs <- object$n_obs
  n_svc <- object$n_svc
  draws <- object$draws

  results <- list()

  for (j in seq_len(n_svc)) {
    term_draws <- draws[, , j]

    summaries <- data.frame(
      obs = seq_len(n_obs),
      term = object$term_names[j],
      coord_1 = object$coords[, 1],
      coord_2 = object$coords[, 2],
      mean = colMeans(term_draws),
      sd = apply(term_draws, 2, sd),
      t(apply(term_draws, 2, quantile, probs = probs))
    )
    names(summaries)[7:ncol(summaries)] <- paste0("q", probs * 100)
    rownames(summaries) <- NULL

    results[[j]] <- summaries
  }

  result <- do.call(rbind, results)

  structure(
    result,
    n_draws = object$n_draws,
    term_names = object$term_names,
    class = c("tulpa_svc_summary", "data.frame")
  )
}


#' Print method for tulpa_svc_posterior
#'
#' @param x A tulpa_svc_posterior object
#' @param ... Ignored
#'
#' @export
print.tulpa_svc_posterior <- function(x, ...) {
  cat("Spatially-varying coefficient posterior\n")
  cat("=======================================\n\n")
  cat("Terms:", paste(x$term_names, collapse = ", "), "\n")
  cat("Locations:", x$n_obs, "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  cat("Covariance function:", x$cov, "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for spatial visualization\n")
  invisible(x)
}


#' Plot method for tulpa_svc_posterior
#'
#' @param x A tulpa_svc_posterior object
#' @param term Which term to plot (name or index). Default: first term.
#' @param type Plot type: "mean" (default), "sd", or quantile (e.g., "q50")
#' @param ... Additional arguments passed to plotting functions
#'
#' @export
plot.tulpa_svc_posterior <- function(x, term = 1, type = "mean", ...) {

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

  # Compute summary to plot
  if (type == "mean") {
    values <- colMeans(draws)
    title <- paste("SVC:", term_name, "(posterior mean)")
  } else if (type == "sd") {
    values <- apply(draws, 2, sd)
    title <- paste("SVC:", term_name, "(posterior SD)")
  } else if (grepl("^q[0-9]+", type)) {
    prob <- as.numeric(gsub("q", "", type)) / 100
    values <- apply(draws, 2, quantile, probs = prob)
    title <- paste0("SVC: ", term_name, " (", type, ")")
  } else {
    stop("Unknown plot type: ", type, call. = FALSE)
  }

  coords <- x$coords

  # Use ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      x = coords[, 1],
      y = coords[, 2],
      value = values
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, color = .data$value)) +
      ggplot2::geom_point(size = 2, ...) +
      ggplot2::scale_color_viridis_c() +
      ggplot2::labs(
        title = title,
        x = "Coordinate 1",
        y = "Coordinate 2",
        color = "Effect"
      ) +
      ggplot2::coord_fixed() +
      theme_tulpa()

    return(p)
  }

  # Base R fallback
  col_ramp <- colorRampPalette(c("blue", "white", "red"))
  n_colors <- 100
  colors <- col_ramp(n_colors)

  val_range <- range(values)
  val_scaled <- (values - val_range[1]) / diff(val_range)
  val_scaled[is.na(val_scaled)] <- 0.5
  point_colors <- colors[pmax(1, pmin(n_colors, ceiling(val_scaled * n_colors)))]

  plot(coords[, 1], coords[, 2],
       col = point_colors,
       pch = 19,
       xlab = "Coordinate 1",
       ylab = "Coordinate 2",
       main = title,
       asp = 1,
       ...)

  invisible(NULL)
}


#' Validate spatial specification against data
#'
#' @param spatial tulpa_spatial object
#' @param data Data frame
#'
#' @return NULL (invisibly); errors if validation fails
#' @keywords internal
validate_spatial <- function(spatial, data) {
  if (is.null(spatial)) return(invisible(NULL))

  # SVC validation is handled separately via validate_svc()
  if (inherits(spatial, "tulpa_svc")) {
    return(invisible(NULL))
  }

  # Check group variable exists
  if (spatial$level == "group") {
    if (!(spatial$group_var %in% names(data))) {
      stop(sprintf("Spatial group variable '%s' not found in data",
                   spatial$group_var), call. = FALSE)
    }

    # Check number of groups matches adjacency matrix
    n_groups <- length(unique(data[[spatial$group_var]]))
    if (n_groups != spatial$n_spatial) {
      stop(sprintf(
        "Number of groups in data (%d) does not match adjacency matrix (%d)",
        n_groups, spatial$n_spatial
      ), call. = FALSE)
    }
  } else {
    # Observation-level spatial
    if (nrow(data) != spatial$n_spatial) {
      stop(sprintf(
        "Number of observations (%d) does not match adjacency matrix (%d)",
        nrow(data), spatial$n_spatial
      ), call. = FALSE)
    }
  }

  # Check connectivity
  if (!is_connected(spatial$adjacency)) {
    warning(
      "Spatial adjacency graph is not fully connected.\n",
      "This may cause identifiability issues. Consider:\n",
      "  - Adding edges to connect isolated components\n",
      "  - Fitting separate models for each connected component",
      call. = FALSE
    )
  }

  invisible(NULL)
}


# =============================================================================
# Spatial Confounding Mitigation
# =============================================================================

#' Restricted Spatial Regression (RSR)
#'
#' @description
#' Apply Restricted Spatial Regression to mitigate spatial confounding.
#' RSR orthogonalizes the spatial effect to the covariate space, preventing
#' the spatial random effect from absorbing covariate information.
#'
#' This is important when covariates are spatially smooth (e.g., climate
#' variables, elevation) because the spatial random effect can "steal"
#' variance from these covariates, leading to biased coefficient estimates.
#'
#' @param spatial A spatial specification (`spatial_gp`, `spatial_car`, etc.)
#' @param restrict_to Formula specifying which covariates to orthogonalize
#'   against (e.g., `~ depth + temp`). The spatial effect will be constrained
#'   to be orthogonal to the column space of these covariates.
#'
#' @return A modified spatial specification with RSR enabled
#'
#' @details
#' The RSR approach (Reich et al., 2006; Hodges & Reich, 2010) modifies the
#' spatial random effect to be orthogonal to the fixed effect design matrix:
#'
#' \deqn{w_{RSR} = (I - P_X) w}
#'
#' where \eqn{P_X = X(X'X)^{-1}X'} is the projection matrix onto the column
#' space of X.
#'
#' **When to use RSR:**
#' - Covariates are spatially smooth (environmental gradients)
#' - Interested in causal interpretation of covariate effects
#' - Coefficients appear attenuated toward zero
#'
#' **When NOT to use RSR:**
#' - Covariates are spatially uncorrelated
#' - Spatial effect is the primary quantity of interest
#' - Prediction is the main goal (not causal inference)
#'
#' @examples
#' # Create RSR spatial structure
#' rsr <- spatial_rsr(
#'   spatial_gp(~ lon + lat),
#'   restrict_to = ~ depth + temp
#' )
#' print(rsr)
#'
#' \donttest{
#' # Generate synthetic spatial data with confounding
#' set.seed(404)
#' n <- 50
#' lon <- runif(n, 0, 10)
#' lat <- runif(n, 0, 10)
#' # Make depth and temp spatially correlated
#' df <- data.frame(
#'   lon = lon,
#'   lat = lat,
#'   depth = lon/5 + rnorm(n, 0, 0.5),  # Correlated with lon
#'   temp = lat/5 + rnorm(n, 0, 0.5),   # Correlated with lat
#'   count = rpois(n, 25),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Standard GP (may have spatial confounding)
#' fit1 <- tulpa(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   spatial = spatial_gp(~ lon + lat),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # RSR to protect depth and temp coefficients
#' fit2 <- tulpa(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   spatial = spatial_rsr(
#'     spatial_gp(~ lon + lat),
#'     restrict_to = ~ depth + temp
#'   ),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Compare coefficient estimates
#' summary(fit1)  # May be attenuated
#' summary(fit2)  # Protected from spatial confounding
#' }
#'
#' @references
#' Reich, B. J., Hodges, J. S., & Zadnik, V. (2006). Effects of residual
#' smoothing on the posterior of the fixed effects in disease-mapping models.
#' Biometrics, 62(4), 1197-1206.
#'
#' Hodges, J. S., & Reich, B. J. (2010). Adding spatially-correlated errors
#' can mess up the fixed effect you love. The American Statistician, 64(4),
#' 325-334.
#'
#' @seealso [spatial_gp()], [spatial_car()]
#'
#' @export
spatial_rsr <- function(spatial, restrict_to) {

  if (!inherits(spatial, "tulpa_spatial")) {
    stop("`spatial` must be a tulpa spatial specification", call. = FALSE)
  }

  if (!inherits(restrict_to, "formula")) {
    stop("`restrict_to` must be a formula", call. = FALSE)
  }

  # Store RSR information in the spatial object
  spatial$rsr <- TRUE
  spatial$rsr_formula <- restrict_to

  # Add RSR class for dispatch
  class(spatial) <- c("tulpa_rsr", class(spatial))

  spatial
}


#' Print method for tulpa_rsr
#'
#' @param x A tulpa_rsr object
#' @param ... Passed to underlying print method
#'
#' @export
print.tulpa_rsr <- function(x, ...) {
  # Print underlying spatial type
  NextMethod()

  cat("\nRestricted Spatial Regression (RSR):\n")
  cat("  Orthogonal to:", deparse(x$rsr_formula), "\n")
  cat("  (Spatial effect constrained to be orthogonal to covariate space)\n")

  invisible(x)
}


#' Compute RSR projection matrix
#'
#' @description
#' Compute the orthogonal projection matrix P_perp = I - P_X that projects
#' the spatial effect into the space orthogonal to the covariates.
#'
#' @param X Design matrix of covariates to orthogonalize against
#'
#' @return Projection matrix (n x n)
#' @keywords internal
compute_rsr_projection <- function(X) {
  n <- nrow(X)
  p <- ncol(X)

  if (p >= n) {
    warning("More covariates than observations; RSR may not be effective",
            call. = FALSE)
  }

  # QR decomposition is more numerically stable than direct inverse
  qr_X <- qr(X)

  # P_X = Q %*% Q' where Q is orthonormal basis for col(X)
  Q <- qr.Q(qr_X)

  # P_perp = I - Q %*% Q'
  P_perp <- diag(n) - Q %*% t(Q)

  P_perp
}


#' Validate RSR specification
#'
#' @param spatial tulpa_rsr object
#' @param data Data frame
#' @param formula Model formula (to extract design matrix)
#'
#' @return Updated spatial object with projection matrix
#' @keywords internal
validate_rsr <- function(spatial, data, formula) {
  if (is.null(spatial) || !inherits(spatial, "tulpa_rsr")) {
    return(spatial)
  }

  # Build design matrix for RSR covariates
  rsr_formula <- spatial$rsr_formula

  # Check if terms exist in data
  rsr_vars <- all.vars(rsr_formula)
  missing_vars <- setdiff(rsr_vars, names(data))
  if (length(missing_vars) > 0) {
    stop(sprintf("RSR variables not found in data: %s",
                 paste(missing_vars, collapse = ", ")), call. = FALSE)
  }

  # Build design matrix
  X_rsr <- model.matrix(rsr_formula, data = data)

  # Compute projection matrix
  spatial$rsr_projection <- compute_rsr_projection(X_rsr)
  spatial$rsr_vars <- rsr_vars

  spatial
}


#' Apply RSR projection to spatial effect
#'
#' @description
#' Project spatial effect into the space orthogonal to covariates.
#' Called during posterior computation.
#'
#' @param w Spatial effect vector (length n)
#' @param P_perp Projection matrix from compute_rsr_projection
#'
#' @return Projected spatial effect (length n)
#' @keywords internal
apply_rsr_projection <- function(w, P_perp) {
  as.vector(P_perp %*% w)
}


# =====================================================================
# SPDE spatial field (Matérn via FEM on triangular mesh)
# =====================================================================

#' SPDE Spatial Field (Matérn via Triangular Mesh)
#'
#' Specify a continuous Matérn spatial field using the SPDE approach
#' (Lindgren, Rue & Lindström 2011). Builds a triangular mesh from
#' observation coordinates, computes FEM matrices, and passes them
#' to tulpa's SPDE Laplace engine with CHOLMOD sparse solver.
#'
#' @param coords A formula `~ x + y` or a two-column matrix of coordinates.
#' @param data Optional data.frame for formula evaluation.
#' @param mesh A pre-built `tulpa_mesh` object. If NULL (default), a mesh
#'   is built automatically from `coords`.
#' @param boundary Optional boundary: a two-column matrix of polygon vertices,
#'   an sf polygon, or NULL for convex hull with extension.
#' @param max_edge Maximum edge length for mesh refinement. A single value
#'   or `c(inner, outer)`.
#' @param cutoff Minimum distance between mesh vertices. Default 0.
#' @param nu Matérn smoothness parameter. Default 1 (exponential covariance).
#'   Currently only integer values supported; fractional smoothness via
#'   rational SPDE is planned.
#' @param prior_range Prior for the spatial range. A numeric vector `c(U, alpha)`
#'   where P(range < U) = alpha. Default `c(0.5, 0.5)`.
#' @param prior_sigma Prior for the marginal standard deviation. A numeric vector
#'   `c(U, alpha)` where P(sigma > U) = alpha. Default `c(1, 0.5)`.
#'
#' @return A `tulpa_spatial` object with type `"spde"`.
#'
#' @export
#'
#' @examples
#' set.seed(42)
#' coords <- cbind(runif(50), runif(50))
#' spec <- spatial_spde(coords)
#' print(spec)
spatial_spde <- function(coords, data = NULL, mesh = NULL,
                         boundary = NULL, max_edge = NULL, cutoff = 0,
                         nu = 1,
                         prior_range = c(0.5, 0.5),
                         prior_sigma = c(1, 0.5)) {

  # Resolve coordinates
  if (inherits(coords, "formula")) {
    if (is.null(data)) stop("data must be provided when coords is a formula", call. = FALSE)
    vars <- all.vars(coords)
    if (length(vars) != 2) stop("formula must have exactly 2 variables", call. = FALSE)
    obs_coords <- cbind(data[[vars[1]]], data[[vars[2]]])
    coord_formula <- coords
  } else {
    obs_coords <- as.matrix(coords)
    coord_formula <- NULL
  }

  if (ncol(obs_coords) != 2) stop("coords must have 2 columns", call. = FALSE)

  # Build mesh if not provided
  if (is.null(mesh)) {
    mesh_args <- list(coords = obs_coords, cutoff = cutoff)
    if (!is.null(boundary)) mesh_args$boundary <- boundary
    if (!is.null(max_edge)) mesh_args$max_edge <- max_edge
    mesh <- do.call(tulpaMesh::tulpa_mesh, mesh_args)
  }

  if (!inherits(mesh, "tulpa_mesh")) {
    stop("mesh must be a tulpa_mesh object", call. = FALSE)
  }

  # Compute FEM matrices
  fem <- tulpaMesh::fem_matrices(mesh, obs_coords = obs_coords)

  # Extract CSC components for C++ interface
  A_csc <- as(fem$A, "CsparseMatrix")
  G1_csc <- as(fem$G, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$C)

  structure(
    list(
      type = "spde",
      mesh = mesh,
      obs_coords = obs_coords,
      coord_formula = coord_formula,
      n_mesh = fem$n_mesh,
      nu = nu,
      prior_range = prior_range,
      prior_sigma = prior_sigma,
      # FEM matrices (sparse)
      C = fem$C,
      G = fem$G,
      A = fem$A,
      # Pre-extracted CSC slots for C++ (avoids repeated extraction at fit time)
      A_x = A_csc@x, A_i = A_csc@i, A_p = A_csc@p,
      C0_diag = C0_diag,
      G1_x = G1_csc@x, G1_i = G1_csc@i, G1_p = G1_csc@p
    ),
    class = c("tulpa_spatial", "list")
  )
}

#' SPDE Spatial Field from Custom Matrices
#'
#' Specify a continuous Matérn spatial field using externally-provided FEM
#' matrices. Use this with meshes from fmesher, rSPDE, or any other source.
#'
#' @param C Mass matrix (n_mesh x n_mesh sparse matrix, e.g. from `fmesher::fm_fem()$c0`).
#' @param G Stiffness matrix (n_mesh x n_mesh sparse matrix, e.g. from `fmesher::fm_fem()$g1`).
#' @param A Projection matrix (n_obs x n_mesh sparse matrix, e.g. from `fmesher::fm_basis()`).
#' @param nu Matérn smoothness parameter. Default 1.
#' @param prior_range Prior for the spatial range. Default `c(0.5, 0.5)`.
#' @param prior_sigma Prior for the marginal standard deviation. Default `c(1, 0.5)`.
#'
#' @return A `tulpa_spatial` object with type `"spde"`.
#'
#' @export
spatial_spde_custom <- function(C, G, A, nu = 1,
                                prior_range = c(0.5, 0.5),
                                prior_sigma = c(1, 0.5)) {
  if (!inherits(C, "Matrix")) C <- as(C, "CsparseMatrix")
  if (!inherits(G, "Matrix")) G <- as(G, "CsparseMatrix")
  if (!inherits(A, "Matrix")) A <- as(A, "CsparseMatrix")

  n_mesh <- nrow(C)
  if (ncol(C) != n_mesh) stop("C must be square", call. = FALSE)
  if (nrow(G) != n_mesh || ncol(G) != n_mesh) stop("G must have same dimensions as C", call. = FALSE)
  if (ncol(A) != n_mesh) stop("A must have n_mesh columns", call. = FALSE)

  A_csc <- as(A, "CsparseMatrix")
  G1_csc <- as(G, "CsparseMatrix")
  C0_diag <- Matrix::diag(C)

  structure(
    list(
      type = "spde",
      mesh = NULL,
      obs_coords = NULL,
      coord_formula = NULL,
      n_mesh = n_mesh,
      nu = nu,
      prior_range = prior_range,
      prior_sigma = prior_sigma,
      C = C, G = G, A = A,
      A_x = A_csc@x, A_i = A_csc@i, A_p = A_csc@p,
      C0_diag = C0_diag,
      G1_x = G1_csc@x, G1_i = G1_csc@i, G1_p = G1_csc@p
    ),
    class = c("tulpa_spatial", "list")
  )
}

#' @export
print.tulpa_spatial <- function(x, ...) {
  if (!is.null(x$type) && x$type == "spde") {
    cat("tulpa_spatial: SPDE (Matérn, nu =", x$nu, ")\n")
    cat("  Mesh nodes:", x$n_mesh, "\n")
    if (!is.null(x$mesh)) {
      cat("  Triangles: ", x$mesh$n_triangles, "\n")
    }
    if (!is.null(x$obs_coords)) {
      cat("  Observations:", nrow(x$obs_coords), "\n")
    }
    cat("  Prior range: P(range <", x$prior_range[1], ") =", x$prior_range[2], "\n")
    cat("  Prior sigma: P(sigma >", x$prior_sigma[1], ") =", x$prior_sigma[2], "\n")
    invisible(x)
  } else {
    NextMethod()
  }
}
