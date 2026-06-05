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

    # Resolve the group values to 1-based adjacency-row indices. The
    # resolver allows empty cells (cells with no observed data) when the
    # mapping is unambiguous (integer 1-based indices, or character /
    # factor labels matching `rownames(adjacency)`). It errors with a
    # clear message when cells can't be identified -- e.g. an unrowed
    # factor whose level count differs from the adjacency size.
    .resolve_spatial_idx(
      values = data[[spatial$group_var]],
      n_spatial_units = spatial$n_spatial,
      adjacency = spatial$adjacency,
      group_var = spatial$group_var
    )
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
#' \dontrun{
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
  .orthogonal_complement_projection(X, "RSR")
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
# SPDE spatial field (Matern via FEM on triangular mesh)
# =====================================================================

#' SPDE Spatial Field (Matern via Triangular Mesh)
#'
#' Specify a continuous Matern spatial field using the SPDE approach
#' (Lindgren, Rue & Lindstrom 2011). Builds a triangular mesh from
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
#' @param nu Matern smoothness parameter. Default 1 (exponential covariance).
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
#' Specify a continuous Matern spatial field using externally-provided FEM
#' matrices. Use this with meshes from fmesher, rSPDE, or any other source.
#'
#' @param C Mass matrix (n_mesh x n_mesh sparse matrix, e.g. from `fmesher::fm_fem()$c0`).
#' @param G Stiffness matrix (n_mesh x n_mesh sparse matrix, e.g. from `fmesher::fm_fem()$g1`).
#' @param A Projection matrix (n_obs x n_mesh sparse matrix, e.g. from `fmesher::fm_basis()`).
#' @param nu Matern smoothness parameter. Default 1.
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
    invisible(x)
  } else {
    NextMethod()
  }
}
