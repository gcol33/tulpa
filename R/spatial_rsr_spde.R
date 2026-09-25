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
#' @param spatial An areal specification -- `spatial_car()`, `spatial_icar()`,
#'   `spatial_bym2()` or a proper-CAR spec -- or an NNGP one, `spatial_gp()`.
#'   The projection is applied by the binomial Polya-Gamma Gibbs sampler, which
#'   carries the field's own prior precision as an adjacency or as Vecchia
#'   factors; an HSGP basis (`spatial_gp(approx = "hsgp")`) and an SPDE mesh
#'   (`spatial_spde()`) are neither, and are refused at construction rather
#'   than accepted and then unfittable.
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
#' space of X. The projector is built at the FIELD's own resolution: one row
#' per areal unit, or one per unique location for an NNGP field, with the
#' restricted design averaged over the observations at each.
#'
#' **What RSR estimates.** The restriction puts no part of the shared, smooth
#' signal in the field, so the fixed effect takes all of it: RSR targets the
#' MARGINAL association between the covariate and the response, where the
#' unrestricted spatial model targets the association conditional on the
#' field. The two differ by exactly the covariate's projection onto the field,
#' so they are different estimands rather than a biased and an unbiased
#' version of one (Bradley, 2024). Measured on a confounded continuous
#' fixture (5 seeds, conditional slope 1.0, marginal 1.71): the restricted fit
#' averaged 0.06 from the marginal value and 0.74 from the conditional one,
#' the unrestricted fit 0.10 and 0.63.
#'
#' **The restriction is not free.** In the geostatistical setting Hanks et al.
#' (2015) measured POORER coverage under RSR than under the spatial model that
#' does not restrict, and credible intervals that can be inappropriately
#' narrow under model misspecification; Khan and Calder (2022) report the same
#' on areal structure, with a non-spatial model competitive on coverage and
#' higher Type-S error rates under RSR. Read an RSR interval as an interval
#' for the marginal association under a correctly specified model, and prefer
#' a posterior-predictive check (Hanks et al., 2015) where that is in doubt.
#'
#' **When to use RSR:**
#' - Covariates are spatially smooth (environmental gradients)
#' - The marginal association is the quantity of interest
#' - Coefficients appear attenuated toward zero
#'
#' **When NOT to use RSR:**
#' - Covariates are spatially uncorrelated
#' - Spatial effect is the primary quantity of interest
#' - Prediction is the main goal
#' - Interval coverage matters more than the point estimate
#'
#' RSR fits are binomial, through `mode = "gibbs"` (which `mode = "auto"`
#' selects for it).
#'
#' @examples
#' # Create RSR spatial structure on an areal field
#' W <- matrix(0, 4, 4)
#' for (i in 1:3) W[i, i + 1] <- W[i + 1, i] <- 1
#' rsr <- spatial_rsr(
#'   spatial_car(W, level = "obs"),
#'   restrict_to = ~ depth + temp
#' )
#' print(rsr)
#'
#' \donttest{
#' # Areal binomial data on a chain of regions, covariate spatially confounded
#' set.seed(404)
#' n_regions <- 12
#' W <- matrix(0, n_regions, n_regions)
#' for (i in 1:(n_regions - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
#' df <- data.frame(region = factor(rep(1:n_regions, each = 5)))
#' df$x <- as.integer(df$region) / 4 + rnorm(nrow(df), 0, 0.5)
#' df$y <- rbinom(nrow(df), 20, plogis(-0.5 + 0.6 * df$x))
#'
#' # RSR orthogonalises the spatial field to x, protecting its coefficient
#' fit <- tulpa(
#'   y ~ x + spatial(region),
#'   data = df,
#'   family = "binomial",
#'   n_trials = rep(20L, nrow(df)),
#'   spatial = spatial_rsr(spatial_car(W, level = "obs"), restrict_to = ~ x),
#'   mode = "auto",
#'   control = list(n_iter = 500L, warmup = 250L)
#' )
#' summary(fit)
#'
#' # The same modifier on a continuous field: one observation per location,
#' # the projector built at the unique coordinates.
#' set.seed(7)
#' n <- 80
#' pts <- data.frame(lon = runif(n), lat = runif(n))
#' pts$x <- as.numeric(scale(pts$lon + pts$lat + rnorm(n, 0, 0.3)))
#' pts$y <- rbinom(n, 25, plogis(-0.2 + pts$x))
#' fit_gp <- tulpa(
#'   y ~ x,
#'   data = pts,
#'   family = "binomial",
#'   n_trials = rep(25L, n),
#'   spatial = spatial_rsr(spatial_gp(~ lon + lat), restrict_to = ~ x),
#'   mode = "gibbs",
#'   control = list(n_iter = 500L, warmup = 250L)
#' )
#' summary(fit_gp)
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
#' Hanks, E. M., Schliep, E. M., Hooten, M. B., & Hoeting, J. A. (2015).
#' Restricted spatial regression in practice: geostatistical models,
#' confounding, and robustness under model misspecification. Environmetrics,
#' 26(4), 243-254.
#'
#' Khan, K., & Calder, C. A. (2022). Restricted spatial regression methods:
#' implications for inference. Journal of the American Statistical
#' Association, 117(537), 482-494.
#'
#' Bradley, J. R. (2024). Restricted spatial regression is reasonable
#' statistical practice: clarifications, interpretations, and new
#' developments. arXiv:2408.05106.
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

  # Two kernels apply the projection: `cpp_pg_binomial_gibbs_rsr()` on an areal
  # neighbour list and `cpp_pg_binomial_gibbs_gp_rsr()` on an NNGP field
  # (gcol33/tulpa#848). A field shape neither of them carries is refused here,
  # where the argument that caused it is still in hand, rather than downstream
  # by a message about a backend the user never chose (gcol33/tulpa#815).
  sp_type <- tolower(spatial$type %||% "")
  if (!sp_type %in% .RSR_FIELDS) {
    stop(sprintf(paste0(
      "spatial_rsr() restricts an areal or NNGP field (%s); got '%s'.\n",
      "The projection is applied by the binomial Polya-Gamma Gibbs sampler, ",
      "which carries an areal neighbour list or an NNGP field and neither an ",
      "HSGP basis nor an SPDE mesh. Build the RSR field on spatial_car() / ",
      "spatial_icar() / spatial_bym2() / spatial_gp(), or drop spatial_rsr() ",
      "and fit the field directly."),
      paste(.RSR_FIELDS, collapse = ", "), spatial$type %||% "<none>"),
      call. = FALSE)
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
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the spatial specification and its restricted-spatial-regression
#'   modifier to the console.
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
#' @param nu Matern smoothness parameter. A positive number. Integer `nu`
#'   (1, 2, 3, ...) gives an exact FEM construction (operator order
#'   `alpha = nu + 1`). Fractional `nu` (e.g. 0.5, 1.5) uses the operator-based
#'   rational SPDE approximation with BRASIL best-rational coefficients
#'   (Bolin & Kirchner 2020; Hofreither 2021); supported by the Laplace fitter
#'   `fit_spde()` (single-point and nested over range/sigma). NUTS and analytic
#' marginal SEs remain integer-only. Default 1.
#' @param prior_range PC prior on the spatial range, `c(U, alpha)` with
#'   P(range < U) = alpha. `NULL` (the default) anchors it on the data: `U` is
#'   a fifth of the diagonal of the coordinates' bounding box and `alpha = 0.5`,
#'   so `U` is the prior median.
#' @param prior_sigma PC prior on the marginal standard deviation, `c(U, alpha)`
#'   with P(sigma > U) = alpha. `NULL` (the default) is `c(3, 0.01)`, the
#'   engine's prior on every field scale.
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
                         prior_range = NULL,
                         prior_sigma = NULL) {

  .validate_spde_nu(nu)

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
  .check_coords_finite(obs_coords, "spatial_spde()")
  .check_spde_prior_args(prior_range, prior_sigma, "spatial_spde()")
  prior_stated <- c(range = !is.null(prior_range), sigma = !is.null(prior_sigma))
  prior_range <- prior_range %||% .nl_default_range_prior(obs_coords)
  prior_sigma <- prior_sigma %||% .nl_scale_anchor()
  if (is.null(prior_range)) {
    stop("spatial_spde(): the coordinates span a single point, so no default ",
         "range prior can be anchored on them; pass `prior_range = c(U, alpha)`.",
         call. = FALSE)
  }

  # Build mesh if not provided. The mesh is built on the DISTINCT sites: a
  # repeated-measures design puts several observations at one location, and a
  # triangulation cannot take a vertex twice (it stopped with the vendored
  # CDT's "Duplicate vertex detected"; gcol33/tulpa#909). Every observation,
  # repeats included, still gets its own row of the projector A below.
  if (is.null(mesh)) {
    mesh_args <- list(coords = unique(obs_coords), cutoff = cutoff)
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
      prior_stated = prior_stated,
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
#' @param nu Matern smoothness parameter. A positive number; integer values
#'   give the exact FEM construction, fractional values the BRASIL rational SPDE
#' approximation (supported by `fit_spde()`;). Default 1.
#' @param prior_range PC prior on the spatial range, `c(U, alpha)` with
#'   P(range < U) = alpha. `NULL` (the default) anchors it on `coords` as
#'   [spatial_spde()] does, and needs them.
#' @param prior_sigma PC prior on the marginal standard deviation, `c(U, alpha)`
#'   with P(sigma > U) = alpha. `NULL` (the default) is `c(3, 0.01)`.
#' @param coords Observation coordinates (an `n_obs x 2` matrix), read only to
#'   anchor the default range prior. Not needed when `prior_range` is given.
#'
#' @return A `tulpa_spatial` object with type `"spde"`.
#'
#' @examples
#' # FEM matrices of a 6-node 1-D chain mesh, observed at the nodes:
#' n <- 6L
#' C <- Matrix::Diagonal(n)
#' G <- Matrix::bandSparse(n, k = c(-1, 0, 1),
#'                         diagonals = list(rep(-1, n - 1), c(1, rep(2, n - 2), 1),
#'                                          rep(-1, n - 1)))
#' A <- Matrix::Diagonal(n)
#' sp <- spatial_spde_custom(C, G, A, prior_range = c(1, 0.5),
#'                           prior_sigma = c(1, 0.01))
#' sp$type
#' @export
spatial_spde_custom <- function(C, G, A, nu = 1,
                                prior_range = NULL,
                                prior_sigma = NULL,
                                coords = NULL) {
  .validate_spde_nu(nu)
  .check_spde_prior_args(prior_range, prior_sigma, "spatial_spde_custom()")
  prior_stated <- c(range = !is.null(prior_range), sigma = !is.null(prior_sigma))
  if (is.null(prior_range)) {
    if (!is.null(coords)) prior_range <- .nl_default_range_prior(coords)
    if (is.null(prior_range)) {
      stop("spatial_spde_custom() needs `prior_range = c(U, alpha)`, or the ",
           "`coords` its default is anchored on: the FEM matrices carry no ",
           "coordinate scale.", call. = FALSE)
    }
  }
  prior_sigma <- prior_sigma %||% .nl_scale_anchor()
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
      prior_stated = prior_stated,
      C = C, G = G, A = A,
      A_x = A_csc@x, A_i = A_csc@i, A_p = A_csc@p,
      C0_diag = C0_diag,
      G1_x = G1_csc@x, G1_i = G1_csc@i, G1_p = G1_csc@p
    ),
    class = c("tulpa_spatial", "list")
  )
}

# A stated SPDE anchor pair is checked where the user set it, so a range or
# scale anchor the PC calibration cannot represent is refused naming the
# argument, not at fit time as an unnamed anchor or a failed grid construction
# (gcol33/tulpa#894). NULL takes the data-anchored default and is not checked.
.check_spde_prior_args <- function(prior_range, prior_sigma, where) {
  if (!is.null(prior_range)) {
    .check_pc_anchor_pair(prior_range, "prior_range", where,
                          param = "range", lower = TRUE)
  }
  if (!is.null(prior_sigma)) {
    .check_pc_anchor_pair(prior_sigma, "prior_sigma", where)
  }
  invisible(TRUE)
}

# print.tulpa_spatial (including the SPDE branch) is defined once in
# spatial_car.R so the areal ICAR/CAR/BYM2 formatter is not shadowed.
