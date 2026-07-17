#' Gaussian process spatial structure (NNGP)
#'
#' @description
#' Specify a Gaussian-process spatial random effect, approximated with a
#' nearest-neighbour GP (NNGP) for scalability. Captures smooth spatial
#' variation from point-referenced coordinates.
#'
#' @param coords A formula (`~ lon + lat`) or character vector of length 2
#'   naming the two coordinate variables in the data.
#' @param approx GP approximation: `"nngp"` (default, a nearest-neighbour GP with
#'   the `cov` / `nu` / `nn` / `solver` arguments) or `"hsgp"` (a Hilbert-space
#'   basis GP with `m` functions per dimension and boundary factor `c`).
#' @param cov Covariance function (NNGP only). One of `"exponential"`,
#'   `"matern"`, `"gaussian"`, or `"spherical"`.
#' @param nu Matern smoothness parameter. Used only when `cov = "matern"`.
#' @param nn Number of nearest neighbours used in the NNGP approximation.
#' @param m Number of HSGP basis functions per dimension (`approx = "hsgp"`).
#' @param c HSGP boundary factor, `>= 1` (`approx = "hsgp"`).
#' @param solver Linear solver for the GP. One of `"auto"`, `"cholesky"`,
#'   `"cg"`, `"pcg"`, or `"gpu"`. `"gpu"` falls back to `"pcg"` when CUDA
#'   support is unavailable.
#' @param cg_tol Convergence tolerance for the (preconditioned) CG solver.
#' @param cg_maxiter Maximum number of (preconditioned) CG iterations.
#' @param shared Whether the spatial effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares the effect; `FALSE` fits
#'   process-specific effects and emits a warning.
#' @param scale_coords Logical. Standardize coordinates before fitting
#'   (default `TRUE`).
#' @param parameterization Latent parameterization. One of `"centered"`,
#'   `"noncentered"`, or `"collapsed"` (the last is deprecated).
#'
#' @return A `tulpa_gp` object (also of class `tulpa_spatial`).
#'
#' @seealso [spatial_car()], [spatial_bym2()] for areal spatial effects.
#'
#' @examples
#' # GP spatial specification from coordinate columns
#' spatial_gp(~ lon + lat)
#' spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
#'
#' @export
spatial_gp <- function(coords,
                       approx = c("nngp", "hsgp"),
                       cov = c("exponential", "matern", "gaussian", "spherical"),
                       nu = 1.5,
                       nn = 15,
                       m = 6,
                       c = 1.5,
                       solver = c("auto", "cholesky", "cg", "pcg", "gpu"),
                       cg_tol = 1e-6,
                       cg_maxiter = 100,
                       shared = NULL,
                       scale_coords = TRUE,
                       parameterization = c("centered", "noncentered", "collapsed")) {

  approx <- match.arg(approx)
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

if (solver == "gpu" && !cpp_gpu_available()) {
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

  # Hilbert-space approximation: a basis-function GP with `m` functions per
  # dimension and boundary factor `c` (the NNGP cov/nu/nn/solver args do not
  # apply). Returns a tulpa_hsgp spec, formerly spatial_hsgp().
  if (approx == "hsgp") {
    if (!is.numeric(m) || length(m) != 1 || m < 3 || m > 50) {
      stop("`m` must be an integer between 3 and 50", call. = FALSE)
    }
    m <- as.integer(m)
    if (!is.numeric(c) || length(c) != 1 || c < 1) {
      stop("`c` (boundary factor) must be >= 1", call. = FALSE)
    }
    if (isFALSE(shared)) .warn_nonshared("spatial effects")
    return(structure(
      list(type = "hsgp", coord_vars = coord_vars, m = m, c = c,
           shared = shared, scale_coords = scale_coords,
           n_obs = NULL, coords_matrix = NULL),
      class = c("tulpa_hsgp", "tulpa_spatial", "list")))
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

  if (isFALSE(shared)) .warn_nonshared("spatial effects")

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
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the Gaussian-process spatial specification to the console.
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

  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}




#' Print method for tulpa_hsgp
#'
#' @param x A tulpa_hsgp object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the Hilbert-space GP spatial specification to the console.
#'
#' @export
print.tulpa_hsgp <- function(x, ...) {
  cat("tulpa Hilbert Space GP (HSGP) spatial specification\n")
  cat("=====================================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Basis functions:", x$m, "per dim (", x$m^2, "total )\n")
  cat("Boundary factor:", x$c, "\n")
  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

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
  coords <- prepare_coords(spatial$coord_vars, data, spatial$scale_coords)
  spatial$n_obs <- nrow(coords)
  spatial$coords_matrix <- coords
  spatial
}


#' Validate HSGP-MSGP (multi-scale GP with HSGP approximation)
#' @noRd
validate_hsgp_multiscale <- function(spatial, data) {
  coords <- prepare_coords(spatial$coord_vars, data, spatial$scale_coords)
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
#'   family = tulpaRatio::tulpa_poisson_gamma(),
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

  if (isFALSE(shared)) .warn_nonshared("multi-scale spatial effects")

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
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the multi-scale spatial specification to the console.
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
  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

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
#'   family = tulpaRatio::tulpa_poisson_gamma(),
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
