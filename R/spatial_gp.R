#' Gaussian process spatial structure (NNGP)
#'
#' @description
#' Specify a Gaussian-process spatial random effect, approximated with a
#' nearest-neighbour GP (NNGP) for scalability. Captures smooth spatial
#' variation from point-referenced coordinates.
#'
#' @param coords A formula (`~ lon + lat`) or character vector naming the
#'   coordinate variables in the data. With `approx = "nngp"` the coordinate
#'   DIMENSION is however many are named: two for a map, one for a transect or
#'   a depth profile, three for a depth-resolved domain. The neighbour graph
#'   and the neighbour covariance both read every column. `approx = "hsgp"`
#'   takes exactly two, and so does any
#'   sampler mode, because both store coordinates at a fixed 2-D stride.
#' @param approx GP approximation: `"nngp"` (default, a nearest-neighbour GP with
#'   the `cov` / `nu` / `nn` arguments) or `"hsgp"` (a Hilbert-space
#'   basis GP with `m` functions per dimension and boundary factor `c`).
#' @param cov Covariance function (NNGP only). One of `"exponential"` or
#'   `"matern"`.
#' @param nu Matern smoothness parameter, one of `1.5` or `2.5`. Used only when
#'   `cov = "matern"` (`nu = 0.5` is `cov = "exponential"`).
#' @param nn Number of nearest neighbours used in the NNGP approximation.
#' @param m Number of HSGP basis functions per dimension (`approx = "hsgp"`).
#' @param c HSGP boundary factor, `>= 1` (`approx = "hsgp"`).
#' @param shared Whether the spatial effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares the effect; `FALSE` fits
#'   process-specific effects and emits a warning.
#' @param scale_coords Logical. Standardize coordinates before fitting
#'   (default `TRUE`).
#' @param parameterization Latent parameterization for the exact-NUTS field.
#'   One of `"noncentered"` (default; samples `z ~ N(0, I)` and reconstructs the
#'   field as `w = f(z, sigma2, phi)`, avoiding the field/hyperparameter funnel),
#'   `"centered"` (places the NNGP density on the field directly), or
#'   `"collapsed"` (deprecated).
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
                       cov = c("exponential", "matern"),
                       nu = 1.5,
                       nn = 15,
                       m = 6,
                       c = 1.5,
                       shared = NULL,
                       scale_coords = TRUE,
                       parameterization = c("noncentered", "centered", "collapsed")) {

  approx <- match.arg(approx)
  cov <- match.arg(cov)
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

  coord_vars <- .parse_coord_spec(coords, "gp()", allow_nd = TRUE)

  # Hilbert-space approximation: a basis-function GP with `m` functions per
  # dimension and boundary factor `c` (the NNGP cov/nu/nn args do not
  # apply). Returns a tulpa_hsgp spec.
  if (approx == "hsgp") {
    # The Hilbert-space basis is assembled on a 2-D domain and stored at a fixed
    # 2-D stride, so unlike the NNGP branch it cannot carry another dimension.
    if (length(coord_vars) != 2L) {
      stop("gp(approx = \"hsgp\") requires exactly 2 coordinate variables ",
           "(x, y); got ", length(coord_vars), ". The NNGP approximation ",
           "accepts any number.", call. = FALSE)
    }
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

  # Validate nu for Matern. The NNGP fit path is wired only for nu in
  # {1.5, 2.5} (gp_cov_type_for_laplace), so reject the rest here rather than
  # deep in the fit. nu = 0.5 is the exponential kernel -- use cov = "exponential".
  if (cov == "matern") {
    if (!is.numeric(nu) || length(nu) != 1 || nu <= 0) {
      stop("`nu` must be a positive number for Matern covariance", call. = FALSE)
    }
    if (!isTRUE(all.equal(nu, 1.5)) && !isTRUE(all.equal(nu, 2.5))) {
      stop("Matern NNGP supports nu in {1.5, 2.5}; got nu = ", format(nu),
           ". Use nu = 1.5 or 2.5, or cov = \"exponential\" for nu = 0.5.",
           call. = FALSE)
    }
  }

  # Validate nn
  if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
    stop("`nn` must be a positive integer", call. = FALSE)
  }
  nn <- as.integer(nn)

  if (isFALSE(shared)) .warn_nonshared("spatial effects")

  structure(
    list(
      type = "gp",
      coord_vars = coord_vars,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      nn = nn,
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
#' @param range_local Plausible range interval for the local scale as
#'   `c(lower, upper)` in coordinate units. Default: `c(0.01, 1)` (after
#'   scaling). Under exact NUTS this is not a hard box: `lower` anchors a PC
#'   prior on that scale's range (`P(range < lower) = 0.05`, the same prior
#'   [spatial_gp()] uses), and the pair places the sampler's starting range at
#'   their geometric mean. The range itself is free on `(0, Inf)`.
#' @param range_regional Plausible range interval for the regional scale, read
#'   the same way. Default: `c(1, 10)` (after scaling). Keeping the two
#'   intervals separated is what identifies the scales against each other.
#' @param cov Covariance function: `"exponential"` (default) or `"matern"`.
#' @param nu Smoothness parameter for Matern covariance, one of `1.5` or `2.5`.
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
#' @param sampler Latent parameterization for the exact-NUTS field. `"auto"`
#'   (default) and `"noncentered"` sample `z ~ N(0, I)` per scale and
#'   reconstruct each field as `w = f(z, sigma2, phi)`, avoiding the
#'   field/hyperparameter funnel; `"centered"` places the NNGP density on each
#'   field directly. `"interweaved"` alternates between parameterizations and
#'   is not implemented on the exact-NUTS path.
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
#' set.seed(101)
#' n <- 60
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n)
#' )
#' df$count <- rpois(n, exp(1 + 0.2 * df$depth))
#'
#' # Both scales are sampled by exact NUTS (mode = "exact"); the field is
#' # returned as gp_local[i] / gp_regional[i] draws.
#' fit <- tulpa(
#'   count ~ depth + temp,
#'   data = df,
#'   family = "poisson",
#'   spatial = spatial_multiscale(
#'     ~ lon + lat,
#'     range_local = c(0.1, 0.5),
#'     range_regional = c(1, 5)
#'   ),
#'   mode = "exact",
#'   control = list(n_iter = 200L, n_warmup = 100L)
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
                               cov = c("exponential", "matern"),
                               nu = 1.5,
                               nn_local = 10,
                               nn_regional = 30,
                               shared = NULL,
                               scale_coords = TRUE,
                               sampler = c("auto", "noncentered", "centered",
                                          "interweaved")) {

  approx <- match.arg(approx)
  cov <- match.arg(cov)
  sampler <- match.arg(sampler)

  coord_vars <- .parse_coord_spec(coords, "multiscale()")

  # Validate scales
  if (length(scales) != 2) {
    stop("Currently only 2 scales are supported", call. = FALSE)
  }

  # The range-prior bounds set the scale separation and are unit-dependent;
  # surface the defaults so an unset call does not silently anchor them.
  if (missing(range_local) || missing(range_regional)) {
    message(sprintf(paste0(
      "spatial_multiscale(): using default range-prior bounds ",
      "range_local = c(%g, %g), range_regional = c(%g, %g) (coordinate units). ",
      "Set them to match your coordinate scale."),
      range_local[1], range_local[2], range_regional[1], range_regional[2]))
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

  # Extract coordinates over however many were named. The NNGP/GP kernels and
  # the neighbour construction both read every coordinate column, so the
  # extraction must not decide the dimension either.
  coords <- as.matrix(data[, gp$coord_vars, drop = FALSE])
  storage.mode(coords) <- "double"

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale coordinates if requested
  if (gp$scale_coords) {
    coords <- scale(coords)
  }

  # Detect unique coordinates (NNGP requires unique locations)
  coord_key <- do.call(paste, c(
    lapply(seq_len(ncol(coords)), function(j) coords[, j]), list(sep = ",")))
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
