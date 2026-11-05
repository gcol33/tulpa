# The Matern smoothness the NNGP paths are wired for.
#
# `gp_cov_type()` maps nu to one kernel code and rejects anything else, so a
# constructor that stores an unsupported nu only defers the error into the fit.
# `spatial_gp()` checked it and its two siblings did not, so
# `spatial_multiscale(nu = 7)` was accepted at construction (gcol33/tulpa#701).
#' @keywords internal
.check_matern_nu <- function(nu, where = "Matern NNGP") {
  if (!is.numeric(nu) || length(nu) != 1 || is.na(nu) || nu <= 0) {
    stop("`nu` must be a positive number for Matern covariance", call. = FALSE)
  }
  if (!isTRUE(all.equal(nu, 1.5)) && !isTRUE(all.equal(nu, 2.5))) {
    stop(where, " supports nu in {1.5, 2.5}; got nu = ", format(nu),
         ". Use nu = 1.5 or 2.5, or cov = \"exponential\" for nu = 0.5.",
         call. = FALSE)
  }
  invisible(nu)
}

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
#'   basis GP with `m` functions per dimension and boundary factor `c`, which
#'   approximates the kernel `cov` / `nu` name).
#' @param cov Covariance function. With `approx = "nngp"`, `"exponential"`
#'   (the default) or `"matern"`. With `approx = "hsgp"`, `"matern"` (the
#'   default, at `nu = 1.5`) or `"squared_exponential"`; the exponential kernel
#'   has no HSGP basis rule. `NULL` takes the approximation's default.
#' @param nu Matern smoothness parameter, one of `1.5` or `2.5`. Used only when
#'   `cov = "matern"` (`nu = 0.5` is `cov = "exponential"`).
#' @param nn Number of nearest neighbours used in the NNGP approximation.
#' @param m Number of HSGP basis functions per dimension (`approx = "hsgp"`),
#'   between 3 and 50. `NULL` (default) chooses it from the data; see Details.
#' @param c HSGP boundary factor, `>= 1` (`approx = "hsgp"`): the basis lives
#'   on the coordinates' bounding box widened by `c`. `NULL` (default) chooses
#'   it from the data; see Details.
#' @param sigma_prior_U,sigma_prior_alpha Penalized-complexity prior on the
#'   field's marginal standard deviation (`approx = "hsgp"`), calibrated so that
#'   `P(sigma > sigma_prior_U) = sigma_prior_alpha`. Defaults to
#'   `P(sigma > 1) = 0.01`. `sigma_prior_U` must be positive and
#'   `sigma_prior_alpha` must lie in `(0, 1)`.
#'   Sampler and nested-Laplace routes use these anchors. Under
#'   `hyperprior = "flat"`, nested Laplace retains explicitly supplied anchors.
#' @param shared Whether the spatial effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares the effect; `FALSE` fits
#'   process-specific effects and emits a warning.
#' @param scale_coords Logical. Standardize coordinates before fitting
#'   (default `TRUE`): each column is centred and all are divided by one
#'   common factor, so distances keep their proportions and an isotropic
#'   kernel stays isotropic. [spatial_range()] reports the range in the
#'   original coordinate units either way.
#' @param parameterization Latent parameterization for the exact-NUTS field.
#'   One of `"noncentered"` (default; samples `z ~ N(0, I)` and reconstructs the
#'   field as `w = f(z, sigma2, phi)`, avoiding the field/hyperparameter funnel),
#'   `"centered"` (places the NNGP density on the field directly), or
#'   `"collapsed"` (deprecated).
#'
#' @return A `tulpa_gp` object (also of class `tulpa_spatial`).
#'
#' @details
#' **HSGP kernel.** The basis approximates a Matern kernel of smoothness 3/2
#' (the default) or 5/2, `sigma^2 (1 + sqrt(3) d / ell) exp(-sqrt(3) d / ell)`
#' and `sigma^2 (1 + sqrt(5) d / ell + 5 d^2 / (3 ell^2)) exp(-sqrt(5) d / ell)`
#' at distance `d`, or the squared-exponential kernel
#' `sigma^2 exp(-d^2 / (2 ell^2))`, through each kernel's spectral density
#' (Riutort-Mayol et al. 2023, Eqs. 1-3). The squared-exponential kernel draws
#' fields smoother than any Matern one, and a field rougher than the kernel
#' assumes is smoothed and its intervals narrowed.
#'
#' For nested Laplace and Hamiltonian inference, the default PC (penalised
#' complexity) prior places half its probability
#' below an effective range of one fifth of the coordinate bounding-box
#' diagonal. The effective range is `rho = 2 * ell`, where `ell` is the kernel
#' lengthscale (`phi` in the NNGP kernels). Both approximations use this
#' anchor. [spatial_multiscale()] takes each scale's anchor
#' from its declared range interval.
#'
#' **HSGP basis size.** A Hilbert-space basis of `m` functions per dimension on
#' a domain widened by `c` has a lower lengthscale cutoff
#' `a c max(S) / m`, where `S` contains the coordinate half-widths and
#' `c >= 1.2`. The coefficient `a`
#' retains 99% of the joint two-dimensional spectral variance using the
#' tensor expansion and cumulative-variance recommendation of Riutort-Mayol
#' et al. (2023, Eqs. 9-12, Section 4.2). It is at least their one-dimensional
#' covariance-error coefficient (Eqs. 19, 21, 23). The boundary factor follows
#' Eqs. 20, 22, 24 and bounds the variance removed by the Dirichlet boundary
#' at the domain's corners by 1%, using the kernel correlations at twice the
#' distance to each face. A rough
#' field asks for a short lengthscale, and a basis too small for it smooths the
#' field and narrows its intervals. Left unset, `m` and `c` follow the
#' iterative procedure of the paper's Section 4.5.1. In its Phase A the first
#' fit takes them from the prior median of the lengthscale, and while the
#' posterior median lengthscale falls below the smallest one the fitted basis
#' represents (the paper's diagnostic, with its slack of 0.01 on the
#' lengthscale divided by `S`), the field is fitted again with the `m` and `c`
#' that estimate asks for. In its Phase B five functions per dimension are
#' added and the field fitted again until the refit passes the diagnostic and
#' is stable against the fit before it: the paper names the lengthscale
#' estimate and the predictive accuracy as what has to be stable and gives no
#' threshold, and stable is read here as the refitted posterior mean of the
#' lengthscale, and of the linear predictor at every observation, lying inside
#' the previous fit's 95% interval. `m` never shrinks below a size already
#' fitted and stops at 50. A supplied `m` or `c` is kept as given, and a
#' supplied `m` ends the procedure after Phase A.
#' [diagnostics()] reports when the posterior median lengthscale lies outside
#' the window of the fitted basis at either end; above it, the field is
#' smoother than the boundary factor allows for, and a larger `c` covers it.
#'
#' @references
#' Riutort-Mayol, G., Buerkner, P.-C., Andersen, M. R., Solin, A. and Vehtari,
#' A. (2023). Practical Hilbert space approximate Bayesian Gaussian processes
#' for probabilistic programming. *Statistics and Computing* 33, 17.
#' \doi{10.1007/s11222-022-10167-2}
#'
#' @seealso [spatial_car()], [spatial_bym2()] for areal spatial effects.
#'
#' @examples
#' # GP spatial specification from coordinate columns
#' spatial_gp(~ lon + lat)
#' spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
#' spatial_gp(~ lon + lat, approx = "hsgp", cov = "squared_exponential")
#'
#' @export

spatial_gp <- function(coords,
                       approx = c("nngp", "hsgp"),
                       cov = NULL,
                       nu = 1.5,
                       nn = 15,
                       m = NULL,
                       c = NULL,
                       sigma_prior_U = 1,
                       sigma_prior_alpha = 0.01,
                       shared = NULL,
                       scale_coords = TRUE,
                       parameterization = c("noncentered", "centered", "collapsed")) {

  approx <- match.arg(approx)
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
    m <- .hsgp_check_m(m)
    c <- .hsgp_check_c(c)
    kernel <- .hsgp_kernel(cov, nu, "gp(approx = \"hsgp\")")
    .check_pc_anchors(sigma_prior_U, sigma_prior_alpha,
                      "sigma_prior_U", "sigma_prior_alpha",
                      "gp(approx = \"hsgp\")")
    if (isFALSE(shared)) .warn_nonshared("spatial effects")
    return(structure(
      list(type = "hsgp", coord_vars = coord_vars, m = m, c = c,
           kernel = kernel,
           basis_rule = list(auto = c(m = is.null(m), c = is.null(c))),
           sigma2_prior_U = as.numeric(sigma_prior_U),
           sigma2_prior_alpha = as.numeric(sigma_prior_alpha),
           sigma_prior_stated = !missing(sigma_prior_U) || !missing(sigma_prior_alpha),
           shared = shared, scale_coords = scale_coords,
           n_obs = NULL, coords_matrix = NULL),
      class = c("tulpa_hsgp", "tulpa_spatial", "list")))
  }

  cov <- match.arg(cov %||% "exponential", c("exponential", "matern"))
  if (cov == "matern") .check_matern_nu(nu)

  # Validate nn
  if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
    stop("`nn` must be a positive integer", call. = FALSE)
  }
  nn <- as.integer(nn)

  if (isFALSE(shared)) .warn_nonshared("spatial effects")

  # The door accepts these on either branch; only the HSGP branch stored them,
  # so an NNGP user's anchors were neither used nor refused and the sampler
  # applied a hardcoded pair of its own (gcol33/tulpa#700).
  .check_pc_anchors(sigma_prior_U, sigma_prior_alpha,
                    "sigma_prior_U", "sigma_prior_alpha", "gp()")

  structure(
    list(
      type = "gp",
      coord_vars = coord_vars,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      sigma2_prior_U = as.numeric(sigma_prior_U),
      sigma2_prior_alpha = as.numeric(sigma_prior_alpha),
      sigma_prior_stated = !missing(sigma_prior_U) || !missing(sigma_prior_alpha),
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


#' @rdname spatial_gp
#' @param x A `tulpa_gp` or `tulpa_hsgp` specification.
#' @param ... Ignored.
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


#' @rdname spatial_gp
#' @export
print.tulpa_hsgp <- function(x, ...) {
  cat("tulpa Hilbert Space GP (HSGP) spatial specification\n")
  cat("=====================================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Kernel:", .hsgp_kernel_info(.hsgp_spec_kernel(x))$label, "\n")
  if (is.null(x$m)) cat("Basis functions: chosen from the data\n")
  else cat("Basis functions:", x$m, "per dim (", x$m^2, "total )\n")
  if (is.null(x[["c"]])) cat("Boundary factor: chosen from the data\n")
  else cat("Boundary factor:", format(x[["c"]], digits = 3), "\n")
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
#' @noRd
validate_hsgp <- function(spatial, data) {
  coords <- prepare_coords(spatial$coord_vars, data, spatial$scale_coords)
  spatial$n_obs <- nrow(coords)
  spatial$coords_matrix <- coords
  .hsgp_basis_init(spatial, coords, .hsgp_prior_lengthscale(coords), "c",
                   "spatial_gp(approx = \"hsgp\")")
}


#' Multi-scale Gaussian process spatial structure
#'
#' @description
#' Specify a multi-scale spatial random effect that decomposes spatial
#' variation into local (fine-scale) and regional (broad-scale) components.
#' Each scale has its own range and variance parameters.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param scales Character vector specifying scale names. Default: `c("local", "regional")`.
#' @param range_local Plausible range interval for the local scale as
#'   `c(lower, upper)` in coordinate units. `NULL` (default) takes the upper
#'   bound 1 (after scaling) and the lower bound from the data: the median, over
#'   the distinct locations, of the distance to the nearest other location. A
#'   range below the spacing of the locations is one the data do not inform,
#'   and with `approx = "hsgp"` a lower bound there asks the basis rule for
#'   more functions than the data can use. This is not a hard box: `lower` anchors a PC prior on that
#'   scale's range (`P(range < lower) = 0.05`, the same prior [spatial_gp()]
#'   uses), on every route that fits the field; a sampler starts the range at
#'   the pair's geometric mean. The range itself is free on `(0, Inf)`. The
#'   range is the Matern range `rho = 2 * ell`, matching [spatial_spde()],
#'   on both approximations. Here `ell` is the kernel lengthscale (`phi` in
#'   the NNGP kernels). The squared-exponential kernel uses the same
#'   effective-range convention.
#' @param range_regional Plausible range interval for the regional scale, read
#'   the same way. Default: `c(1, 10)` (after scaling). Keeping the two
#'   intervals separated is what identifies the scales against each other.
#' @param cov Covariance function, read as [spatial_gp()] reads it: with
#'   `approx = "nngp"`, `"exponential"` (default) or `"matern"`; with
#'   `approx = "hsgp"`, `"matern"` (default, at `nu = 1.5`) or
#'   `"squared_exponential"`.
#' @param nu Smoothness parameter for Matern covariance, one of `1.5` or `2.5`.
#' @param nn_local Number of nearest neighbors for local scale. Default 10.
#' @param nn_regional Number of nearest neighbors for regional scale. Default 30.
#' @param shared Logical; if TRUE (default), spatial effects enter both
#'   all processes.
#' @param scale_coords Logical; if TRUE (default), coordinates are centred
#'   and divided by one common factor before computing distances (see
#'   [spatial_gp()]).
#' @param approx Approximation method: `"nngp"` (default) for Nearest Neighbor
#'   GP; `"hsgp"` for a Hilbert-space basis shared by both scales, which is
#'   fitted by the nested-Laplace path only (`mode = "auto"` or
#'   `"nested_laplace"`); the samplers carry the NNGP form.
#' @param m Number of HSGP basis functions per dimension, between 3 and 50;
#'   only used when `approx = "hsgp"`, where each scale carries its own basis
#'   of m^2 functions. A value given applies to both scales. `NULL` (default)
#'   chooses each scale's from the data by the rule [spatial_gp()] describes:
#'   first at the median of that scale's range prior, then at its posterior
#'   median lengthscale while the basis does not represent it.
#' @param c_boundary Boundary factor for HSGP domain extension, `>= 1`; only
#'   used when `approx = "hsgp"`. A value given applies to both scales; `NULL`
#'   (default) chooses each scale's from the data.
#' @param sampler Latent parameterization for the exact-NUTS field. `"auto"`
#'   (default) and `"noncentered"` sample `z ~ N(0, I)` per scale and
#'   reconstruct each field as `w = f(z, sigma2, phi)`, avoiding the
#'   field/hyperparameter funnel; `"centered"` places the NNGP density on each
#'   field directly.
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
#' **Fitting.** `mode = "auto"`, `"structured"` and `"nested_laplace"` fit the
#' field by nested Laplace, as two field blocks of the multi-block driver (two
#' NNGP blocks, each on its own neighbour structure, or two HSGP blocks on one
#' basis). Each scale carries its own SD and range axes on the outer grid,
#' placed and refined like any other field axis, so both scales' variances and
#' ranges are integrated. The exact samplers (`mode = "exact"` and its
#' siblings) sample the NNGP form with the same priors. [spatial_range()]
#' reports each scale's `sigma` and `range` (`3 * phi`), tagged `[local]` /
#' `[regional]`; [field_effects()] reports the summed field per distinct
#' location, or one scale with `component =`.
#'
#' **Identifiability.** Both fields are proper, so neither needs a level
#' constraint; the constant they share with the intercept is held by their
#' priors. What tells the scales apart is the range priors: the regional
#' field's PC prior puts 5% of its mass below `range_regional[1]`, and the
#' local field's below `range_local[1]`, while neither range is bounded above.
#' Where the data do not separate the scales, the local range can therefore
#' reach into the regional interval and the two scales share the variance,
#' which widens both intervals rather than switching the labels. On simulated
#' Poisson data with 300 locations on a 10 x 10 square (local SD 0.63 and
#' range 0.5, regional SD 1 and range 3, `range_local = c(0.2, 1.5)`,
#' `range_regional = c(1.5, 10)`), the nested 95% intervals of all four
#' hyperparameters covered the truth on each of 5 data sets, with posterior
#' means 0.66 / 1.1 (local SD / range) and 1.03 / 7.3 (regional); the
#' component maps correlated with the simulated fields at 0.49 (local), 0.79
#' (regional) and 0.85 (sum).
#'
#' **Computational cost.** On those data sets the nested fit took 97-125 s
#' (100 locations: 29 s). Exact NUTS on 100 locations took 453 s for 4 chains
#' of 1000 iterations and had not converged (R-hat 1.14 and bulk ESS 13 on the
#' local log-variance), its chains disagreeing on how the variance splits
#' between the scales.
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
#' \donttest{
#' set.seed(101)
#' n <- 25
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n)
#' )
#' df$count <- rpois(n, exp(1 + 0.2 * df$depth))
#'
#' # Both scales' variances and ranges are integrated by nested Laplace.
#' fit <- tulpa(
#'   count ~ depth + temp,
#'   data = df,
#'   family = "poisson",
#'   spatial = spatial_multiscale(
#'     ~ lon + lat,
#'     range_local = c(0.1, 0.5),
#'     range_regional = c(1, 5),
#'     nn_local = 5L,
#'     nn_regional = 8L
#'   )
#' )
#' spatial_range(fit)
#' head(field_effects(fit, component = "regional"))
#' }
#'
#' @seealso [spatial_gp()] for single-scale GP, [temporal_multiscale()] for
#'   multi-scale temporal effects
#'
#' @export
spatial_multiscale <- function(coords,
                               scales = c("local", "regional"),
                               approx = c("nngp", "hsgp"),
                               m = NULL,
                               c_boundary = NULL,
                               range_local = NULL,
                               range_regional = c(1, 10),
                               cov = NULL,
                               nu = 1.5,
                               nn_local = 10,
                               nn_regional = 30,
                               shared = NULL,
                               scale_coords = TRUE,
                               sampler = c("auto", "noncentered", "centered")) {

  approx <- match.arg(approx)
  kernel <- NULL
  if (approx == "hsgp") {
    kernel <- .hsgp_kernel(cov, nu, "multiscale(approx = \"hsgp\")")
    cov <- NULL
  } else {
    cov <- match.arg(cov %||% "exponential", c("exponential", "matern"))
    if (cov == "matern") .check_matern_nu(nu, "Matern multi-scale NNGP")
  }
  sampler <- match.arg(sampler)

  coord_vars <- .parse_coord_spec(coords, "multiscale()")

  # Validate scales
  if (length(scales) != 2) {
    stop("Currently only 2 scales are supported", call. = FALSE)
  }

  # The range-prior bounds set the scale separation and are unit-dependent;
  # surface the defaults so an unset call does not silently anchor them.
  if (is.null(range_local) || missing(range_regional)) {
    message(sprintf(paste0(
      "spatial_multiscale(): using default range-prior bounds ",
      "range_local = %s, range_regional = c(%g, %g) (coordinate units). ",
      "Set them to match your coordinate scale."),
      if (is.null(range_local)) {
        sprintf("c(<nearest-neighbour spacing>, %g)", .MSGP_RANGE_LOCAL_UPPER)
      } else sprintf("c(%g, %g)", range_local[1], range_local[2]),
      range_regional[1], range_regional[2]))
  }

  # Validate range specifications
  if (!is.null(range_local) &&
      (length(range_local) != 2 || range_local[1] >= range_local[2])) {
    stop("`range_local` must be c(lower, upper) with lower < upper", call. = FALSE)
  }
  if (length(range_regional) != 2 || range_regional[1] >= range_regional[2]) {
    stop("`range_regional` must be c(lower, upper) with lower < upper", call. = FALSE)
  }

  # Check range separation
  if ((range_local[2] %||% .MSGP_RANGE_LOCAL_UPPER) > range_regional[1]) {
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
  if (approx == "hsgp") {
    m <- .hsgp_check_m(m)
    c_boundary <- .hsgp_check_c(c_boundary, "c_boundary")
  }

  structure(
    list(
      type = "multiscale",
      approx = approx,
      m = m,
      c_boundary = c_boundary,
      coord_vars = coord_vars,
      scales = scales,
      range_local = range_local,
      range_regional = range_regional,
      cov = cov,
      nu = if (identical(cov, "matern")) nu else NULL,
      kernel = kernel,
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


#' @rdname spatial_multiscale
#' @param x A `tulpa_multiscale` specification.
#' @param ... Ignored.
#' @export
print.tulpa_multiscale <- function(x, ...) {
  cat("tulpa Multi-Scale spatial specification\n")
  cat("========================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Scales:", paste(x$scales, collapse = " + "), "\n\n")

  cat("Local scale:\n")
  rl <- x$range_local %||%
    c("nearest-neighbour spacing", .MSGP_RANGE_LOCAL_UPPER)
  cat("  Range prior: [", rl[1], ", ", rl[2], "]\n", sep = "")
  cat("  Neighbors:", x$nn_local, "\n")

  cat("\nRegional scale:\n")
  cat("  Range prior: [", x$range_regional[1], ", ", x$range_regional[2], "]\n", sep = "")
  cat("  Neighbors:", x$nn_regional, "\n")

  cat("\nCovariance:", .spatial_cov_label(x), "\n")
  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate a GP spatial specification against data
#'
#' Fills in the neighbor structure ([spatial_gp()]) or structures
#' ([spatial_multiscale()]) that [tulpa_gibbs()] and [tulpa_laplace()] need
#' from a `spatial =` spec, since those design-matrix doors take no `data`
#' argument to validate one themselves. [tulpa()] calls this internally, so it
#' only needs to be called directly when fitting through a design-matrix door.
#'
#' @param gp A [spatial_gp()] or [spatial_multiscale()] object.
#' @param data Data frame containing the coordinate columns named in `gp`.
#'
#' @return Updated spatial object with computed neighbor structure.
#' @seealso [validate_temporal_multiscale()], [spatial_gp()],
#'   [spatial_multiscale()]
#' @examples
#' set.seed(1)
#' d <- data.frame(lon = runif(30), lat = runif(30))
#' gp <- validate_gp(spatial_gp(~ lon + lat, nn = 5), d)
#' gp$n_spatial
#' @export
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

  .check_coords_finite(coords, "Coordinate columns", scale = gp$scale_coords)

  # Scale coordinates if requested, by one common factor so an isotropic kernel
  # stays isotropic in the user's geometry (gcol33/tulpa#907).
  if (gp$scale_coords) {
    coords <- .scale_coords_isotropic(coords)
  }

  # Detect unique coordinates (NNGP requires unique locations)
  ul <- .unique_locations(coords)
  obs_to_loc <- ul$obs_to_loc
  unique_coords <- ul$unique_coords
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
    # The local range interval left unset starts at the spacing of the
    # locations, the shortest range the data inform.
    if (is.null(gp$range_local)) {
      lo <- .nn_spacing(unique_coords)
      if (!is.finite(lo) || lo >= .MSGP_RANGE_LOCAL_UPPER) {
        stop("spatial_multiscale(): the locations' nearest-neighbour spacing (",
             format(lo, digits = 3), ") leaves no default local range interval ",
             "below ", .MSGP_RANGE_LOCAL_UPPER, "; set `range_local`.",
             call. = FALSE)
      }
      gp$range_local <- c(lo, .MSGP_RANGE_LOCAL_UPPER)
    }
    # Each scale carries its own HSGP basis, sized by the rule from the median
    # of its own lengthscale prior (the PC range prior anchored at the scale's
    # lower bound, `.multiscale_nl_blocks()`) and refitted on its own
    # lengthscale posterior (`b1.lengthscale` local, `b2.lengthscale`
    # regional). One shared basis would have to reach the local scale's
    # lengthscale with the boundary factor the regional one needs.
    if (identical(gp$approx, "hsgp")) {
      scs <- c("local", "regional")
      gp$scale_basis <- stats::setNames(lapply(seq_along(scs), function(b) {
        ell0 <- .pc_range_quantile(
          0.5, as.numeric(gp[[paste0("range_", scs[b])]])[1L],
          .MSGP_RANGE_ALPHA, .HSGP_LENGTHSCALE_PRIOR_DIM) / .SPATIAL_RANGE_FACTOR
        .hsgp_basis_init(
          list(m = gp$m, c = gp$c_boundary, kernel = gp$kernel,
               basis_rule = list(axes = paste0("b", b, ".lengthscale"))),
          coords, ell0, "c",
          sprintf("spatial_multiscale(approx = \"hsgp\"), %s scale", scs[b]))
      }), scs)
    }
  }

  gp
}


# The spacing of a set of locations: the median, over the distinct locations,
# of the distance to the nearest other one. NA for fewer than two locations.
# Computed in blocks of rows, so memory stays linear in the number of
# locations.
.nn_spacing <- function(coords) {
  xy <- unique(as.matrix(coords))
  n <- nrow(xy)
  if (n < 2L) return(NA_real_)
  sq <- rowSums(xy^2)
  d_min <- numeric(n)
  for (s in seq(1L, n, by = 512L)) {
    i <- s:min(n, s + 511L)
    d2 <- outer(sq[i], sq, `+`) - 2 * xy[i, , drop = FALSE] %*% t(xy)
    d2[cbind(seq_along(i), i)] <- Inf
    d_min[i] <- sqrt(pmax(apply(d2, 1L, min), 0))
  }
  stats::median(d_min)
}

# The upper bound of the default local range interval of spatial_multiscale(),
# in scaled coordinate units; its lower bound is the locations' spacing
# (`.nn_spacing()`).
.MSGP_RANGE_LOCAL_UPPER <- 1

# The distinct locations of a coordinate matrix and each row's one, in order of
# first appearance. An NNGP field lives on locations, not rows: two rows at one
# site would be two field values at distance 0, perfectly correlated, and the
# neighbour covariance singular. Shared by validate_gp() and validate_svc().
.unique_locations <- function(coords) {
  coord_key <- do.call(paste, c(
    lapply(seq_len(ncol(coords)), function(j) coords[, j]), list(sep = ",")))
  unique_keys <- unique(coord_key)
  list(obs_to_loc = as.integer(match(coord_key, unique_keys)),
       unique_coords = coords[match(unique_keys, coord_key), , drop = FALSE])
}
