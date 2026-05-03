temporal_multiscale <- function(time_var,
                                trend = c("rw2", "rw1", "none"),
                                seasonal = NULL,
                                short_term = c("ar1", "iid", "none"),
                                group_var = NULL,
                                shared = NULL) {

  trend <- match.arg(trend)
  short_term <- match.arg(short_term)

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  if (!is.null(group_var)) {
    if (!is.character(group_var) || length(group_var) != 1) {
      stop("`group_var` must be a single character string", call. = FALSE)
    }
  }

  # Validate seasonal
  if (!is.null(seasonal)) {
    if (!is.numeric(seasonal) || length(seasonal) != 1 || seasonal < 2) {
      stop("`seasonal` must be an integer >= 2 (the period)", call. = FALSE)
    }
    seasonal <- as.integer(seasonal)
  }

  # Check at least one component is active
  if (trend == "none" && is.null(seasonal) && short_term == "none") {
    stop("At least one temporal component must be active", call. = FALSE)
  }

  # Determine which components are active
  components <- character(0)
  if (trend != "none") components <- c(components, "trend")
  if (!is.null(seasonal) && seasonal >= 2) components <- c(components, "seasonal")
  if (short_term != "none") components <- c(components, "short_term")

  structure(
    list(
      type = "multiscale",
      time_var = time_var,
      group_var = group_var,
      shared = shared,
      # Component specifications
      trend = trend,
      seasonal = seasonal,
      short_term = short_term,
      components = components,
      # Filled in during validation
      n_times = NULL,
      n_groups = NULL,
      time_index = NULL,
      group_index = NULL,
      precision_structures = NULL
    ),
    class = c("tulpa_temporal_multiscale", "tulpa_temporal", "list")
  )
}


#' Print method for tulpa_temporal_multiscale
#'
#' @param x A tulpa_temporal_multiscale object
#' @param ... Ignored
#'
#' @export
print.tulpa_temporal_multiscale <- function(x, ...) {
  cat("tulpa Multi-Scale temporal specification\n")
  cat("=========================================\n\n")

  cat("Time variable:", x$time_var, "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  cat("\nComponents:\n")

  if (x$trend != "none") {
    trend_name <- switch(x$trend,
      rw1 = "RW1 (first-order random walk)",
      rw2 = "RW2 (second-order random walk, smooth)"
    )
    cat("  Trend:", trend_name, "\n")
  }

  if (!is.null(x$seasonal) && x$seasonal >= 2) {
    cat("  Seasonal: Period =", x$seasonal, "(cyclic RW1)\n")
  }

  if (x$short_term != "none") {
    short_name <- switch(x$short_term,
      ar1 = "AR(1) (autocorrelated residuals)",
      iid = "IID (independent residuals)"
    )
    cat("  Short-term:", short_name, "\n")
  }

  cat("\nShared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_times)) {
    cat("Time points:", x$n_times, "\n")
  }
  if (!is.null(x$n_groups) && x$n_groups > 1) {
    cat("Groups:", x$n_groups, "\n")
  }

  invisible(x)
}


#' Validate multi-scale temporal specification
#'
#' @param temporal tulpa_temporal_multiscale object
#' @param data Data frame
#'
#' @return Updated object with indices computed
#' @keywords internal
validate_temporal_multiscale <- function(temporal, data) {
  if (is.null(temporal)) return(NULL)
  if (!inherits(temporal, "tulpa_temporal_multiscale")) {
    # Use regular validation for single-component temporal
    return(validate_temporal(temporal, data))
  }

  assert_columns_exist(temporal$time_var, data, role = "Temporal")
  if (!is.null(temporal$group_var)) {
    assert_columns_exist(temporal$group_var, data, role = "Temporal group")
  }

  # Get time values and create indices
  time_vals <- data[[temporal$time_var]]

  if (is.factor(time_vals)) {
    time_factor <- time_vals
  } else {
    unique_times <- sort(unique(time_vals))
    time_factor <- factor(time_vals, levels = unique_times)
  }

  temporal$n_times <- nlevels(time_factor)
  temporal$time_index <- as.integer(time_factor)
  temporal$time_levels <- levels(time_factor)

  # Validate minimum time points
  if (temporal$trend == "rw2" && temporal$n_times < 3) {
    stop("RW2 trend requires at least 3 time points", call. = FALSE)
  }

  if (!is.null(temporal$seasonal) && temporal$n_times < temporal$seasonal) {
    warning(sprintf(
      "Number of time points (%d) is less than seasonal period (%d).\n",
      temporal$n_times, temporal$seasonal
    ), "Seasonal component may not be well-identified.", call. = FALSE)
  }

  # Handle grouping
  if (!is.null(temporal$group_var)) {
    group_vals <- data[[temporal$group_var]]
    group_factor <- as.factor(group_vals)
    temporal$n_groups <- nlevels(group_factor)
    temporal$group_index <- as.integer(group_factor)
    temporal$group_levels <- levels(group_factor)
  } else {
    temporal$n_groups <- 1L
    temporal$group_index <- rep(1L, nrow(data))
  }

  # Build precision structures for each component
  temporal$precision_structures <- list()

  if (temporal$trend != "none") {
    temporal$precision_structures$trend <- list(
      type = temporal$trend,
      T = temporal$n_times,
      cyclic = FALSE,
      rank_deficiency = if (temporal$trend == "rw1") 1 else 2
    )
  }

  if (!is.null(temporal$seasonal) && temporal$seasonal >= 2) {
    temporal$precision_structures$seasonal <- list(
      type = "rw1",
      T = temporal$seasonal,
      cyclic = TRUE,
      rank_deficiency = 1
    )
  }

  if (temporal$short_term == "ar1") {
    temporal$precision_structures$short_term <- list(
      type = "ar1",
      T = temporal$n_times,
      rank_deficiency = 0
    )
  } else if (temporal$short_term == "iid") {
    temporal$precision_structures$short_term <- list(
      type = "iid",
      T = temporal$n_times,
      rank_deficiency = 0
    )
  }

  # Calculate total number of temporal parameters
  n_params <- 0
  if (temporal$trend != "none") {
    n_params <- n_params + temporal$n_times
  }
  if (!is.null(temporal$seasonal) && temporal$seasonal >= 2) {
    n_params <- n_params + temporal$seasonal
  }
  if (temporal$short_term != "none") {
    n_params <- n_params + temporal$n_times
  }

  temporal$n_temporal_params <- n_params * temporal$n_groups

  temporal
}


# =============================================================================
# Gaussian Process Temporal Structure
# =============================================================================

#' Gaussian Process temporal structure
#'
#' @description
#' Specify a Gaussian Process (GP) temporal random effect for irregularly-spaced
#' or continuous time points. Unlike RW1/RW2/AR1 which assume equally-spaced
#' observations, GP temporal effects model correlation as a function of time
#' distance.
#'
#' This is particularly useful for:
#' - Irregularly-spaced time series
#' - Continuous time (e.g., exact timestamps)
#' - Smooth temporal trends with uncertainty
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`). Should be
#'   numeric (continuous time) or convertible to numeric.
#' @param cov Covariance function: `"exponential"` (default, rough),
#'   `"matern"` (tunable smoothness), `"gaussian"` (very smooth), or
#'   `"periodic"` (for seasonal patterns).
#' @param nu Smoothness parameter for Matern covariance. Common values:
#'   - 0.5: Equivalent to exponential (rough)
#'   - 1.5: Once differentiable (moderate smoothness)
#'   - 2.5: Twice differentiable (smooth)
#'   Ignored for non-Matern covariance functions.
#' @param period Period for periodic covariance (e.g., 12 for monthly, 365 for
#'   daily data with annual cycle). Only used when `cov = "periodic"`.
#' @param group_var Optional name of grouping variable for panel data.
#'   If provided, separate GPs are estimated for each group.
#' @param shared Logical; if TRUE (default), temporal effect enters both
#'   all processes.
#' @param scale_coords Logical; if TRUE (default), time values are scaled to
#'   unit variance before computing distances.
#' @param parameterization Parameterization for GP effects:
#'   `"noncentered"` (default) stores z ~ N(0,1) and scales by covariance
#'   (better for weakly-informed effects);
#'   `"centered"` stores effects directly (better for strongly-informed effects).
#'
#' @return A `tulpa_temporal_gp` object
#'
#' @details
#' The GP temporal model adds a time-correlated random effect:
#'
#' \deqn{\eta(t) = X\beta + f(t)}
#'
#' where \eqn{f(t)} follows a Gaussian process:
#' \deqn{f(t) \sim GP(0, \sigma^2 C(|t - t'|; \phi))}
#'
#' The correlation function \eqn{C(d; \phi)} depends on time distance \eqn{d}:
#'
#' - **Exponential**: \eqn{C(d) = \exp(-d/\phi)} - continuous but not differentiable
#' - **Matern**: Smooth with tunable roughness via \eqn{\nu}
#' - **Gaussian**: \eqn{C(d) = \exp(-(d/\phi)^2)} - infinitely differentiable
#' - **Periodic**: \eqn{C(d) = \exp(-2\sin^2(\pi d/p)/\phi^2)} - for seasonal data
#'
#' **Implementation**: Uses a state-space representation for O(n) computational
#' complexity when possible (exponential, Matern with half-integer nu).
#'
#' @examples
#' # Create GP temporal specification
#' temporal_gp("timestamp")
#' temporal_gp("day", cov = "matern", nu = 1.5)
#' temporal_gp("month", cov = "periodic", period = 12)
#'
#' \dontrun{
#' # Irregularly-spaced time series (not run - GP temporal experimental)
#' set.seed(140)
#' times <- sort(runif(30, 0, 100))
#' df <- data.frame(
#'   time = times,
#'   x = rnorm(30),
#'   count = rpois(30, lambda = 20),
#'   effort = rgamma(30, shape = 4, rate = 1)
#' )
#'
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   temporal = temporal_gp("time"),
#'   iter = 200, warmup = 100, chains = 1
#' )
#' }
#'
#' @seealso [temporal_rw1()], [temporal_ar1()] for equally-spaced temporal effects,
#'   [spatial_gp()] for spatial GP effects
#'
#' @export
temporal_gp <- function(time_var,
                        cov = c("exponential", "matern", "gaussian", "periodic"),
                        nu = 1.5,
                        period = NULL,
                        group_var = NULL,
                        shared = NULL,
                        scale_coords = TRUE,
                        parameterization = c("noncentered", "centered")) {

  cov <- match.arg(cov)
  parameterization <- match.arg(parameterization)

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  if (!is.null(group_var)) {
    if (!is.character(group_var) || length(group_var) != 1) {
      stop("`group_var` must be a single character string", call. = FALSE)
    }
  }

  # Validate nu for Matern
  if (cov == "matern") {
    if (!is.numeric(nu) || length(nu) != 1 || nu <= 0) {
      stop("`nu` must be a positive number for Matern covariance", call. = FALSE)
    }
  }

  # Validate period for periodic covariance
  if (cov == "periodic") {
    if (is.null(period) || !is.numeric(period) || length(period) != 1 || period <= 0) {
      stop("`period` must be a positive number when using periodic covariance",
           call. = FALSE)
    }
  }

  # Warning for non-shared temporal effects
  if (isFALSE(shared)) {
    warning(
      "Non-shared temporal GP effects (shared = FALSE) means effects are not shared across processes.\n",
      "Consider whether temporal effects should be shared between\n",
      "processes if shared confounding structure is expected.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "gp",
      time_var = time_var,
      group_var = group_var,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      period = if (cov == "periodic") period else NULL,
      shared = shared,
      scale_coords = scale_coords,
      parameterization = parameterization,
      cyclic = FALSE,
      # Filled in during validation
      n_times = NULL,
      n_groups = NULL,
      n_obs = NULL,
      time_values = NULL,
      time_index = NULL,
      group_index = NULL,
      time_distances = NULL
    ),
    class = c("tulpa_temporal_gp", "tulpa_temporal", "list")
  )
}


#' Print method for tulpa_temporal_gp
#'
#' @param x A tulpa_temporal_gp object
#' @param ... Ignored
#'
#' @export
print.tulpa_temporal_gp <- function(x, ...) {
  cat("tulpa Gaussian Process temporal specification\n")
  cat("==============================================\n\n")

  cat("Time variable:", x$time_var, "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  cov_str <- x$cov
  if (x$cov == "matern" && !is.null(x$nu)) {
    cov_str <- sprintf("%s (nu = %.1f)", x$cov, x$nu)
  } else if (x$cov == "periodic" && !is.null(x$period)) {
    cov_str <- sprintf("%s (period = %.1f)", x$cov, x$period)
  }
  cat("Covariance:", cov_str, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
    cat("Unique time points:", x$n_times, "\n")
  }

  invisible(x)
}


#' Validate temporal GP specification against data
#'
#' @param temporal tulpa_temporal_gp object
#' @param data Data frame
#'
#' @return Updated object with computed time structure
#' @keywords internal
validate_temporal_gp <- function(temporal, data) {
  if (is.null(temporal)) return(NULL)
  if (!inherits(temporal, "tulpa_temporal_gp")) return(temporal)

  N <- nrow(data)

  # Check time variable exists
  if (!(temporal$time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", temporal$time_var),
         call. = FALSE)
  }

  # Extract time values
  time_vals <- data[[temporal$time_var]]

  # Convert to numeric
  if (!is.numeric(time_vals)) {
    if (inherits(time_vals, c("Date", "POSIXt"))) {
      time_vals <- as.numeric(time_vals)
    } else {
      stop("Time variable must be numeric or a date/time type", call. = FALSE)
    }
  }

  # Check for missing values
  if (any(is.na(time_vals))) {
    stop("Time variable contains missing values", call. = FALSE)
  }

  # Scale time values if requested
  if (temporal$scale_coords) {
    time_vals <- as.vector(scale(time_vals))
  }

  temporal$n_obs <- N

  # Extract unique time values (sorted) and create time index
  # GP uses one effect per unique time, not per observation
  unique_times <- sort(unique(time_vals))
  temporal$unique_time_values <- unique_times  # Unique times for GP cov matrix
  temporal$n_times <- length(unique_times)

  # Create time_index: maps each obs to its unique time point (1-based)
  temporal$time_index <- match(time_vals, unique_times)

  # Keep original time_values for backward compatibility
  temporal$time_values <- unique_times  # Now unique times, not per-obs

  # Handle grouping
  if (!is.null(temporal$group_var)) {
    if (!(temporal$group_var %in% names(data))) {
      stop(sprintf("Group variable '%s' not found in data", temporal$group_var),
           call. = FALSE)
    }
    group_vals <- data[[temporal$group_var]]
    group_factor <- as.factor(group_vals)
    temporal$n_groups <- nlevels(group_factor)
    temporal$group_index <- as.integer(group_factor)
    temporal$group_levels <- levels(group_factor)
  } else {
    temporal$n_groups <- 1L
    temporal$group_index <- rep(1L, N)
  }

  # Compute time distances for GP on unique times (not N x N)
  n_unique <- temporal$n_times
  if (n_unique <= 500) {
    temporal$time_distances <- outer(unique_times, unique_times,
                                     FUN = function(x, y) abs(x - y))
  } else {
    # For large data, we'll compute distances on-the-fly in C++
    temporal$time_distances <- NULL
  }

  # Total temporal parameters = n_times (one effect per unique time), not N
  temporal$n_temporal_params <- temporal$n_times * temporal$n_groups

  temporal
}


# =============================================================================
# Temporally-Varying Coefficients (TVC)
# =============================================================================

#' Temporally-Varying Coefficients (TVC)
#'
#' @description
#' Specify temporally-varying coefficients for tulpa models. TVCs allow
#' regression coefficients to change smoothly over time using a Gaussian
#' process or random walk prior. This captures how effects evolve temporally.
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`).
#' @param terms Which coefficients should vary over time. Options:
#'   - Integer vector: Column indices of design matrix (1 = intercept)
#'   - Character vector: Coefficient names (e.g., `"(Intercept)"`, `"depth"`)
#'   - Formula: `~ 1 + depth` for intercept and depth
#' @param structure Temporal structure for the varying coefficients:
#'   - `"rw1"`: First-order random walk (default)
#'   - `"rw2"`: Second-order random walk (smoother)
#'   - `"ar1"`: First-order autoregressive
#'   - `"gp"`: Gaussian process (for irregular time spacing)
#' @param group_var Optional name of grouping variable for panel data.
#' @param shared Logical; if TRUE (default), TVC effects enter both
#'   all processes. Set to FALSE for process-specific TVCs
#'   (triggers warning about potential confounding).
#'
#' @return A `tulpa_tvc` object
#'
#' @details
#' The TVC model extends the linear predictor:
#'
#' \deqn{\eta(t) = X\beta + \tilde{X}(t)w(t)}
#'
#' where:
#' - \eqn{\beta} are global (non-time-varying) coefficients
#' - \eqn{\tilde{X}(t)} is the subset of covariates with TVCs
#' - \eqn{w(t)} are time-varying adjustments
#'
#' Each TVC follows a temporal process:
#' \deqn{w_j(t) \sim RW1(\sigma_j^2) \text{ or other temporal structure}}
#'
#' **Interpretation**: A positive TVC for depth at time t means the depth
#' effect is stronger at t than the global average. The temporal variance
#' \eqn{\sigma^2_j} quantifies how much the effect varies over time.
#'
#' **Identifiability**: TVCs are centered (sum-to-zero constraint) to separate
#' the time-varying component from the global effect.
#'
#' @examples
#' # Create TVC specification
#' tvc <- temporal_tvc("year", terms = 1)
#' print(tvc)
#'
#' \dontrun{
#' # Generate synthetic data with time-varying effect (not run - TVC experimental)
#' set.seed(150)
#' df <- data.frame(
#'   year = rep(2010:2025, each = 5),
#'   x = rnorm(80),
#'   count = rpois(80, lambda = 20),
#'   effort = rgamma(80, shape = 4, rate = 1)
#' )
#'
#' # Time-varying intercept (random temporal field)
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   tvc = temporal_tvc("year", terms = 1),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#' summary(fit)
#'
#' # Time-varying effect of x
#' fit2 <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   tvc = temporal_tvc("year", terms = c("(Intercept)", "x"), structure = "rw2"),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Extract and plot the time-varying coefficients
#' tvc_effects <- tvc(fit2)
#' plot(tvc_effects, "x")
#' }
#'
#' @seealso [spatial_svc()] for spatially-varying coefficients,
#'   [temporal_rw1()] for temporal random effects,
#'   [tvc()] for extracting TVC posteriors
#'
#' @export
