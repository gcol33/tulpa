#' RW1 temporal structure (First-order Random Walk)
#'
#' @description
#' Specify a first-order random walk temporal random effect. RW1 penalizes
#' first differences, so adjacent time points are smoothed toward each other:
#' `phi[t] - phi[t-1] ~ N(0, sigma^2)`.
#'
#' @param time_var A formula (`~ time`) or single character string naming the
#'   time variable in the data.
#' @param group_var Optional formula (`~ g`) or character string naming a
#'   grouping variable. When supplied, a separate random walk is fit per group;
#'   `NULL` (default) fits a single walk shared across all observations.
#' @param cyclic Logical. If `TRUE`, the random walk wraps around so the last
#'   time point is a neighbour of the first (cyclic boundary, e.g. month of
#'   year). Default `FALSE`.
#' @param shared Whether the temporal effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares the effect; `FALSE` fits
#'   process-specific effects and emits a warning about unshared confounding.
#'
#' @return A `tulpa_temporal` object.
#'
#' @details
#' The precision matrix is rank `T - 1` (one constraint needed). RW1 is the
#' least smooth of the random-walk priors; for smoother trends see
#' [temporal_rw2()], and for a stationary alternative see [temporal_ar1()].
#'
#' @examples
#' # Create temporal RW1 specification
#' temporal_rw1("year")
#' temporal_rw1("month", cyclic = TRUE)
#'
#' @seealso [temporal_rw2()], [temporal_ar1()] for other temporal priors.
#'
#' @export
temporal_rw1 <- function(time_var, group_var = NULL, cyclic = FALSE,
                         shared = NULL) {

  time_var <- .coerce_var_arg(time_var, "time_var", "~ time")
  if (is.null(time_var)) {
    stop("`time_var` must be a formula (~ time) or single character string",
         call. = FALSE)
  }
  group_var <- .coerce_var_arg(group_var, "group_var")

  if (isFALSE(shared)) .warn_nonshared("temporal effects")

 structure(
   list(
     type = "rw1",
     time_var = time_var,
     group_var = group_var,
     cyclic = cyclic,
     shared = shared,
     # These will be filled in when validated against data
     n_times = NULL,
     n_groups = NULL,
     time_index = NULL,
     group_index = NULL
   ),
   class = c("tulpa_temporal", "list")
 )
}


#' RW2 temporal structure (Second-order Random Walk)
#'
#' @description
#' Specify a second-order random walk temporal random effect.
#' RW2 penalizes deviations from linearity, resulting in smoother trends
#' than RW1: `phi[t] - 2*phi[t-1] + phi[t-2] ~ N(0, sigma^2)`.
#'
#' @inheritParams temporal_rw1
#'
#' @return A `tulpa_temporal` object
#'
#' @details
#' RW2 produces smoother curves than RW1 because it penalizes the second
#' derivative (curvature) rather than the first derivative (slope).
#' It requires at least 3 time points.
#'
#' The precision matrix is rank T-2 (two constraints needed).
#'
#' @examples
#' # Create temporal RW2 specification
#' temporal_rw2("year")
#'
#' \donttest{
#' # Smooth temporal trend
#' set.seed(126)
#' df <- data.frame(
#'   year = rep(1:20, each = 4),
#'   x = rnorm(80)
#' )
#' trend <- sin(seq(0, 2, length.out = 20))
#' df$count <- rpois(80, exp(1 + 0.3 * df$x + trend[df$year]))
#'
#' fit <- tulpa(
#'   count ~ x,
#'   data = df,
#'   family = "poisson",
#'   temporal = temporal_rw2("year"),
#'   mode = "auto"
#' )
#' summary(fit)
#' }
#'
#' @export
temporal_rw2 <- function(time_var, group_var = NULL, cyclic = FALSE,
                         shared = NULL) {

  time_var <- .coerce_var_arg(time_var, "time_var", "~ time")
  if (is.null(time_var)) {
    stop("`time_var` must be a formula (~ time) or single character string",
         call. = FALSE)
  }
  group_var <- .coerce_var_arg(group_var, "group_var")

  if (isFALSE(shared)) .warn_nonshared("temporal effects")

 structure(
   list(
     type = "rw2",
     time_var = time_var,
     group_var = group_var,
     cyclic = cyclic,
     shared = shared,
     n_times = NULL,
     n_groups = NULL,
     time_index = NULL,
     group_index = NULL
   ),
   class = c("tulpa_temporal", "list")
 )
}


#' AR1 temporal structure (First-order Autoregressive)
#'
#' @description
#' Specify a first-order autoregressive temporal random effect.
#' AR1 models temporal correlation where each time point depends on
#' the previous one: `phi[t] = rho * phi[t-1] + epsilon[t]`.
#'
#' Unlike RW1/RW2, AR1 is a proper (stationary) model with an estimated
#' autocorrelation parameter rho.
#'
#' @inheritParams temporal_rw1
#' @param rho_prior Prior for the autocorrelation parameter. Default is
#'   `NULL` which uses a Uniform(-1, 1) prior. Can specify a Beta prior
#'   on (rho+1)/2 for more informative priors.
#'
#' @return A `tulpa_temporal` object
#'
#' @details
#' The AR1 process has marginal variance sigma^2 / (1 - rho^2) and
#' correlation between time points t and s of rho^|t-s|.
#'
#' The precision matrix is tridiagonal and full rank, so no constraints
#' are needed.
#'
#' @examples
#' # Create temporal AR1 specification
#' temporal_ar1("year")
#' temporal_ar1("year", group = "site")
#'
#' \donttest{
#' # AR1 temporal correlation
#' set.seed(127)
#' df <- data.frame(
#'   year = rep(1:20, each = 3),
#'   x = rnorm(60)
#' )
#' f <- as.numeric(arima.sim(list(ar = 0.7), 20, sd = 0.4))
#' df$count <- rpois(60, exp(1 + 0.3 * df$x + f[df$year]))
#'
#' fit <- tulpa(
#'   count ~ x,
#'   data = df,
#'   family = "poisson",
#'   temporal = temporal_ar1("year"),
#'   mode = "auto"
#' )
#' summary(fit)
#' }
#'
#' @export
temporal_ar1 <- function(time_var, group_var = NULL, shared = NULL,
                         rho_prior = NULL) {

  time_var <- .coerce_var_arg(time_var, "time_var", "~ time")
  if (is.null(time_var)) {
    stop("`time_var` must be a formula (~ time) or single character string",
         call. = FALSE)
  }
  group_var <- .coerce_var_arg(group_var, "group_var")
  if (isFALSE(shared)) .warn_nonshared("temporal effects")

  structure(
   list(
     type = "ar1",
     time_var = time_var,
     group_var = group_var,
     cyclic = FALSE,  # AR1 is not cyclic
     shared = shared,
     rho_prior = rho_prior,
     n_times = NULL,
     n_groups = NULL,
     time_index = NULL,
     group_index = NULL
   ),
   class = c("tulpa_temporal", "list")
 )
}


#' Print method for tulpa_temporal
#'
#' @param x A tulpa_temporal object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the temporal specification to the console.
#'
#' @export
print.tulpa_temporal <- function(x, ...) {
 cat("tulpa temporal specification\n")
 cat("============================\n\n")

 type_name <- switch(x$type,
   rw1 = "RW1 (First-order Random Walk)",
   rw2 = "RW2 (Second-order Random Walk)",
   ar1 = "AR1 (First-order Autoregressive)",
   x$type
 )

 cat("Type:", type_name, "\n")
 cat("Time variable:", x$time_var, "\n")

 if (!is.null(x$group_var)) {
   cat("Group variable:", x$group_var, "\n")
 }

 if (x$type %in% c("rw1", "rw2") && x$cyclic) {
   cat("Cyclic: Yes\n")
 }

 cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

 if (!is.null(x$n_times)) {
   cat("Time points:", x$n_times, "\n")
 }
 if (!is.null(x$n_groups) && x$n_groups > 1) {
   cat("Groups:", x$n_groups, "\n")
 }

 invisible(x)
}


#' Validate temporal specification against data
#'
#' @param temporal tulpa_temporal object
#' @param data Data frame
#'
#' @return Updated tulpa_temporal object with indices computed
#' @keywords internal
validate_temporal <- function(temporal, data) {
 if (is.null(temporal)) return(NULL)

 # Check time variable exists
 if (!(temporal$time_var %in% names(data))) {
   stop(sprintf("Temporal variable '%s' not found in data",
                temporal$time_var), call. = FALSE)
 }

 # Check group variable exists if specified
 if (!is.null(temporal$group_var)) {
   if (!(temporal$group_var %in% names(data))) {
     stop(sprintf("Temporal group variable '%s' not found in data",
                  temporal$group_var), call. = FALSE)
   }
 }

 # Get time values and create indices
 time_vals <- data[[temporal$time_var]]

 # Convert to factor to get consistent indexing
 if (is.factor(time_vals)) {
   time_factor <- time_vals
 } else {
   # Sort unique values to ensure temporal ordering
   unique_times <- sort(unique(time_vals))
   time_factor <- factor(time_vals, levels = unique_times)
 }

 temporal$n_times <- nlevels(time_factor)
 temporal$time_index <- as.integer(time_factor)
 temporal$time_levels <- levels(time_factor)

 # Check minimum time points for RW2
 if (temporal$type == "rw2" && temporal$n_times < 3) {
   stop("RW2 requires at least 3 time points", call. = FALSE)
 }

 # Handle grouping
 if (!is.null(temporal$group_var)) {
   group_vals <- data[[temporal$group_var]]
   group_factor <- as.factor(group_vals)
   temporal$n_groups <- nlevels(group_factor)
   temporal$group_index <- as.integer(group_factor)
   temporal$group_levels <- levels(group_factor)

   # Total temporal parameters = n_times * n_groups
   temporal$n_temporal_params <- temporal$n_times * temporal$n_groups
 } else {
   temporal$n_groups <- 1L
   temporal$group_index <- rep(1L, nrow(data))
   temporal$n_temporal_params <- temporal$n_times
 }

 temporal
}
