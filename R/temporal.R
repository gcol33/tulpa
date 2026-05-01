#' Temporal structure specifications for tulpa
#'
#' @description
#' Functions to specify temporal random effects for tulpa models.
#' Temporal effects are shared between processes by default,
#' which helps prevent bias from temporally-structured
#' unmeasured confounders.
#'
#' @name tulpa_temporal
NULL

#' RW1 temporal structure (First-order Random Walk)
#'
#' @description
#' Specify a first-order random walk temporal random effect.
#' RW1 assumes that the difference between consecutive time points is
#' normally distributed: `phi[t] - phi[t-1] ~ N(0, sigma^2)`.
#'
#' This is an intrinsic model (improper prior) that captures smooth
#' temporal trends. It is equivalent to an ICAR model on a 1D chain.
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`).
#' @param group_var Optional name of grouping variable for panel data.
#'   If provided, separate random walks are estimated for each group.
#' @param cyclic Logical; if TRUE, the random walk wraps around (first and
#'   last time points are neighbors). Useful for seasonal data.
#' @param shared Logical; if TRUE (default), temporal effect enters both
#'   all process linear predictors identically.
#'
#' @return A `tulpa_temporal` object
#'
#' @details
#' The RW1 precision matrix Q has the form (for T time points):
#' \deqn{Q[t,t] = 1 \text{ for } t = 1, T}
#' \deqn{Q[t,t] = 2 \text{ for } 1 < t < T}
#' \deqn{Q[t,t-1] = Q[t-1,t] = -1}
#'
#' This is a rank-deficient matrix (rank T-1), so a sum-to-zero constraint
#' is applied for identifiability.
#'
#' @examples
#' # Create temporal RW1 specification
#' temporal_rw1("year")
#' temporal_rw1("month", cyclic = TRUE)
#'
#' \donttest{
#' # Simple temporal trend
#' set.seed(123)
#' df <- data.frame(
#'   year = rep(2010:2021, each = 4),
#'   x = rnorm(48),
#'   count = rpois(48, lambda = 20),
#'   effort = rgamma(48, shape = 5, rate = 1)
#' )
#'
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   temporal = temporal_rw1("year"),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Panel data: separate trends per site
#' set.seed(124)
#' df_panel <- data.frame(
#'   year = rep(2015:2026, each = 5),
#'   site = rep(paste0("site", 1:5), times = 12),
#'   x = rnorm(60),
#'   count = rpois(60, lambda = 15),
#'   effort = rgamma(60, shape = 3, rate = 1)
#' )
#'
#' fit2 <- tulpa(
#'   count | effort ~ x + (1 | site),
#'   data = df_panel,
#'   family = tulpa_poisson_gamma(),
#'   temporal = temporal_rw1("year", group_var = "site"),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Cyclic for monthly seasonal data
#' set.seed(125)
#' df_monthly <- data.frame(
#'   month = rep(1:12, times = 4),
#'   x = rnorm(48),
#'   count = rpois(48, lambda = 10),
#'   effort = rgamma(48, shape = 2, rate = 1)
#' )
#'
#' fit3 <- tulpa(
#'   count | effort ~ x,
#'   data = df_monthly,
#'   family = tulpa_poisson_gamma(),
#'   temporal = temporal_rw1("month", cyclic = TRUE),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#' }
#'
#' @export
