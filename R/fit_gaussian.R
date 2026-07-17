#' Fit a Gaussian linear model via tulpa's generic engine
#'
#' Proof-of-concept function demonstrating the tulpa generic interface.
#' Fits y ~ Normal(X * beta, sigma) with HMC sampling.
#'
#' @param formula A formula (e.g., y ~ x1 + x2)
#' @param data A data frame
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on every coefficient with SD `sd`
#'   (default `list(mean = 0, sd = 10)`).
#' @param control List of numerical / sampler knobs: `iter` (total iterations,
#'   default 2000), `warmup` (default 1000), `step_size` (HMC step size, default
#'   0.05), `n_leapfrog` (default 10), `seed` (`NULL` draws from the session
#'   RNG).
#'
#' @return A list with draws matrix, posterior means, and metadata
#' @export
tulpa_gaussian <- function(formula, data,
                           beta_prior = list(mean = 0, sd = 10),
                           control = list()) {
  .check_control(control, .CONTROL_KEYS$gaussian, "tulpa_gaussian")
  sigma_beta <- .beta_prior_ridge_sd(beta_prior, default_sd = 10)
  iter       <- as.integer(control$iter %||% 2000L)
  warmup     <- as.integer(control$warmup %||% 1000L)
  step_size  <- control$step_size %||% 0.05
  n_leapfrog <- as.integer(control$n_leapfrog %||% 10L)
  seed       <- control$seed

  # Parse formula
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, data)

  if (!is.numeric(y)) {
    stop("`formula` must name a numeric response for tulpa_gaussian().",
         call. = FALSE)
  }
  if (length(y) != nrow(X)) {
    stop(sprintf(
      "length(y) (%d) must equal nrow(model.matrix) (%d).",
      length(y), nrow(X)), call. = FALSE)
  }

  # Fit via C++
  fit <- cpp_tulpa_fit_gaussian(
    y_r = y,
    X_r = X,
    sigma_beta = sigma_beta,
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    step_size = step_size,
    n_leapfrog = as.integer(n_leapfrog),
    seed = as.integer(seed %||% sample.int(.Machine$integer.max, 1L))
  )

  # Add sigma (not log_sigma) to means
  fit$sigma <- exp(fit$means["log_sigma"])
  fit$formula <- formula
  fit$N <- length(y)
  fit$p <- ncol(X)

  class(fit) <- "tulpa_fit"
  fit
}

