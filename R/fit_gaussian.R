#' Fit a Gaussian linear model via tulpa's generic engine
#'
#' Proof-of-concept function demonstrating the tulpa generic interface.
#' Fits y ~ Normal(X * beta, sigma) with HMC sampling.
#'
#' @param formula A formula (e.g., y ~ x1 + x2)
#' @param data A data frame
#' @param sigma_beta Prior SD for regression coefficients (default 10)
#' @param iter Total iterations (default 2000)
#' @param warmup Warmup iterations (default 1000)
#' @param step_size HMC step size (default 0.05)
#' @param n_leapfrog Number of leapfrog steps (default 10)
#' @param seed Random seed (default 42)
#'
#' @return A list with draws matrix, posterior means, and metadata
#' @export
tulpa_gaussian <- function(formula, data, sigma_beta = 10, iter = 2000,
                           warmup = 1000, step_size = 0.05, n_leapfrog = 10,
                           seed = 42) {
  # Parse formula
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, data)

  if (!is.numeric(y)) stop("Response must be numeric")
  if (length(y) != nrow(X)) stop("Response and design matrix dimension mismatch")

  # Fit via C++
  fit <- cpp_tulpa_fit_gaussian(
    y_r = y,
    X_r = X,
    sigma_beta = sigma_beta,
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    step_size = step_size,
    n_leapfrog = as.integer(n_leapfrog),
    seed = as.integer(seed)
  )

  # Add sigma (not log_sigma) to means
  fit$sigma <- exp(fit$means["log_sigma"])
  fit$formula <- formula
  fit$N <- length(y)
  fit$p <- ncol(X)

  class(fit) <- "tulpa_fit"
  fit
}

#' @export
print.tulpa_fit <- function(x, ...) {
  cat("tulpa fit\n")
  cat("  N =", x$N, ", p =", x$p, "\n")
  cat("  Samples:", x$n_samples, "\n\n")
  cat("Posterior means:\n")
  print(round(x$means, 4))
  cat("\nsigma:", round(x$sigma, 4), "\n")
  invisible(x)
}
