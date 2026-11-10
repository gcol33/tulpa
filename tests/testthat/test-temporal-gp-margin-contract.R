test_that("dense temporal GP kernels carry the requested relative nugget", {
  times <- c(0, .1, .4, 1, 1.5, 2)
  n <- length(times)
  sigma2 <- .4
  phi <- .7
  lo <- .01
  hi <- 10
  z <- sin(seq_len(n))
  index <- rep(seq_len(n), each = 2L)
  y <- cos(seq_along(index))
  X <- matrix(1, length(y), 1L)
  for (covariance in c("matern", "gaussian", "periodic")) {
    lag <- abs(outer(times, times, "-"))
    correlation <- switch(covariance,
      matern = (1 + sqrt(3) * lag / phi) * exp(-sqrt(3) * lag / phi),
      gaussian = exp(-(lag / phi)^2),
      periodic = exp(-2 * sin(pi * lag / 3)^2 / phi^2))
    covariance_matrix <- sigma2 * (correlation + diag(1e-6, n))
    L <- t(chol(covariance_matrix))
    field <- drop(L %*% z)
    for (coordinate in c("centered", "noncentered")) {
      spec <- list(type = "gp", parameterization = coordinate, n_times = n,
        n_groups = 1L, time_idx = as.integer(index), time_values = times,
        cov = covariance, nu = 1.5, period = 3, relative_nugget = 1e-6,
        phi_prior_lower = lo, phi_prior_upper = hi)
      q <- c(.2, log(sigma2), qlogis((phi - lo) / (hi - lo)),
        if (coordinate == "centered") field else z)
      r <- tulpa:::cpp_test_field_coordinates(y, X, temporal_spec = spec, point = q)
      rate <- -log(.01)
      hyper <- log(rate) - rate * sqrt(sigma2) - log(2) + .5 * log(sigma2) +
        log(phi - lo) + log(hi - phi) - log(hi - lo)
      field_lp <- if (coordinate == "noncentered") -.5 * sum(z^2) else
        -.5 * sum(z^2) - sum(log(diag(L))) - .5 * n * log(2 * pi)
      expected <- -.5 * .2^2 / 2.5^2 + hyper + field_lp +
        sum(dnorm(y, .2 + field[index], 1, log = TRUE))
      expect_equal(r$raw_lp, expected, tolerance = 1e-10)
      expect_equal(tail(r$stored, n), field, tolerance = 1e-12)
      expect_true(r$verified)
      expect_equal(r$gradient, r$numerical, tolerance = 3e-5)
    }
  }
})
