test_that("HSGP screening preserves the unscreened posterior on a wide basis", {
  skip_on_cran()
  set.seed(11L)
  n <- 250L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  D <- as.matrix(dist(d[, 1:2]))
  K <- .5 * exp(-D / .05) + .6 * exp(-D / .5)
  w <- as.numeric(t(chol(K + diag(1e-6, n))) %*% rnorm(n))
  d$y <- rpois(n, exp(.3 + .4 * d$x + w))
  fit <- function(prune) suppressWarnings(tulpa(y ~ x, data = d,
    family = "poisson", spatial = spatial_gp(~lon + lat, approx = "hsgp",
      m = 26L, c = 1.6), control = list(progress = FALSE, prune = prune)))
  screened <- fit(TRUE)
  full <- fit(FALSE)
  expect_lt(abs(screened$log_evidence - full$log_evidence), .001)
  quantiles <- function(f) log(cbind(f$theta_ci_lo, f$theta_median, f$theta_ci_hi))
  u <- log(full$theta_grid)
  sd <- sqrt(colSums(full$weights * sweep(u, 2L, colSums(full$weights * u))^2))
  expect_lt(max(abs((quantiles(screened) - quantiles(full)) / sd)), .01)
})
