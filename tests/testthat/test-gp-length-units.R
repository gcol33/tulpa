# A GP field fitted on standardized coordinates (scale_coords = TRUE, the
# default) reports its lengthscale in the units of the coordinates as supplied
# at every reporting door, the units spatial_range() already reads in.

skip_on_cran()

test_that("GP lengthscale is reported in coordinate units at every door", {
  skip_on_cran()
  set.seed(3)
  n <- 150
  d <- data.frame(lon = runif(n, 0, 10), lat = runif(n, 0, 10), x = rnorm(n))
  D <- as.matrix(dist(d[, c("lon", "lat")]))
  w <- as.numeric(t(chol(exp(-D / 3) + diag(1e-8, n))) %*% rnorm(n))
  d$y <- rpois(n, exp(0.5 + 0.4 * d$x + 0.8 * w))
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat), mode = "nested_laplace",
               control = list(progress = FALSE))
  k <- tulpa:::.coord_scale(fit$spatial)
  expect_gt(k, 2)                     # the coordinates were standardized

  hp <- summary(fit)$hyperparameters
  row <- hp[hp$parameter == "phi_gp", ]
  expect_equal(row$mean, unname(fit$theta_mean[["phi_gp"]]) * k)
  expect_equal(row$ci_hi, unname(fit$theta_ci_hi[["phi_gp"]]) * k)
  # The sigma2 axis carries no length and is untouched.
  expect_equal(hp$mean[hp$parameter == "sigma2"],
               unname(fit$theta_mean[["sigma2"]]))

  g <- tulpa_outer_grid(fit)
  expect_equal(unname(g$nodes[, "phi_gp"]), unname(fit$theta_grid[, "phi_gp"]) * k)
  expect_equal(unname(g$nodes[, "sigma2"]), unname(fit$theta_grid[, "sigma2"]))

  set.seed(1)
  hd <- tulpa_hyper_draws(fit, n = 4000)
  expect_true(all(hd[, "phi_gp"] >= min(g$nodes[, "phi_gp"]) / 2))
  expect_equal(stats::median(hd[, "phi_gp"]), row$median, tolerance = 0.1)

  # spatial_range()'s range is three lengthscales, in the same units.
  sr <- spatial_range(fit)
  expect_equal(sr["range", "mean"], 3 * row$mean, tolerance = 1e-8)
})
