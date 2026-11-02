# A fixed-effect coefficient's scale is set by the units of its covariate. The
# NUTS chains disperse their starts on the linear-predictor scale of each
# coefficient and adapt the metric without an absolute floor, so a design
# carrying a calendar-year column samples like its standardized version.

.nds_data <- function() {
  set.seed(3)
  year <- rep(1:25, each = 10); x <- rnorm(250)
  data.frame(year = year, x = x, y = rpois(250, exp(0.3 + 0.5 * x)))
}

test_that("an unscaled interaction design gives four moving, agreeing chains", {
  skip_on_cran()
  d <- .nds_data()
  fit <- tulpa(y ~ x * year, data = d, family = "poisson", mode = "exact",
               control = list(n_iter = 600, warmup = 300, n_chains = 4,
                              seed = 1))
  expect_true(isTRUE(fit$convergence$ok))
  per_chain_sd <- tapply(seq_len(nrow(fit$draws)), fit$chain_id, function(i) {
    min(apply(fit$draws[i, , drop = FALSE], 2, stats::sd))
  })
  expect_true(all(per_chain_sd > 0))
  expect_equal(sum(fit$divergent), 0)
  # Against the Laplace fit of the same posterior (a Poisson GLM at n = 250 is
  # close to Gaussian): the posterior means agree to a fraction of a posterior SD.
  lap <- tulpa(y ~ x * year, data = d, family = "poisson", mode = "laplace")
  sds <- apply(fit$draws, 2, stats::sd)
  expect_lt(max(abs(colMeans(fit$draws) - coef(lap)) / sds), 0.25)
})

test_that("a coefficient on an elevation-in-metres column mixes", {
  skip_on_cran()
  set.seed(11)
  n <- 300; x <- rnorm(n); elev <- runif(n, 200, 2500)
  d <- data.frame(x = x, elev = elev,
                  y = rpois(n, exp(0.3 + 0.5 * x + 0.0005 * (elev - 1300))))
  fit <- tulpa(y ~ x + elev, data = d, family = "poisson", mode = "hmc",
               control = list(n_iter = 1000, warmup = 500, n_chains = 4,
                              seed = 1))
  expect_true(isTRUE(fit$convergence$ok))
  # The adapted metric reaches the coefficient's own scale, so trajectories
  # stay short: a metric floored at an absolute variance ran them to the
  # maximum tree depth.
  expect_lt(mean(fit$treedepth), 5)
})
