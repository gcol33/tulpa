# AR1 rho prior (gcol33/tulpa#209): temporal_ar1(rho_prior=) places a Beta(a, b)
# prior on u = (rho + 1)/2, honoured on both the sampler and nested-Laplace paths.

test_that("temporal_ar1 validates rho_prior is a Beta prior", {
  expect_s3_class(temporal_ar1("t", rho_prior = prior_beta(5, 2)), "tulpa_temporal")
  expect_null(temporal_ar1("t")$rho_prior)
  expect_error(temporal_ar1("t", rho_prior = prior_normal(0, 1)), "Beta prior")
  expect_error(temporal_ar1("t", rho_prior = 0.5), "Beta prior")
})

test_that(".ar1_rho_beta_ab maps the prior to (a, b) shape params", {
  expect_identical(.ar1_rho_beta_ab(NULL), c(1, 1))           # default = Uniform
  expect_identical(.ar1_rho_beta_ab(prior_beta(5, 2)), c(5, 2))
})

test_that("nested-Laplace outer grid reweight honours the AR1 rho prior", {
  res <- list(theta_grid = cbind(tau = rep(1, 3), rho = c(0.0, 0.5, 0.9)),
              log_marginal = c(-1, -2, -3))

  # Beta(1, 1) is Uniform: no reweight.
  r0 <- .nl_apply_ar1_rho_prior(res, "ar1", list(rho_prior = prior_beta(1, 1)))
  expect_identical(r0$log_marginal, res$log_marginal)

  # A NULL prior on the block is also a no-op (default path).
  r_null <- .nl_apply_ar1_rho_prior(res, "ar1", list(rho_prior = NULL))
  expect_identical(r_null$log_marginal, res$log_marginal)

  # Beta(10, 1) density u^9 increases monotonically in u = (rho + 1)/2, so the
  # reweight is monotone increasing across the rho grid and largest at rho = 0.9.
  r1 <- .nl_apply_ar1_rho_prior(res, "ar1", list(rho_prior = prior_beta(10, 1)))
  d1 <- r1$log_marginal - res$log_marginal
  expect_true(all(diff(d1) > 0))
  expect_equal(which.max(d1), 3L)

  # The added term equals the exact Beta log-density (no logit Jacobian on the
  # natural-scale grid): (a-1) log(u) + (b-1) log(1-u).
  u <- 0.5 * (res$theta_grid[, "rho"] + 1)
  expect_equal(d1, 9 * log(u))

  # Non-AR1 blocks and grids without a rho axis are untouched.
  expect_identical(
    .nl_apply_ar1_rho_prior(res, "rw1", list(rho_prior = prior_beta(10, 1)))$log_marginal,
    res$log_marginal)
})

test_that("informative AR1 rho prior shifts the nested posterior toward the prior", {
  skip_on_cran()
  skip_if_not(nzchar(Sys.getenv("TULPA_SLOW_TESTS")), "slow recovery test")
  set.seed(11)
  Tn <- 60
  rho_true <- 0.3
  z <- numeric(Tn); z[1] <- rnorm(1)
  for (t in 2:Tn) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, 0.4)
  z <- z - mean(z)
  x <- rnorm(Tn)
  y <- rpois(Tn, exp(0.2 + 0.3 * x + z))
  d <- data.frame(y = y, x = x, t = seq_len(Tn))

  fit_flat <- tulpa(y ~ x + temporal_ar1(~ t), data = d, family = "poisson")
  # A prior concentrated on strong positive autocorrelation should pull the rho
  # posterior above the flat-prior fit on this short, weakly-identified series.
  fit_hi <- tulpa(y ~ x + temporal_ar1(~ t, rho_prior = prior_beta(12, 2)),
                  data = d, family = "poisson")
  rho_flat <- temporal_corr(fit_flat)["rho_ar1", "mean"]
  rho_hi   <- temporal_corr(fit_hi)["rho_ar1", "mean"]
  expect_true(is.finite(rho_flat) && is.finite(rho_hi))
  expect_gt(rho_hi, rho_flat)
})
