test_that("generic LikelihoodSpec path runs through production NUTS", {
  skip_on_cran()

  set.seed(11)
  n <- 12L
  x <- seq(-1, 1, length.out = n)
  X <- cbind(1, x)
  y <- 0.4 + 0.8 * x + rnorm(n, sd = 0.2)

  fit <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y,
    X_r = X,
    n_iter = 24L,
    n_warmup = 12L,
    max_treedepth = 3L,
    adapt_delta = 0.8,
    seed = 11L,
    verbose = FALSE
  )

  expect_equal(fit$sampler, "nuts")
  expect_equal(fit$n_params, 3L)
  expect_equal(ncol(fit$draws), 3L)
  expect_equal(nrow(fit$draws), 12L)
  expect_true(all(is.finite(fit$draws)))
  expect_true(all(is.finite(fit$log_prob)))
})
