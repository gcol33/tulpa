# plot_diagnostics() had no draws-provenance gate: plot_rhat() / plot_ess() /
# geweke_test() message and return on a non-chain fit, but plot_diagnostics()
# went on to pick a trace parameter from colnames(fit$draws)[1], which is NULL
# on a draw-less fit, and errored "argument is of length zero" testing NULL
# %in% dimnames(...) (gcol33/tulpa#772). No test file referenced this
# function before.

test_that("plot_diagnostics() gates on a non-chain fit instead of erroring", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("patchwork")
  set.seed(1)
  d <- data.frame(x = rnorm(200)); d$y <- rpois(200, exp(0.3 + 0.4 * d$x))
  f_lap <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  expect_false(tulpa:::.tulpa_is_chain(f_lap))
  expect_message(res <- plot_diagnostics(f_lap), "not an MCMC chain")
  expect_null(res)
})

test_that("plot_diagnostics() still builds a panel for a chain fit", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("patchwork")
  set.seed(1)
  d <- data.frame(x = rnorm(200)); d$y <- rpois(200, exp(0.3 + 0.4 * d$x))
  f_hmc <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
                 control = list(n_iter = 200L, warmup = 100L, n_chains = 2L,
                                seed = 1L))
  expect_true(tulpa:::.tulpa_is_chain(f_hmc))
  expect_silent(res <- plot_diagnostics(f_hmc))
  expect_s3_class(res, "patchwork")
})
