# Chain-diagnostic plots and the warnings around an unconverged chain.

.dpw_short_fit <- function(mode) {
  set.seed(1)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.3 + 0.5 * d$x))
  fit <- NULL
  w <- character(0)
  withCallingHandlers(
    fit <- tulpa(y ~ x, data = d, family = "poisson", mode = mode,
                 control = c(list(n_iter = 40, warmup = 20, seed = 1),
                             if (mode == "hmc") list(n_chains = 2))),
    warning = function(m) {
      w <<- c(w, conditionMessage(m)); invokeRestart("muffleWarning")
    })
  list(fit = fit, warnings = w)
}

test_that("the non-convergence warning offers NUTS only to a non-NUTS chain", {
  skip_on_cran()
  h <- .dpw_short_fit("hmc")
  msg <- grep("has not converged", h$warnings, value = TRUE)
  expect_length(msg, 1L)
  expect_false(grepl("mode = 'hmc'", msg))
  m <- .dpw_short_fit("mala")
  msg <- grep("has not converged", m$warnings, value = TRUE)
  expect_length(msg, 1L)
  expect_true(grepl("or use mode = 'hmc'", msg, fixed = TRUE))
})

test_that("criteria warn on a chain recorded as not converged", {
  skip_on_cran()
  h <- .dpw_short_fit("hmc")
  fit <- h$fit
  expect_false(isTRUE(fit$convergence$ok))
  expect_warning(pointwise_loglik(fit), class = "tulpa_unconverged_criterion")
  ok <- fit
  ok$convergence$ok <- TRUE
  w <- character(0)
  withCallingHandlers(
    tab <- compare_models(a = fit, b = ok, criterion = "waic"),
    warning = function(m) {
      w <<- c(w, conditionMessage(m)); invokeRestart("muffleWarning")
    })
  expect_length(grep("compare_models", w), 1L)
  expect_true(grepl("'a'", w[grep("compare_models", w)]))
  expect_false(grepl("'b'", w[grep("compare_models", w)]))
  expect_length(grep("pointwise_loglik", w), 0L)
  expect_equal(nrow(tab), 2L)
})

test_that("the default divergence / pairs parameters are scalars, not field values", {
  skip_on_cran()
  set.seed(1)
  K <- 9L; W <- matrix(0, K, K)
  for (i in 1:(K - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
  d <- data.frame(x = rnorm(180), region = factor(sample(1:K, 180, TRUE)))
  d$y <- rpois(180, exp(0.2 + 0.4 * d$x))
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(region), data = d, family = "poisson",
    spatial = list(type = "icar", adjacency = W), mode = "hmc",
    control = list(n_iter = 100, warmup = 50, n_chains = 2, seed = 1)))
  dp <- tulpa:::.plot_default_pars(fit, fit$draws)
  expect_identical(dp$fixed, c("(Intercept)", "x"))
  expect_false(any(grepl("\\[", dp$hyper)))
  expect_true("log_tau_spatial" %in% dp$hyper)
})

test_that("plot_energy draws the marginal energy centred within each chain", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")
  h <- .dpw_short_fit("hmc")
  p <- plot_energy(h$fit)
  dat <- p$data
  marg <- dat$value[dat$type == "Marginal E"]
  expect_lt(abs(mean(marg)), 1e-8)
  expect_equal(length(marg), length(h$fit$energy))
})
