# Power-scaling prior/likelihood sensitivity (gcol33/tulpa C1). The CJS +
# gradient pipeline is validated end-to-end: a tight prior that conflicts with
# the data must be far more prior-sensitive than a weak prior on the same data
# and must flag a prior-data conflict; the likelihood component must be finite
# and informative; and the input guards must fire.

test_that("power-scaling flags a tight conflicting prior as more sensitive", {
  skip_on_cran()
  set.seed(7)
  n <- 250L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.4 + 0.8 * d$x))

  fit_weak  <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
                     beta_prior = list(mean = 0, sd = 5))
  fit_tight <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
                     beta_prior = list(mean = 0, sd = 0.15))

  s_weak  <- tulpa_powerscale_sensitivity(fit_weak,  d, prior = list(mean = 0, sd = 5))
  s_tight <- tulpa_powerscale_sensitivity(fit_tight, d, prior = list(mean = 0, sd = 0.15))

  # Likelihood component is finite and informative.
  expect_true(all(is.finite(s_weak$likelihood)))
  expect_true(all(s_weak$likelihood > 0))

  # The tight, conflicting prior is much more prior-sensitive on the slope.
  islope <- which(s_tight$variable == "x")
  expect_gt(s_tight$prior[islope], s_weak$prior[islope])

  # ... and a conflict is flagged for it.
  expect_true(any(grepl("conflict", s_tight$diagnosis)))
})

test_that("hyperparameter-prior sensitivity reads the stored per-draw log-prior", {
  skip_on_cran()
  set.seed(9)
  n <- 300L
  g <- factor(rep(1:12, length.out = n))
  b0 <- rnorm(12, 0, 0.8); b1 <- rnorm(12, 0, 0.5)
  d <- data.frame(x = rnorm(n), g = g)
  d$y <- rbinom(n, 1, plogis(0.2 + 0.6 * d$x +
                             b0[as.integer(g)] + b1[as.integer(g)] * d$x))

  # Random slopes route mode = "laplace" to the RE-covariance integrator,
  # whose mixture draws now carry the per-draw hyperparameter log-prior.
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "laplace")
  expect_false(is.null(fit$hyper_log_prior_draws))
  expect_length(fit$hyper_log_prior_draws, nrow(fit$draws))
  expect_true(all(is.finite(fit$hyper_log_prior_draws)))

  s <- tulpa_powerscale_sensitivity(fit, d)
  expect_true("hyperparameter" %in% names(s))
  expect_true(all(is.finite(s$hyperparameter)))
  expect_true(all(s$hyperparameter >= 0))

  # A plain fixed-effect Laplace fit has no stored hyper log-prior -> NA.
  d2 <- data.frame(x = rnorm(100)); d2$y <- rpois(100, exp(0.3 + 0.4 * d2$x))
  fit2 <- tulpa(y ~ x, data = d2, family = "poisson", mode = "laplace")
  s2 <- tulpa_powerscale_sensitivity(fit2, d2)
  expect_true(all(is.na(s2$hyperparameter)))
})

test_that("power-scaling: likelihood-only mode and input guards", {
  skip_on_cran()
  set.seed(8)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.3 + 0.5 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  s <- tulpa_powerscale_sensitivity(fit, d)         # no prior -> likelihood only
  expect_true(all(is.na(s$prior)))
  expect_true(all(is.finite(s$likelihood)))
  expect_equal(nrow(s), 2L)

  # Prior component without a prior spec errors.
  expect_error(tulpa_powerscale_sensitivity(fit, d, prior = list(mean = 0)), "sd")

  # Spatial fit rejected.
  sp <- structure(list(spatial = list(type = "icar"), family = "poisson"),
                  class = "tulpa_fit")
  expect_error(tulpa_powerscale_sensitivity(sp, d), "not supported")
})
