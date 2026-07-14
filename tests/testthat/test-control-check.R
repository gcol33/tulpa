# Typo protection on the control surface: every front door validates its
# `control` names against the canonical whitelist (.CONTROL_KEYS), and
# tulpa()'s reserved `...` errors instead of swallowing misspelled arguments.

test_that("tulpa() errors on stray ... arguments instead of ignoring them", {
  d <- data.frame(y = rpois(40, 3), x = rnorm(40))
  expect_error(
    tulpa(y ~ x, data = d, familly = "poisson"),
    "unknown argument.*familly"
  )
  expect_error(
    tulpa(y ~ x, data = d, family = "poisson", n_iter = 500L),
    "control = list()"
  )
})

test_that("front doors reject misspelled control knobs", {
  d <- data.frame(y = rpois(40, 3), x = rnorm(40))
  expect_error(
    tulpa(y ~ x, data = d, family = "poisson",
          control = list(adaptve_grid = TRUE)),
    "Unknown control knob.*adaptve_grid"
  )
  expect_error(
    tulpa_nested_laplace(y = d$y, n_trials = rep(1L, 40), X = cbind(1, d$x),
                         prior = list(type = "iid", sigma_grid = c(0.5, 1)),
                         family = "poisson",
                         control = list(max_itr = 10L)),
    "Unknown control knob.*max_itr"
  )
  expect_error(
    tulpa_ep(y ~ x, data = d, family = "poisson",
             control = list(sweeps = 10L)),
    "Unknown control knob.*sweeps"
  )
})

test_that("the joint front door hard-errors on the renamed diagnose_draws knob", {
  expect_error(
    tulpa_nested_laplace_joint(
      responses = list(a = list(y = rnorm(10), n_trials = rep(1L, 10),
                                X = matrix(1, 10, 1), family = "gaussian",
                                phi = 1.0)),
      prior = list(type = "iid", sigma_grid = c(0.5, 1)),
      control = list(diagnose_draws = 100L)),
    "renamed.*k_samples"
  )
})

test_that("valid control knobs still pass validation", {
  d <- data.frame(y = rpois(60, 3), x = rnorm(60))
  fit <- suppressMessages(tulpa(y ~ x, data = d, family = "poisson",
                                mode = "laplace",
                                control = list(max_iter = 50L, tol = 1e-6)))
  expect_s3_class(fit, "tulpa_fit")
})
