# Typo protection on the control surface: every front door validates its
# `control` names against the canonical whitelist (.CONTROL_KEYS), and
# tulpa()'s reserved `...` errors instead of swallowing misspelled arguments.

test_that("tulpa() errors on stray ... arguments instead of ignoring them", {
  skip_on_cran()
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

# mode = "laplace" built its tulpa_laplace() argument list with no `control`
# field, so the numerical knobs passed tulpa()'s union check and were dropped,
# and every knob another backend reads was accepted in silence
# (gcol33/tulpa#870).
test_that("mode = 'laplace' forwards max_iter / tol / n_threads to tulpa_laplace()", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(30))
  d$y <- rpois(30, exp(1 + d$x))
  f1 <- tulpa(y ~ x, d, family = "poisson", mode = "laplace",
              control = list(max_iter = 1L))
  # The front door applies its default fixed-effect prior; the reference
  # solve takes the same one so the comparison is of max_iter alone.
  ref <- tulpa_laplace(d$y, NULL, cbind(1, d$x), family = "poisson",
                       max_iter = 1L, beta_prior = f1$beta_prior)
  expect_false(isTRUE(as.logical(f1$converged)))
  expect_equal(unname(coef(f1)), unname(ref$mode[1:2]), tolerance = 1e-8)

  f_def <- tulpa(y ~ x, d, family = "poisson", mode = "laplace")
  expect_true(isTRUE(as.logical(f_def$converged)))
  expect_false(isTRUE(all.equal(unname(coef(f1)), unname(coef(f_def)))))
})

test_that("mode = 'laplace' refuses control knobs tulpa_laplace() does not read", {
  set.seed(1)
  d <- data.frame(x = rnorm(30))
  d$y <- rpois(30, exp(1 + d$x))
  expect_error(
    tulpa(y ~ x, d, family = "poisson", mode = "laplace",
          control = list(n_iter = 5, adaptive_grid = TRUE, adapt_delta = 0.99)),
    "Unknown control knob.*mode = 'laplace'.*n_iter.*adaptive_grid.*adapt_delta")
  expect_error(
    tulpa(y ~ x, d, family = "poisson", mode = "laplace",
          control = list(checkpoint = list(path = tempfile()))),
    "Unknown control knob.*checkpoint")
})
