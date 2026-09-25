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

# Under mode = "auto" the caller cannot know which backend the router picks, so
# `n_threads` -- read by every Laplace candidate -- was a hard error exactly
# when the model's terms sent auto to a sampler (gcol33/tulpa#911). Auto drops
# it with a message naming the backend; an explicit sampler mode still refuses.
test_that("mode = 'auto' drops n_threads when it resolves to a sampler", {
  ctl <- list(n_threads = 2L, n_iter = 10L)
  auto_hmc <- list(backend = "hmc", explicit = FALSE, requested = "auto")
  expect_message(out <- tulpa:::.auto_drop_unread_threads(ctl, auto_hmc),
                 "resolved to the sampler backend 'hmc'.*n_threads")
  expect_null(out$n_threads)
  expect_identical(out$n_iter, 10L)
  # An explicit mode keeps its knob (the sampler branch refuses it), and a
  # Laplace backend reads it.
  expect_identical(
    tulpa:::.auto_drop_unread_threads(ctl, list(backend = "hmc", explicit = TRUE)),
    ctl)
  expect_identical(
    tulpa:::.auto_drop_unread_threads(
      ctl, list(backend = "nested_laplace", explicit = FALSE)),
    ctl)
})

test_that("auto + n_threads fits a sampler-only field; hmc + n_threads refuses", {
  skip_if_not_slow()
  set.seed(1)
  tt <- sort(runif(15, 0, 50))
  d <- data.frame(t = rep(tt, each = 2))
  d$x <- rnorm(30)
  d$y <- d$x + sin(d$t / 5) + rnorm(30, 0, 0.3)
  expect_message(
    f <- tulpa(y ~ x, d, phi = 0.09, temporal = temporal_gp("t"),
               control = list(n_threads = 1, n_iter = 200, warmup = 100,
                              n_chains = 1, seed = 1)),
    "does not read `control\\$n_threads`")
  expect_identical(f$backend, "hmc")
  expect_error(
    tulpa(y ~ x, d, phi = 0.09, temporal = temporal_gp("t"), mode = "hmc",
          control = list(n_threads = 1)),
    "does not read `control\\$n_threads`")
})
