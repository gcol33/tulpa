# What the front door accepts, refuses, and says while doing it.
#
# Each of these was a documented argument the door did not honour, or a
# plausible mistake that surfaced an R internal naming no argument.

set.seed(4)
n  <- 150L
x  <- rnorm(n)
g  <- rep(1:15, each = 10L)
u  <- rnorm(15, 0, 0.5)
d  <- data.frame(
  x = x, g = factor(g),
  yb = rbinom(n, 1, plogis(0.2 + 0.5 * x + u[g])),
  yp = rpois(n, exp(0.3 + 0.4 * x + u[g]))
)

test_that("re_prior$hyperprior is accepted", {
  # gcol33/tulpa#667: documented in ?tulpa and read by the front door, and
  # missing from .RE_PRIOR_KEYS -- and tulpa_check_control() runs first, so the
  # documented key was rejected as unknown.
  expect_true("hyperprior" %in% tulpa:::.RE_PRIOR_KEYS)
  skip_on_cran()
  fit <- tulpa(yb ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "laplace", re_prior = list(hyperprior = "pc_lkj"))
  expect_s3_class(fit, "tulpa_fit")
})

test_that("control$re_cov is validated and honoured on any RE model", {
  # gcol33/tulpa#668: it was read only when a term carried a slope, so on a
  # (1 | g) model any value -- including a typo -- was accepted, had no effect,
  # and the fit silently conditioned at sigma_re = 1.
  expect_identical(tulpa:::.re_cov_method(list(), "nested"), "nested")
  expect_identical(tulpa:::.re_cov_method(list(re_cov = "gibbs"), "nested"),
                   "gibbs")
  expect_error(tulpa:::.re_cov_method(list(re_cov = "bogus"), "nested"))

  skip_on_cran()
  expect_error(
    tulpa(yb ~ x + (1 | g), data = d, family = "binomial", mode = "laplace",
          control = list(re_cov = "bogus")))
  fit <- tulpa(yb ~ x + (1 | g), data = d, family = "binomial",
               mode = "laplace", control = list(re_cov = "gibbs"))
  expect_identical(fit$backend, "re_cov_gibbs")
})

test_that("sigma_re warns wherever the fit determines the scale itself", {
  # gcol33/tulpa#669: ?tulpa promised the warning for eb / re_cov_* / gibbs /
  # agq; only the EB and random-slope routes gave it, and agq / gibbs / hmc
  # dropped the value in silence.
  est <- tulpa:::.re_scale_estimating_backends()
  for (b in c("eb", "agq", "gibbs", "re_cov_nested", "re_cov_gibbs", "hmc")) {
    expect_true(b %in% est, info = b)
  }
  expect_false("laplace" %in% est)

  skip_on_cran()
  expect_warning(
    tulpa(yb ~ x + (1 | g), data = d, family = "binomial", mode = "agq",
          sigma_re = 0.7),
    "ignored for mode = 'agq'")
})

test_that("n_trials is read the same way on every door", {
  # gcol33/tulpa#677: a scalar errored at the C++ boundary on `laplace` and was
  # recycled on `mala` / `imh_laplace`, and n_trials with a non-binomial family
  # was read by nothing -- no signal for a user who meant a binomial.
  skip_on_cran()
  f1 <- tulpa(yb ~ x, data = d, family = "binomial", n_trials = 1L,
              mode = "laplace")
  expect_s3_class(f1, "tulpa_fit")

  expect_error(
    tulpa(yp ~ x, data = d, family = "poisson", n_trials = rep(5L, n),
          mode = "laplace"),
    "not read by family")
  expect_error(
    tulpa(yb ~ x, data = d, family = "binomial", n_trials = rep(1L, 5L),
          mode = "laplace"),
    "length 1 .recycled. or nrow")
})

test_that("front-door misuse names the argument, not an R internal", {
  # gcol33/tulpa#679
  skip_on_cran()
  expect_error(tulpa(yb ~ x, data = d, family = stats::binomial(),
                     mode = "laplace"), NA)
  expect_error(tulpa(yp ~ x, data = d, family = "poisson", mode = NULL),
               "`mode` must be a single string")
  expect_error(tulpa(~ x, data = d, family = "poisson", mode = "laplace"),
               "`formula` must be two-sided")
  expect_error(tulpa(yb ~ x, data = d[0, ], family = "binomial",
                     mode = "laplace"),
               "`data` has no rows")
  expect_error(tulpa(yb ~ x, data = d, family = c("a", "b"), mode = "laplace"),
               "`family` must be a single family name")
})

test_that("a family object maps to the engine's own family name", {
  expect_identical(tulpa:::.family_object_to_name(stats::binomial()), "binomial")
  expect_identical(tulpa:::.family_object_to_name(stats::poisson()), "poisson")
  expect_identical(tulpa:::.family_object_to_name(stats::gaussian()), "gaussian")
  # A non-canonical link rides the name in the `<base>_<link>` convention rather
  # than being silently fitted at the canonical one.
  expect_identical(tulpa:::.family_object_to_name(stats::binomial("probit")),
                   "binomial_probit")
})

test_that("the ziformula RE guard walks the AST", {
  # gcol33/tulpa#680: a regex on deparsed code false-positives on a `|` inside a
  # string literal and is papered over by any() when the deparse wraps.
  dd <- data.frame(y = rpois(20, 2), x = rnorm(20), g = gl(4, 5),
                   lab = rep("a|b", 20), stringsAsFactors = FALSE)
  expect_error(tulpa:::.zi_design(~ 1 + (1 | g), dd, 20L),
               "fixed effects only")
  # A `|` that is not a bar term is not a random effect.
  expect_silent(tulpa:::.zi_design(~ x, dd, 20L))
})

test_that("the temporal field constructor warns on shared = FALSE like its twin", {
  # gcol33/tulpa#702: .warn_nonshared() exists because the same input warned or
  # stayed silent depending only on which constructor it went to.
  expect_warning(tulpa:::.temporal_field_spec(~ 1 || t, structure = "rw1",
                                              shared = FALSE),
                 "Non-shared")
})

test_that("every Matern constructor checks nu at construction", {
  # gcol33/tulpa#701: spatial_gp() validated it and spatial_multiscale() stored
  # anything, deferring the error into the fit.
  expect_error(spatial_gp(~ lon + lat, cov = "matern", nu = 7), "nu in")
  expect_error(spatial_multiscale(~ lon + lat, cov = "matern", nu = 7), "nu in")
  expect_s3_class(spatial_multiscale(~ lon + lat, cov = "matern", nu = 2.5),
                  "tulpa_spatial")
})

test_that("an NNGP spec carries the amplitude anchors the door accepts", {
  # gcol33/tulpa#700: they were validated and stored on the HSGP branch only, so
  # an NNGP user's anchors were neither used nor refused, and the sampler spec
  # hardcoded a second default that disagreed with the engine's.
  sp <- spatial_gp(~ lon + lat, sigma_prior_U = 3, sigma_prior_alpha = 0.02)
  expect_equal(sp$sigma2_prior_U, 3)
  expect_equal(sp$sigma2_prior_alpha, 0.02)
  expect_error(spatial_gp(~ lon + lat, sigma_prior_U = -1))

  # The default is the engine's own (1, 0.01) -- one anchor, one default.
  d0 <- spatial_gp(~ lon + lat)
  expect_equal(d0$sigma2_prior_U, 1)
  expect_equal(d0$sigma2_prior_alpha, 0.01)
})
