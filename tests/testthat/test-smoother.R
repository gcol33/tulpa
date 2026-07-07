# Covariate smoothers s(x) (gcol33/tulpa C2): an RW2/RW1 GMRF over the binned
# covariate, integrated by the nested-Laplace temporal kernels. Recovery anchor:
# the fitted smooth must track a known nonlinear function at the nodes.

test_that("the parser strips s() from the fixed formula and records the call", {
  p <- tulpa_parse_formula(y ~ x1 + s(elev) + s(prec, k = 15) + (1 | g))
  expect_equal(p$n_smooth_terms, 2L)
  fx <- attr(stats::terms(p$fixed_formula), "term.labels")
  expect_equal(fx, "x1")
  expect_equal(p$n_re_terms, 1L)
})

test_that(".smooth_block_from_call bins and validates", {
  d <- data.frame(x = seq(-1, 1, length.out = 200), z = letters[1:4])

  b <- tulpa:::.smooth_block_from_call(quote(s(x)), d, globalenv())
  expect_equal(b$block$type, "rw2")
  expect_equal(b$block$n_times, 30L)
  expect_equal(length(b$block$temporal_idx), 200L)
  expect_true(all(b$block$temporal_idx >= 1L & b$block$temporal_idx <= 30L))
  expect_equal(length(b$meta$nodes), 30L)

  # Few unique values: nodes at the sorted unique values.
  d2 <- data.frame(x = rep(1:5, 40))
  b2 <- tulpa:::.smooth_block_from_call(quote(s(x, k = 30)), d2, globalenv())
  expect_equal(b2$block$n_times, 5L)
  expect_equal(b2$meta$nodes, 1:5)

  # Options.
  b3 <- tulpa:::.smooth_block_from_call(quote(s(x, k = 10, structure = "rw1")),
                                        d, globalenv())
  expect_equal(b3$block$type, "rw1")
  expect_equal(b3$block$n_times, 10L)

  expect_error(tulpa:::.smooth_block_from_call(quote(s(z)), d, globalenv()),
               "numeric")
  expect_error(tulpa:::.smooth_block_from_call(quote(s(x, k = 2)), d, globalenv()),
               "k")
  expect_error(
    tulpa:::.smooth_block_from_call(quote(s(x, structure = "spline")), d,
                                    globalenv()),
    "rw")
})

test_that("s(x) routes through nested Laplace and recovers a sin curve", {
  skip_on_cran()
  set.seed(71)
  n <- 500L
  d <- data.frame(x = runif(n, -2, 2))
  f <- function(x) sin(2 * x)
  d$y <- rpois(n, exp(0.4 + f(d$x)))

  fit <- tulpa(y ~ s(x, k = 25), data = d, family = "poisson")
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "nested_laplace")
  expect_match(fit$selection_reason, "smoother")

  sm <- smooth_effects(fit)
  expect_s3_class(sm, "data.frame")
  expect_equal(nrow(sm), 25L)
  expect_true(all(is.finite(sm$estimate)))

  # The RW2 smooth is identified up to the level absorbed by the intercept;
  # compare centered shapes.
  f_true <- f(sm$x) - mean(f(sm$x))
  f_hat  <- sm$estimate - mean(sm$estimate)
  expect_gt(stats::cor(f_hat, f_true), 0.95)
  expect_lt(mean(abs(f_hat - f_true)), 0.25)
})

test_that("s(x) works alongside fixed effects for a gaussian response", {
  skip_on_cran()
  set.seed(73)
  n <- 400L
  d <- data.frame(x = runif(n, 0, 1), z = rnorm(n))
  d$y <- 0.5 + 0.8 * d$z + cos(2 * pi * d$x) + rnorm(n, 0, 0.5)

  fit <- tulpa(y ~ z + s(x, k = 20), data = d, family = "gaussian", phi = 0.25)
  cf <- coef(fit)
  expect_lt(abs(cf[["z"]] - 0.8), 0.1)

  sm <- smooth_effects(fit)
  f_true <- cos(2 * pi * sm$x) - mean(cos(2 * pi * sm$x))
  expect_gt(stats::cor(sm$estimate - mean(sm$estimate), f_true), 0.95)
})

test_that("two smoothers fit through the joint multi-block path", {
  skip_on_cran()
  set.seed(79)
  n <- 500L
  d <- data.frame(x1 = runif(n, -2, 2), x2 = runif(n, -2, 2))
  f1 <- function(x) sin(2 * x); f2 <- function(x) 0.5 * x^2 - 0.7
  d$y <- rpois(n, exp(0.3 + f1(d$x1) + f2(d$x2)))

  fit <- tulpa(y ~ s(x1, k = 15) + s(x2, k = 15), data = d, family = "poisson")
  expect_equal(fit$backend, "nested_laplace")

  s1 <- smooth_effects(fit, 1L)
  s2 <- smooth_effects(fit, 2L)
  expect_gt(stats::cor(s1$estimate, f1(s1$x)), 0.9)
  expect_gt(stats::cor(s2$estimate, f2(s2$x)), 0.9)
})

test_that("unsupported smoother combinations error clearly", {
  d <- data.frame(x = runif(60), time = rep(1:10, 6),
                  lon = rnorm(60), lat = rnorm(60))
  d$y <- rbinom(60, 1, 0.5)

  # ModelData sampler backends do not thread smoother blocks.
  expect_error(
    tulpa(y ~ s(x), data = d, family = "binomial", mode = "vi"),
    "ModelData")
  # Continuous spatial fields run their own integrator.
  expect_error(
    tulpa(y ~ s(x), data = d, family = "binomial",
          spatial = spatial_gp(~ lon + lat)),
    "areal")

  # smooth_effects on a smooth-free fit.
  plain <- structure(list(family = "poisson"), class = "tulpa_fit")
  expect_error(smooth_effects(plain), "no s\\(")
})
