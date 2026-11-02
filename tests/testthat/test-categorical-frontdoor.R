# The categorical drivers as tulpa() reaches them: the control surface each
# driver reads, the multinomial reference class, and what summary() reports
# for a fit that has no outer grid.

skip_on_cran()

.cat_ordinal_data <- function(seed = 1L, n = 300L) {
  set.seed(seed)
  x <- rnorm(n)
  Fm <- plogis(outer(-0.8 * x, c(-1, 0.5, 2), "+"))
  P <- cbind(Fm, 1) - cbind(0, Fm)
  data.frame(y = ordered(apply(P, 1, function(p) sample.int(4L, 1L, prob = p))),
             x = x)
}

.cat_multinomial_data <- function(seed = 5L, n = 400L) {
  set.seed(seed)
  x <- rnorm(n)
  eta <- cbind(0.5 + 1.0 * x, -0.3 - 0.8 * x)
  P <- cbind(exp(eta), 1); P <- P / rowSums(P)
  data.frame(y = factor(c("a", "b", "c")[apply(P, 1, function(p)
    sample.int(3L, 1L, prob = p))]), x = x)
}

test_that("a knob the categorical driver does not read is refused at tulpa()", {
  skip_on_cran()
  d <- .cat_ordinal_data()
  expect_error(tulpa(y ~ x, d, family = "ordinal",
                     control = list(n_threads = 2)),
               "Unknown control knob.*n_threads")
  expect_error(tulpa(y ~ x, d, family = "ordinal", control = list(tol = 1e-4)),
               "Unknown control knob.*tol")
  dm <- .cat_multinomial_data()
  expect_error(tulpa(y ~ x, dm, family = "multinomial",
                     control = list(n_threads = 2)),
               "Unknown control knob.*n_threads")
  f <- tulpa(y ~ x, dm, family = "multinomial",
             control = list(n_draws = 50L, seed = 1L))
  expect_equal(nrow(f$draws), 50L)
})

test_that("baseline picks the multinomial reference class", {
  skip_on_cran()
  dm <- .cat_multinomial_data()
  f0 <- tulpa(y ~ x, dm, family = "multinomial")
  fa <- tulpa(y ~ x, dm, family = "multinomial", baseline = "a")
  expect_identical(f0$baseline, "c")
  expect_identical(fa$baseline, "a")
  expect_identical(fa$classes, c("b", "c", "a"))
  expect_true(all(startsWith(names(coef(fa)), "b:") |
                    startsWith(names(coef(fa)), "c:")))
  # The same model under another reference: the class probabilities agree up
  # to the ridge prior, which sits on different contrasts in the two fits.
  p0 <- fitted(f0)[, c("a", "b", "c")]
  pa <- fitted(fa)[, c("a", "b", "c")]
  expect_lt(max(abs(p0 - pa)), 0.01)
  # The contrast b vs a is the difference of the two fits' contrasts against
  # c, at the mode, again up to the prior.
  ba <- coef(f0)[c("b:(Intercept)", "b:x")] - coef(f0)[c("a:(Intercept)", "a:x")]
  expect_equal(unname(coef(fa)[c("b:(Intercept)", "b:x")]), unname(ba),
               tolerance = 0.05)
  fd <- tulpa_multinomial(y ~ x, dm, baseline = "a")
  expect_equal(coef(fd), coef(fa))
})

test_that("baseline is refused where it means nothing", {
  skip_on_cran()
  dm <- .cat_multinomial_data()
  expect_error(tulpa(y ~ x, dm, family = "multinomial", baseline = "z"),
               "one level of the response")
  expect_error(tulpa(y ~ x, .cat_ordinal_data(), family = "ordinal",
                     baseline = "1"), "no reference class")
  expect_error(tulpa(y ~ x, data.frame(y = rnorm(20), x = rnorm(20)),
                     phi = 1, baseline = "a"), "no reference class")
})

test_that("summary() of a categorical fit prints no all-NA band line", {
  skip_on_cran()
  out_o <- capture.output(print(summary(tulpa_ordinal(y ~ x,
                                                       .cat_ordinal_data()))))
  out_m <- capture.output(print(summary(tulpa_multinomial(
    y ~ x, .cat_multinomial_data()))))
  for (out in list(out_o, out_m)) {
    expect_false(any(grepl("NA \\(NA\\)", out)))
    expect_false(any(grepl("pareto_k", out)))
  }
})

test_that("a categorical mode-find that stops early says so", {
  skip_on_cran()
  dm <- .cat_multinomial_data()
  expect_warning(f <- tulpa(y ~ x, dm, family = "multinomial",
                            control = list(max_iter = 1L)),
                 "stopped before converging")
  expect_false(f$converged)
  expect_identical(f$convergence$kind, "mode")
  expect_silent(tulpa(y ~ x, dm, family = "multinomial"))
})
