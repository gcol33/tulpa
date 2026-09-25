# Post-fit method gaps (gcol33/tulpa#899): confint() validates `level` / `parm`
# and reads a sampler's hyperparameters off its draws, durbin_watson() takes a
# fit as moran_i() does, and VarCorr() reports a gaussian fit's residual SD.

.sampler_like_fit <- function() {
  set.seed(1)
  S <- 400L
  dr <- cbind(`(Intercept)` = rnorm(S, 1, 0.1), x = rnorm(S, 2, 0.2),
              log_sigma_re = rnorm(S, -0.5, 0.3))
  structure(list(draws = dr, draws_kind = "chain", n_fixed = 2L,
                 fixed_names = c("(Intercept)", "x"),
                 param_names = colnames(dr)),
            class = "tulpa_fit")
}

test_that("confint() refuses a level outside (0, 1) naming it", {
  fit <- .sampler_like_fit()
  for (bad in list(1.5, 0, 1, -0.2, NA, "0.9", c(0.9, 0.95))) {
    expect_error(confint(fit, level = bad), "`level` must be a single number",
                 info = format(bad))
  }
  expect_error(summary(fit, level = 1.5), "`level`")
  expect_error(tidy(fit, conf.level = 0), "`conf.level`")
  expect_equal(colnames(confint(fit, level = 0.9)), c("5 %", "95 %"))
})

test_that("confint() names an unknown parm instead of subscript-out-of-bounds", {
  fit <- .sampler_like_fit()
  expect_error(confint(fit, parm = "zz"), "Unknown `parm`: zz")
  expect_error(confint(fit, parm = 3), "`parm` indices")
  expect_error(confint(fit, parm = 1.5), "`parm` indices")
  expect_error(confint(fit, parm = TRUE), "`parm` must be")
  expect_equal(rownames(confint(fit, parm = 2)), "x")
  expect_equal(rownames(confint(fit, parm = "x")), "x")
})

test_that("confint() reads a sampler hyperparameter off the draws", {
  fit <- .sampler_like_fit()
  ci <- confint(fit, parm = c("x", "log_sigma_re"))
  expect_equal(rownames(ci), c("x", "log_sigma_re"))
  expect_equal(unname(ci["log_sigma_re", ]),
               unname(stats::quantile(fit$draws[, "log_sigma_re"],
                                      c(0.025, 0.975))),
               tolerance = 1e-12)
  # The fixed-effect rows are exactly what confint() reports without `parm`.
  expect_equal(ci["x", ], confint(fit)["x", ])
})

test_that("durbin_watson() takes a fit and a time order (#899)", {
  set.seed(3)
  e  <- as.numeric(stats::arima.sim(list(ar = 0.6), n = 80))
  tt <- sample(80)
  # Residuals recorded out of time order are put back in it.
  expect_equal(durbin_watson(e[tt], time = tt)$statistic,
               durbin_watson(e)$statistic)
  expect_error(durbin_watson(e, time = 1:3), "`time`")
  expect_error(durbin_watson(list(1, 2, 3)), "`object` must be a fitted model")

  # A fit is read through its residuals, as moran_i() reads it.
  fit <- structure(list(), class = "tulpa_fit")
  testthat::local_mocked_bindings(
    residuals.tulpa_fit = function(object, type = "pearson", ...) e,
    .package = "tulpa")
  expect_equal(durbin_watson(fit)$statistic, durbin_watson(e)$statistic)
})

test_that("VarCorr() carries a residual row for a gaussian fit (#899)", {
  skip_on_cran()
  set.seed(1)
  G <- 20L; per <- 8L; n <- G * per; g <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  d <- data.frame(y = 1 + 0.5 * x + rnorm(G, 0, 0.8)[g] + rnorm(n, 0, 0.6),
                  x = x, g = factor(g))
  fit <- tulpa(y ~ x + (1 | g), data = d, mode = "eb", estimate_phi = TRUE)
  vc <- VarCorr(fit)
  expect_equal(vc$term, c("g", "Residual"))
  expect_true(is.na(vc$coef[2]))
  expect_equal(vc$sd[2], sqrt(fit$phi), tolerance = 1e-12)
  expect_equal(vc$source, c("estimated", "estimated"))
  expect_named(attr(vc, "cov"), "g")

  fit_c <- tulpa(y ~ x + (1 | g), data = d, mode = "laplace",
                 sigma_re = 0.8, phi = 0.36)
  vc_c <- VarCorr(fit_c)
  expect_equal(vc_c$sd[vc_c$term == "Residual"], 0.6, tolerance = 1e-12)
  expect_equal(vc_c$source[vc_c$term == "Residual"], "conditioned")

  # A family without a residual variance has no such row.
  d$y <- rpois(n, 2)
  fit_p <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb")
  expect_false("Residual" %in% VarCorr(fit_p)$term)
})
