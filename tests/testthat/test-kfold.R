
test_that("tulpa_kfold rejects spatial / temporal-field fits and callless fits", {
  spatial_fit <- structure(
    list(spatial = list(type = "icar"), call = quote(tulpa()),
         formula = y ~ x, family = "poisson"),
    class = "tulpa_fit")
  expect_error(tulpa_kfold(spatial_fit, data = data.frame(y = 1, x = 1)),
               "not supported")

  callless <- structure(
    list(formula = y ~ x, family = "poisson"),
    class = "tulpa_fit")
  expect_error(tulpa_kfold(callless, data = data.frame(y = 1, x = 1)),
               "no `\\$call`")
})

skip_on_cran()
# Refit-based K-fold CV (gcol33/tulpa C5). The machinery is validated
# end-to-end: refit-CV must award a higher held-out predictive density to the
# correctly-specified model than to a null model, be deterministic given the
# fold assignment, and refuse fits whose data cannot be subset (spatial /
# temporal fields).

test_that("tulpa_kfold refits and scores held-out predictive density", {
  skip_on_cran()
  set.seed(3)
  n <- 200L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.4 + 0.8 * d$x))

  fit_true <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  fit_null <- tulpa(y ~ 1, data = d, family = "poisson", mode = "laplace")

  folds <- rep(seq_len(5L), length.out = n)
  cv_true <- tulpa_kfold(fit_true, data = d, folds = folds)
  cv_null <- tulpa_kfold(fit_null, data = d, folds = folds)

  expect_true(is.finite(cv_true$elpd_kfold))
  expect_true(is.finite(cv_true$se_elpd_kfold) && cv_true$se_elpd_kfold > 0)
  expect_length(cv_true$pointwise, n)
  expect_false(anyNA(cv_true$pointwise))
  expect_equal(cv_true$K, 5L)

  # The correctly-specified model predicts held-out data better.
  expect_gt(cv_true$elpd_kfold, cv_null$elpd_kfold)

  # Deterministic given the fold assignment.
  cv_again <- tulpa_kfold(fit_true, data = d, folds = folds)
  expect_equal(cv_again$elpd_kfold, cv_true$elpd_kfold)
})

test_that("tulpa_kfold's elpd tracks the in-sample log predictive density", {
  skip_on_cran()
  set.seed(11)
  n <- 240L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  cv <- tulpa_kfold(fit, data = d, folds = rep(seq_len(6L), length.out = n))
  # Held-out elpd is a per-observation average around the in-sample mean
  # log-density (a Poisson mean here is O(1) nats); sanity-bound it.
  mean_elpd <- cv$elpd_kfold / n
  expect_gt(mean_elpd, -6)
  expect_lt(mean_elpd, 0)
})

test_that("a refit injects n_trials only where the family reads one", {
  skip_on_cran()
  # Every fit carries a `$n_trials` field since gcol33/tulpa#781 -- the design
  # bundle defaults it to rep(1, n) whatever the family -- so the presence of
  # the field says nothing about whether the model uses one. Reading it as the
  # signal injected `n_trials =` into a poisson refit, which tulpa() refuses
  # (gcol33/tulpa#837). The signal is the family.
  set.seed(3)
  n <- 120L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.4 + 0.8 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  expect_false(is.null(fit$n_trials))          # the field is there ...
  expect_false(tulpa:::.family_reads_trials(fit$family))   # ... and means nothing
  cv <- tulpa_kfold(fit, data = d, folds = rep(seq_len(4L), length.out = n))
  expect_true(is.finite(cv$elpd_kfold))

  # A binomial fit with real denominators still gets the training rows' own
  # counts, so the refit is scored against the model that was fitted.
  set.seed(5)
  db <- data.frame(x = rnorm(n), nt = sample(2:8, n, replace = TRUE))
  db$y <- rbinom(n, db$nt, plogis(-0.3 + 0.7 * db$x))
  fitb <- tulpa(y ~ x, data = db, family = "binomial", n_trials = db$nt,
                mode = "laplace")
  expect_true(tulpa:::.family_reads_trials(fitb$family))
  cvb <- tulpa_kfold(fitb, data = db, folds = rep(seq_len(4L), length.out = n))
  expect_true(is.finite(cvb$elpd_kfold))
  expect_false(anyNA(cvb$pointwise))
})

test_that("a bounded response is scored through each held-out row's bounds", {
  skip_on_cran()
  set.seed(2)
  n <- 100L
  d <- data.frame(x = rnorm(n))
  d$y <- 1 + 0.5 * d$x + rnorm(n, sd = 0.6)
  folds <- rep(seq_len(4L), length.out = n)

  # An interval whose bounds coincide is an exactly observed value, and a
  # truncation range open on both sides keeps all the mass: either response
  # is the gaussian model, fold for fold.
  ref <- tulpa_kfold(tulpa(y ~ x, data = d, family = "gaussian",
                           mode = "structured"), data = d, folds = folds)
  ex  <- tulpa_kfold(tulpa(y | cens(y, y) ~ x, data = d, family = "gaussian",
                           mode = "laplace"), data = d, folds = folds)
  op  <- tulpa_kfold(tulpa(y | trunc() ~ x, data = d, family = "gaussian",
                           mode = "laplace"), data = d, folds = folds)
  expect_equal(ex$pointwise, ref$pointwise, tolerance = 1e-8)
  expect_equal(op$pointwise, ref$pointwise, tolerance = 1e-8)

  # Left-censored at 0.8: a censored row's held-out density is the predictive
  # mass below its bound, over the training refit's draws and dispersions.
  d$lo <- ifelse(d$y < 0.8, -Inf, d$y)
  d$hi <- ifelse(d$y < 0.8, 0.8, d$y)
  d$yc <- ifelse(d$y < 0.8, NA, d$y)
  fit <- tulpa(yc | cens(lo, hi) ~ x, data = d, family = "gaussian",
               mode = "laplace")
  cv <- tulpa_kfold(fit, data = d, folds = folds)
  expect_false(anyNA(cv$pointwise))

  test <- which(folds == 1L)
  fit1 <- tulpa(yc | cens(lo, hi) ~ x, data = d[-test, ], family = "gaussian",
                mode = "laplace")
  eta <- tulpa:::.tulpa_eta_draws(fit1, newdata = d[test, ],
                                  synth_seed = tulpa:::.DIAGNOSTIC_SEED$criteria)
  proc <- tulpa:::.tulpa_response_process(fit1, "test")
  sd <- sqrt(tulpa:::.tulpa_phi_draws(fit1, proc, attr(eta, "cells"), nrow(eta)))
  cen <- is.na(d$yc[test])
  expect_true(any(cen))
  by_hand <- vapply(seq_along(test), function(j) {
    p <- if (cen[j]) pnorm(0.8, eta[, j], sd) else dnorm(d$y[test[j]], eta[, j], sd)
    log(mean(p))
  }, numeric(1))
  expect_equal(cv$pointwise[test], by_hand, tolerance = 1e-8)
})
