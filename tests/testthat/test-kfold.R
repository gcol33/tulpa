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
