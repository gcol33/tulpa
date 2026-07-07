# tulpa_reloo() (gcol33/tulpa C5): PSIS-LOO with exact refits at flagged
# observations. Consistency anchor: with every observation flagged, reloo is
# exact LOO-CV and must match tulpa_kfold with K = n (same held-out density,
# fold = observation). Plus the kfold binomial-trials plumbing: a stored
# n_trials argument must be subset per training partition, not evaluated
# full-length.

test_that("reloo with everything flagged equals kfold at K = n", {
  skip_on_cran()
  set.seed(41)
  n <- 40L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.6 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  rl <- tulpa_reloo(fit, data = d, k_threshold = -Inf)
  cv <- tulpa_kfold(fit, data = d, folds = seq_len(n))

  expect_equal(rl$reloo_idx, seq_len(n))
  # Same refits, same held-out scoring; only the fixed-effect draw noise of
  # the draw-free Laplace tier separates them.
  expect_equal(rl$pointwise, cv$pointwise, tolerance = 0.05)
  expect_equal(rl$elpd_loo, cv$elpd_kfold, tolerance = 0.5)
})

test_that("reloo refits exactly the observations above the threshold", {
  skip_on_cran()
  set.seed(43)
  n <- 150L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.4 + 0.7 * d$x))
  # A high-leverage outlier to stress the importance ratios.
  d$x[1] <- 3.5; d$y[1] <- 3L

  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "mala",
               control = list(n_iter = 2000, warmup = 500))

  base <- tulpa_reloo(fit, data = d, k_threshold = Inf)
  expect_length(base$reloo_idx, 0L)
  expect_length(base$pareto_k, n)

  # Flag the top-2 k-hat observations; stored draws make the baseline
  # deterministic across the two calls.
  th <- sort(base$pareto_k, decreasing = TRUE)[3L]
  rl <- tulpa_reloo(fit, data = d, k_threshold = th)
  expect_length(rl$reloo_idx, 2L)
  expect_setequal(rl$reloo_idx, order(base$pareto_k, decreasing = TRUE)[1:2])
  expect_true(all(is.finite(rl$pointwise)))
  # Unflagged observations keep their PSIS values.
  keep <- setdiff(seq_len(n), rl$reloo_idx)
  expect_equal(rl$pointwise[keep], base$pointwise[keep])
})

test_that("kfold threads a stored n_trials argument through each refit", {
  skip_on_cran()
  set.seed(47)
  n <- 160L
  d <- data.frame(x = rnorm(n))
  trials <- rep(c(4L, 9L), length.out = n)
  d$y <- rbinom(n, trials, plogis(-0.2 + 0.8 * d$x))

  fit <- tulpa(y ~ x, data = d, family = "binomial", mode = "laplace",
               n_trials = trials)
  fit_null <- tulpa(y ~ 1, data = d, family = "binomial", mode = "laplace",
                    n_trials = trials)

  folds <- rep(seq_len(4L), length.out = n)
  cv      <- tulpa_kfold(fit, data = d, folds = folds)
  cv_null <- tulpa_kfold(fit_null, data = d, folds = folds)

  expect_true(is.finite(cv$elpd_kfold))
  expect_false(anyNA(cv$pointwise))
  expect_gt(cv$elpd_kfold, cv_null$elpd_kfold)

  # reloo picks the stored trials up too.
  rl <- tulpa_reloo(fit, data = d, k_threshold = Inf)
  expect_true(is.finite(rl$elpd_loo))
})

test_that("reloo rejects spatial fits and callless fits", {
  spatial_fit <- structure(
    list(spatial = list(type = "icar"), call = quote(tulpa()),
         formula = y ~ x, family = "poisson"),
    class = "tulpa_fit")
  expect_error(tulpa_reloo(spatial_fit, data = data.frame(y = 1, x = 1)),
               "not supported")

  callless <- structure(
    list(formula = y ~ x, family = "poisson"),
    class = "tulpa_fit")
  expect_error(tulpa_reloo(callless, data = data.frame(y = 1, x = 1)),
               "no `\\$call`")
})
