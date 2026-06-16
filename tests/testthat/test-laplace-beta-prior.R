# Tests for the optional fixed-effect Gaussian prior on tulpa_laplace()
# (gcol33/tulpa#27). These exercise the real C++ multi-RE Laplace solver:
#
#   * the NULL default reproduces the historical weak prior N(0, 100^2);
#   * a tight prior shrinks the MAP toward the prior mean;
#   * a weak prior recovers the data-generating coefficients (vs glm());
#   * per-coefficient priors shrink only the targeted columns;
#   * validation rejects malformed priors and the spatial combination.

# Simulate a logistic GLM. With no formula-side random effects the Laplace
# path fits a pure fixed-effects model (no spurious group intercept), so an
# unpenalized fit is the plain logistic MLE and matches glm() on every column.
sim_logit <- function(n = 600L, beta = c(-0.5, 1.2, -0.8), seed = 1L) {
  set.seed(seed)
  p <- length(beta)
  X <- cbind(1, matrix(rnorm(n * (p - 1L)), nrow = n))
  eta <- as.numeric(X %*% beta)
  y <- rbinom(n, 1L, plogis(eta))
  list(y = y, X = X, beta = beta)
}

fit_beta <- function(d, beta_prior = NULL) {
  fit <- tulpa_laplace(
    y = d$y, n_trials = rep(1L, length(d$y)), X = d$X,
    family = "binomial", beta_prior = beta_prior,
    max_iter = 200L, tol = 1e-10
  )
  fit$mode[seq_len(ncol(d$X))]
}


test_that("beta_prior = NULL reproduces the explicit weak prior N(0, 100^2)", {
  skip_on_cran()
  d <- sim_logit()
  b_default <- fit_beta(d, NULL)
  b_weak    <- fit_beta(d, list(mean = 0, sd = 100))
  # tau_beta = 1e-4 <-> sd = 100: the two paths must hit the same mode.
  expect_equal(b_default, b_weak, tolerance = 1e-8)
})


test_that("weak prior recovers glm() coefficients", {
  skip_on_cran()
  d <- sim_logit(n = 800L)
  b_tulpa <- fit_beta(d, list(mean = 0, sd = 100))
  g <- glm(d$y ~ d$X[, -1], family = binomial())
  b_glm <- unname(coef(g))
  # No spurious group intercept now, so a weak prior (tau = 1e-4, negligible
  # vs the likelihood) matches glm on every column including the intercept.
  expect_equal(b_tulpa, b_glm, tolerance = 1e-2)
})


test_that("weak prior recovers the data-generating coefficients", {
  skip_on_cran()
  d <- sim_logit(n = 2000L, beta = c(0.3, -1.0, 0.7), seed = 7L)
  b <- fit_beta(d, list(mean = 0, sd = 100))
  expect_equal(b, d$beta, tolerance = 0.15)
})


test_that("tightening the prior shrinks the MAP monotonically toward 0", {
  skip_on_cran()
  d <- sim_logit()
  # Posterior precision = prior precision + likelihood info, so |beta| toward
  # the mean is monotone non-increasing as sd shrinks.
  sds <- c(100, 1, 0.1, 0.01)
  norms <- vapply(sds, function(s) {
    sum(abs(fit_beta(d, list(mean = 0, sd = s))[-1]))  # slope L1 norm
  }, numeric(1))
  expect_true(all(diff(norms) < 0))
  # A prior far tighter than the likelihood (tau = 1e4 vs info ~150) collapses
  # the slopes onto the prior mean.
  expect_true(all(abs(fit_beta(d, list(mean = 0, sd = 0.01))[-1]) < 0.1))
})


test_that("a very tight prior pulls the MAP toward a non-zero prior mean", {
  skip_on_cran()
  d <- sim_logit(n = 1500L)
  target <- 2.0
  b_tight <- fit_beta(d, list(mean = target, sd = 0.01))
  expect_equal(b_tight, rep(target, length(b_tight)), tolerance = 0.1)
})


test_that("per-coefficient sd shrinks the targeted column far more than others", {
  skip_on_cran()
  d <- sim_logit(n = 1500L)
  b_weak <- fit_beta(d, list(mean = 0, sd = 100))
  # Shrink only the second coefficient (first slope) hard.
  b_mix <- fit_beta(d, list(mean = 0, sd = c(100, 0.01, 100)))
  expect_true(abs(b_mix[2]) < 0.1)                 # targeted column squeezed to ~0
  moved_target <- abs(b_mix[2] - b_weak[2])
  moved_other  <- abs(b_mix[3] - b_weak[3])
  # The untargeted column shifts only a little (the logistic refit rebalances),
  # and far less than the targeted one -- the prior is column-selective.
  expect_lt(moved_other, 0.3)
  expect_lt(moved_other, 0.5 * moved_target)
})


test_that("beta_prior validation rejects malformed input", {
  d <- sim_logit(n = 50L)
  expect_error(fit_beta(d, list(mean = 0)),            "must supply `sd`")
  expect_error(fit_beta(d, list(sd = -1)),             "must be positive")
  expect_error(fit_beta(d, list(sd = c(1, 2))),        "length 1 or 3")
  expect_error(fit_beta(d, 2.5),                       "must be NULL or a list")
})


test_that("sd = +Inf is a no-op penalty: MAP equals the unpenalized MLE", {
  skip_on_cran()
  d <- sim_logit(n = 800L)
  # Precision 1 / Inf^2 = 0 on every coefficient -> no penalty at all. With no
  # spurious latent the Laplace mode is exactly the logistic MLE.
  b_noprior <- fit_beta(d, list(mean = 0, sd = Inf))
  g <- glm(d$y ~ d$X[, -1], family = binomial())
  expect_equal(b_noprior, unname(coef(g)), tolerance = 1e-3)
})


test_that("per-coefficient sd = Inf leaves only the finite-sd columns penalized", {
  skip_on_cran()
  d <- sim_logit(n = 1500L)
  b_weak <- fit_beta(d, list(mean = 0, sd = 100))
  # Inf on cols 1 and 3 (no penalty), tight on col 2 -> only col 2 shrinks; the
  # Inf columns stay at their (near-)unpenalized values.
  b_mix  <- fit_beta(d, list(mean = 0, sd = c(Inf, 0.01, Inf)))
  expect_true(abs(b_mix[2]) < 0.1)
  expect_equal(b_mix[c(1, 3)], b_weak[c(1, 3)], tolerance = 0.3)
})


test_that("beta_prior errors on the spatial Laplace path", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- adj[2, 3] <- adj[3, 2] <- adj[3, 4] <- adj[4, 3] <- 1
  spatial <- list(type = "icar", adjacency = adj, spatial_idx = c(1L, 2L, 3L, 4L))
  X <- cbind(1, rnorm(4))
  expect_error(
    tulpa_laplace(
      y = c(0L, 1L, 1L, 0L), n_trials = rep(1L, 4), X = X,
      family = "binomial", spatial = spatial,
      beta_prior = list(mean = 0, sd = 1)
    ),
    "not supported on the spatial Laplace path"
  )
})
