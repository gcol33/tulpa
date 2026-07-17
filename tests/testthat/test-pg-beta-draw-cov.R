# The Polya-Gamma Gibbs beta update samples N(m, (X'WX + P)^{-1}) via a
# triangular solve on the Cholesky factor. Posterior MEANS are insensitive to
# getting that solve wrong (the mean uses the full two-triangle chol_solve),
# so recovery tests cannot catch a mis-sampled covariance: this test pins the
# DRAW covariance against the asymptotic truth. With n large and a flat prior
# the beta posterior is ~ N(beta_mle, vcov(glm)); a correlated design makes
# the lower-vs-upper triangle distinction numerically large.

test_that("PG Gibbs beta draws have the conjugate posterior covariance", {
  skip_on_cran()
  set.seed(303)

  n  <- 1500L
  x1 <- rnorm(n)
  x2 <- 0.85 * x1 + sqrt(1 - 0.85^2) * rnorm(n)   # strongly correlated pair
  X  <- cbind(1, x1, x2)
  beta_true <- c(0.5, 1.0, -1.0)
  ntr <- rep(2L, n)
  y   <- rbinom(n, ntr, plogis(as.numeric(X %*% beta_true)))

  fit <- tulpa_gibbs(
    y = y, n_trials = ntr, X = X,
    group = rep(1L, n), n_groups = 0L,
    family = "binomial",
    beta_prior = list(mean = 0, sd = 10),
    control = list(n_iter = 5000L, warmup = 1000L)
  )

  draws <- fit$beta
  expect_equal(ncol(draws), 3L)

  ref <- stats::glm(cbind(y, ntr - y) ~ x1 + x2, family = stats::binomial())
  V   <- stats::vcov(ref)

  # Marginal SDs within 20% of the asymptotic truth (MC noise ~2-3%; the
  # transposed-solve bug displaced the first coordinate's SD by ~40%).
  sd_draws <- apply(draws, 2, stats::sd)
  sd_ref   <- sqrt(diag(V))
  expect_true(all(abs(sd_draws / sd_ref - 1) < 0.20),
              info = paste("sd ratio:",
                           paste(round(sd_draws / sd_ref, 3), collapse = ", ")))

  # The x1-x2 posterior correlation must match the design-induced one.
  cor_draws <- stats::cor(draws[, 2], draws[, 3])
  cor_ref   <- stats::cov2cor(V)[2, 3]
  expect_lt(abs(cor_draws - cor_ref), 0.10)

  # Means still recover (guards against a fix that breaks the mean solve).
  expect_true(all(abs(colMeans(draws) - coef(ref)) < 0.15))
})
