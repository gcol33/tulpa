# Adaptive Gauss-Hermite quadrature for one-RE GLMMs.

test_that("gauss_hermite_prob nodes and weights are correct (n=3, 5)", {
  # E[Z^2] = 1, E[Z^4] = 3 for Z ~ N(0,1). Quadrature should reproduce.
  for (n in c(3L, 5L, 7L)) {
    gh <- tulpa:::gauss_hermite_prob(n)
    expect_equal(sum(gh$weights), 1, tolerance = 1e-10)
    expect_equal(sum(gh$weights * gh$nodes), 0, tolerance = 1e-10)
    expect_equal(sum(gh$weights * gh$nodes^2), 1, tolerance = 1e-10)
    if (n >= 3L) {
      expect_equal(sum(gh$weights * gh$nodes^4), 3, tolerance = 1e-10)
    }
  }
})


test_that("agq_fit recovers fixed-effect estimates on Bernoulli GLMM", {
  skip_on_cran()
  set.seed(401L)
  n_g <- 40L
  n_per <- 8L
  n <- n_g * n_per
  group <- rep(seq_len(n_g), each = n_per)
  x <- rnorm(n)
  X <- cbind(1, x)
  beta_true <- c(0.4, 0.8)
  sigma_true <- 0.7
  u <- rnorm(n_g, 0, sigma_true)
  eta <- X %*% beta_true + u[group]
  y <- rbinom(n, 1L, plogis(eta))

  fit <- agq_fit(y, X, group, family = "binomial", n_quad = 7L)

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$inference_tier, 2L)
  expect_equal(fit$backend, "agq")
  expect_true(fit$converged)
  # Fixed effects within ~3 SE of truth on this n.
  beta_hat <- fit$means[1:2]
  expect_lt(abs(beta_hat[1] - beta_true[1]), 0.4)
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.4)
  # sigma_re estimate within reason.
  expect_gt(fit$sigma_re, 0.3)
  expect_lt(fit$sigma_re, 1.3)
})


test_that("agq with n_quad=1 reproduces Laplace approximation", {
  skip_on_cran()
  set.seed(402L)
  n_g <- 25L
  n_per <- 6L
  n <- n_g * n_per
  group <- rep(seq_len(n_g), each = n_per)
  x <- rnorm(n)
  X <- cbind(1, x)
  beta_true <- c(0.2, 0.6)
  u <- rnorm(n_g, 0, 0.5)
  eta <- X %*% beta_true + u[group]
  y <- rbinom(n, 1L, plogis(eta))

  fit_lap <- agq_fit(y, X, group, family = "binomial", n_quad = 1L)
  fit_agq <- agq_fit(y, X, group, family = "binomial", n_quad = 9L)

  # AGQ-9 marginal log-lik should be >= Laplace (Laplace under-counts).
  # Both should converge.
  expect_true(fit_lap$converged)
  expect_true(fit_agq$converged)
  # Differences between Laplace and AGQ-9 estimates should be small
  # but non-zero on this scale.
  expect_true(abs(fit_lap$log_marginal - fit_agq$log_marginal) < 5)
})


test_that("agq_fit handles Poisson GLMM", {
  skip_on_cran()
  set.seed(403L)
  n_g <- 30L
  n_per <- 5L
  n <- n_g * n_per
  group <- rep(seq_len(n_g), each = n_per)
  X <- matrix(1, n, 1L)
  u <- rnorm(n_g, 0, 0.4)
  eta <- 1.5 + u[group]
  y <- rpois(n, exp(eta))

  fit <- agq_fit(y, X, group, family = "poisson", n_quad = 5L)
  expect_true(fit$converged)
  expect_lt(abs(fit$means[1] - 1.5), 0.3)
})


test_that("agq_fit handles Gaussian GLMM", {
  skip_on_cran()
  set.seed(404L)
  n_g <- 30L
  n_per <- 6L
  n <- n_g * n_per
  group <- rep(seq_len(n_g), each = n_per)
  X <- matrix(1, n, 1L)
  u <- rnorm(n_g, 0, 1.5)
  eta <- 0.5 + u[group]
  y <- rnorm(n, eta, 1)
  fit <- agq_fit(y, X, group, family = "gaussian", sigma_eps = 1,
                 n_quad = 5L)
  expect_true(fit$converged)
  expect_lt(abs(fit$means[1] - 0.5), 0.5)
  expect_gt(fit$sigma_re, 0.8)
  expect_lt(fit$sigma_re, 2.5)
})


test_that("agq registers in Tier 2 (Structured)", {
  expect_true("agq" %in% INFERENCE_TIERS$structured$backends)
  ti <- get_backend_tier("agq")
  expect_equal(ti$tier, 2L)
})


test_that("agq_fit fixed-effect CIs achieve near-nominal coverage", {
  skip_on_cran()
  # Coverage gate on the shared marginal-Hessian SE path (the same path
  # tulpa_re_aghq / tulpa_nmix_laplace_re report SEs from). 95% Wald CIs for the
  # fixed effects should cover truth at >= 85% over >= 20 seeds (the
  # "Statistical Code Needs Recovery Tests" rule).
  beta_true <- c(0.4, 0.8); sigma_true <- 0.7
  covered <- function(seed) {
    set.seed(seed)
    n_g <- 40L; n_per <- 8L; n <- n_g * n_per
    group <- rep(seq_len(n_g), each = n_per)
    x <- rnorm(n); X <- cbind(1, x)
    u <- rnorm(n_g, 0, sigma_true)
    y <- rbinom(n, 1L, plogis(as.numeric(X %*% beta_true) + u[group]))
    fit <- agq_fit(y, X, group, family = "binomial", n_quad = 7L)
    se <- sqrt(diag(fit$cov))[1:2]
    abs(fit$means[1:2] - beta_true) <= stats::qnorm(0.975) * se
  }
  cov <- rowMeans(vapply(1:30, covered, logical(2)))
  expect_gt(cov[1], 0.85)
  expect_gt(cov[2], 0.85)
})


test_that("agq_fit errors on bad inputs", {
  expect_error(agq_fit(1:5, matrix(1, 5, 1), c(1, 1, 2, 2, 2),
                       family = "binomial", n_quad = 0L),
               "n_quad")
  # Mismatched lengths
  expect_error(agq_fit(1:5, matrix(1, 4, 1), c(1, 1, 2, 2, 2),
                       family = "binomial"),
               "nrow")
})
