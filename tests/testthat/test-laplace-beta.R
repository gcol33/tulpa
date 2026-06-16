test_that("tulpa_laplace(family='beta') rejects y outside (0, 1)", {
  X <- cbind(1, runif(5))
  expect_error(
    tulpa_laplace(y = c(0, 0.2, 0.4, 0.6, 0.8), n_trials = NULL, X = X,
                  family = "beta", phi = 10),
    "strictly in"
  )
  expect_error(
    tulpa_laplace(y = c(0.1, 0.3, 0.5, 0.7, 1), n_trials = NULL, X = X,
                  family = "beta", phi = 10),
    "strictly in"
  )
  expect_error(
    tulpa_laplace(y = c(0.1, 0.3, 0.5, 0.7, 0.9), n_trials = NULL, X = X,
                  family = "beta", phi = -1),
    "phi"
  )
})

test_that("tulpa_laplace_beta recovers parameters on simulated data", {
  skip_on_cran()
  set.seed(2026)
  N <- 400
  x <- runif(N, -2, 2)
  beta_true <- c(0.3, -0.9)
  phi_true  <- 25
  eta <- beta_true[1] + beta_true[2] * x
  mu  <- plogis(eta)
  y   <- rbeta(N, mu * phi_true, (1 - mu) * phi_true)
  y   <- pmin(pmax(y, 1e-6), 1 - 1e-6)

  fit <- tulpa_laplace_beta(y = y, X = model.matrix(~ x))

  expect_true(fit$phi_converged)
  expect_lt(abs(fit$mode[1] - beta_true[1]), 0.1)
  expect_lt(abs(fit$mode[2] - beta_true[2]), 0.1)
  expect_lt(abs(fit$phi - phi_true) / phi_true, 0.20)
})

test_that("tulpa_laplace_beta matches betareg to high precision", {
  skip_if_not_installed("betareg")
  skip_on_cran()

  set.seed(11)
  N <- 500
  x <- runif(N, -1.5, 1.5)
  eta <- 0.5 - 1.2 * x
  mu  <- plogis(eta)
  y   <- rbeta(N, mu * 30, (1 - mu) * 30)
  y   <- pmin(pmax(y, 1e-6), 1 - 1e-6)

  fit_t <- tulpa_laplace_beta(y = y, X = model.matrix(~ x))
  fit_b <- betareg::betareg(y ~ x)

  br <- coef(fit_b)
  expect_lt(abs(fit_t$mode[1] - br[["(Intercept)"]]), 0.005)
  expect_lt(abs(fit_t$mode[2] - br[["x"]]),           0.005)
  expect_lt(abs(fit_t$phi     - br[["(phi)"]]) / br[["(phi)"]], 0.02)
})

test_that("tulpa_laplace(family='beta') handles a random intercept", {
  skip_on_cran()
  set.seed(7)
  N <- 600
  G <- 20
  group <- rep(seq_len(G), length.out = N)
  u <- rnorm(G, sd = 0.5)
  x <- runif(N, -1, 1)
  eta <- 0.2 + 0.8 * x + u[group]
  mu  <- plogis(eta)
  y   <- rbeta(N, mu * 40, (1 - mu) * 40)
  y   <- pmin(pmax(y, 1e-6), 1 - 1e-6)

  re_list <- list(list(idx = as.integer(group), n_groups = G, sigma = 0.5))
  fit <- tulpa_laplace(
    y = y, n_trials = NULL, X = model.matrix(~ x),
    re_list = re_list, family = "beta", phi = 40
  )

  expect_true(is.finite(fit$log_marginal))
  expect_lt(abs(fit$mode[1] - 0.2), 0.3)
  expect_lt(abs(fit$mode[2] - 0.8), 0.2)
})
