test_that("tulpa_nuts_beta rejects y outside (0, 1)", {
  X <- cbind(1, runif(5))
  expect_error(
    tulpa_nuts_beta(y = c(0, 0.2, 0.4, 0.6, 0.8), X = X, n_iter = 50L,
                    n_warmup = 25L, seed = 1L),
    "strictly in"
  )
  expect_error(
    tulpa_nuts_beta(y = c(0.1, 0.3, 0.5, 0.7, 1), X = X, n_iter = 50L,
                    n_warmup = 25L, seed = 1L),
    "strictly in"
  )
})

test_that("tulpa_nuts_beta rejects bad prior SDs", {
  X <- cbind(1, runif(5))
  y <- c(0.1, 0.3, 0.5, 0.7, 0.9)
  expect_error(tulpa_nuts_beta(y, X, sigma_beta = -1, n_iter = 50L,
                               n_warmup = 25L, seed = 1L),
               "sigma_beta")
  expect_error(tulpa_nuts_beta(y, X, log_phi_prior_sd = 0, n_iter = 50L,
                               n_warmup = 25L, seed = 1L),
               "log_phi_prior_sd")
})

test_that("tulpa_nuts_beta recovers parameters on simulated data", {
  set.seed(2026)
  N <- 400
  x <- runif(N, -2, 2)
  beta_true <- c(0.3, -0.9)
  phi_true  <- 25
  eta <- beta_true[1] + beta_true[2] * x
  mu  <- plogis(eta)
  y   <- rbeta(N, mu * phi_true, (1 - mu) * phi_true)
  y   <- pmin(pmax(y, 1e-6), 1 - 1e-6)

  fit <- tulpa_nuts_beta(
    y = y, X = model.matrix(~ x),
    n_iter = 1000L, n_warmup = 500L, seed = 2026L
  )

  draws <- fit$draws
  beta1_post <- mean(draws[, "beta[1]"])
  beta2_post <- mean(draws[, "beta[2]"])
  phi_post   <- fit$phi_summary[["mean"]]

  expect_lt(abs(beta1_post - beta_true[1]), 0.1)
  expect_lt(abs(beta2_post - beta_true[2]), 0.1)
  expect_lt(abs(phi_post - phi_true) / phi_true, 0.20)

  # Sanity on the sampler itself.
  expect_true(mean(fit$accept_prob) > 0.5)
  expect_true(sum(fit$divergent) < 0.05 * length(fit$divergent))
})

test_that("tulpa_nuts_beta agrees with tulpa_laplace_beta in mean", {
  set.seed(11)
  N <- 500
  x <- runif(N, -1.5, 1.5)
  eta <- 0.5 - 1.2 * x
  mu  <- plogis(eta)
  y   <- rbeta(N, mu * 30, (1 - mu) * 30)
  y   <- pmin(pmax(y, 1e-6), 1 - 1e-6)

  X   <- model.matrix(~ x)
  fit_l <- tulpa_laplace_beta(y = y, X = X)
  fit_n <- tulpa_nuts_beta(y = y, X = X,
                           n_iter = 1500L, n_warmup = 750L, seed = 11L)

  beta_l <- fit_l$mode[1:2]
  beta_n <- colMeans(fit_n$draws[, c("beta[1]", "beta[2]")])

  # Posterior means under a weak prior should be close to the Laplace mode
  # for a well-identified GLM. Allow a generous slack to avoid flake on the
  # 750-iter warmup.
  expect_lt(abs(beta_l[1] - beta_n[1]), 0.05)
  expect_lt(abs(beta_l[2] - beta_n[2]), 0.05)
  expect_lt(abs(fit_l$phi - fit_n$phi_summary[["mean"]]) / fit_l$phi, 0.10)
})
