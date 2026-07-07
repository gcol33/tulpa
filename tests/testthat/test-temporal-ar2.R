# AR(2) temporal field via the tgmrf latent-block path (gcol33/tulpa C10). The
# precision builder is proven to be the EXACT stationary AR(2) GMRF (its inverse
# is the Yule-Walker Toeplitz covariance; it is pentadiagonal and PD); the PACF
# parameterization is proven to stay inside the stationarity triangle; and the
# block recovers the AR coefficients end-to-end through nested Laplace.

test_that(".ar2_precision is the exact stationary AR(2) GMRF", {
  n <- 60L; phi1 <- 0.5; phi2 <- 0.3; tau <- 4
  Q <- as.matrix(tulpa:::.ar2_precision(n, phi1, phi2, tau))

  # Pentadiagonal (order-2 Markov): nothing beyond band 2.
  expect_true(all(Q[abs(row(Q) - col(Q)) > 2] == 0))

  # Inverse == stationary Toeplitz covariance from the Yule-Walker recursion.
  s2 <- 1 / tau
  g <- numeric(n)
  g[1] <- s2 * (1 - phi2) / ((1 + phi2) * ((1 - phi2)^2 - phi1^2))
  g[2] <- phi1 * g[1] / (1 - phi2)
  for (k in 3:n) g[k] <- phi1 * g[k - 1] + phi2 * g[k - 2]
  Sigma <- stats::toeplitz(g)
  expect_equal(solve(Q), Sigma, tolerance = 1e-6)

  # Interior marginal variance == gamma0; symmetric and positive definite.
  expect_equal(diag(solve(Q))[n %/% 2L], g[1], tolerance = 1e-6)
  expect_true(isSymmetric(unname(Q), tol = 1e-8))
  expect_true(all(eigen(Q, symmetric = TRUE, only.values = TRUE)$values > 0))
})

test_that("PACF parameterization stays inside the stationarity triangle", {
  for (a1 in c(-2.5, -0.6, 0.7, 2.5)) for (a2 in c(-2.5, 0, 1.8)) {
    phi <- tulpa:::.ar2_pacf_to_phi(tanh(a1), tanh(a2))
    p1 <- phi[["phi1"]]; p2 <- phi[["phi2"]]
    expect_true(p2 > -1 && (p1 + p2) < 1 && (p2 - p1) < 1)
  }
})

test_that("temporal_ar2() builds a valid tgmrf latent block", {
  blk <- temporal_ar2(rep(seq_len(40L), each = 2L))
  expect_s3_class(blk, "tgmrf")
  expect_s3_class(blk, "tulpa_latent_block")
  expect_equal(blk$n_latent, 40L)
  expect_equal(blk$theta_dim, 3L)
})

test_that("temporal_ar2() recovers the AR(2) coefficients through nested Laplace", {
  skip_on_cran()
  set.seed(3)
  Tt <- 120L
  phi1 <- 0.6; phi2 <- 0.25; s_eps <- 0.4
  w <- numeric(Tt); w[1:2] <- rnorm(2, 0, 0.6)
  for (t in 3:Tt) w[t] <- phi1 * w[t - 1] + phi2 * w[t - 2] + rnorm(1, 0, s_eps)
  d <- data.frame(t = seq_len(Tt), y = w + rnorm(Tt, 0, 0.25))

  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ latent(temporal_ar2(d$t)), data = d, family = "gaussian",
    mode = "nested_laplace")))

  expect_true(is.finite(fit$log_marginal[which.max(fit$log_marginal)]))
  expect_length(fit$theta_mean, 3L)

  # theta_mean = (log_tau, atanh_psi1, atanh_psi2) -> AR coefficients.
  psi1 <- tanh(fit$theta_mean[[2]]); psi2 <- tanh(fit$theta_mean[[3]])
  phi  <- tulpa:::.ar2_pacf_to_phi(psi1, psi2)
  # Single-realization AR recovery is noisy; require the right region.
  expect_gt(phi[["phi1"]], 0.25)
  expect_lt(phi[["phi1"]], 0.95)
  expect_gt(phi[["phi2"]], -0.1)
  expect_lt(phi[["phi2"]], 0.6)
})
