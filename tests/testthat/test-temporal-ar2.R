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

test_that(".arp_* generalize the AR(2) construction (gcol33/tulpa C10)", {
  # Levinson-Durbin PACF map: round-trips against stats::ARMAacf(pacf = TRUE)
  # for a general AR(3), and reduces to the closed form at p = 2.
  psi <- c(0.5, -0.3, 0.2)
  phi <- tulpa:::.arp_pacf_to_phi(psi)
  expect_equal(unname(stats::ARMAacf(ar = phi, lag.max = 3, pacf = TRUE)),
               psi, tolerance = 1e-10)
  expect_equal(tulpa:::.arp_pacf_to_phi(c(0.4, 0.2)),
               unname(tulpa:::.ar2_pacf_to_phi(0.4, 0.2)), tolerance = 1e-12)

  # Autocovariances match stats::ARMAacf up to the gamma0 scale.
  n <- 40L; tau <- 4
  g <- tulpa:::.arp_autocov(n, phi, 1 / tau)
  acf_ref <- as.numeric(stats::ARMAacf(ar = phi, lag.max = n - 1L))
  expect_equal(g / g[1], acf_ref, tolerance = 1e-8)

  # Precision: width-3 band, PD, and inverse == Toeplitz(gamma).
  Q <- as.matrix(tulpa:::.arp_precision(n, phi, tau))
  expect_true(all(Q[abs(row(Q) - col(Q)) > 3] == 0))
  expect_true(all(eigen(Q, symmetric = TRUE, only.values = TRUE)$values > 0))
  expect_equal(solve(Q), stats::toeplitz(g), tolerance = 1e-6)

  # Any tanh-mapped psi stays stationary: all AR polynomial roots outside the
  # unit circle.
  for (seed in 1:5) {
    set.seed(seed)
    ps <- tanh(rnorm(4, 0, 1.5))
    ph <- tulpa:::.arp_pacf_to_phi(ps)
    roots <- polyroot(c(1, -ph))
    expect_true(all(Mod(roots) > 1))
  }
})

test_that("temporal_ar() builds valid blocks and pins the p = 2 equivalence", {
  idx <- rep(seq_len(30L), each = 2L)
  blk3 <- temporal_ar(idx, p = 3)
  expect_s3_class(blk3, "tgmrf")
  expect_equal(blk3$theta_dim, 4L)
  expect_equal(blk3$name %||% "ar3", "ar3")

  # temporal_ar2() is the p = 2 case: identical precision at identical theta.
  blk2  <- temporal_ar(idx, p = 2)
  blk2b <- temporal_ar2(idx)
  th <- c(0.2, atanh(0.4), atanh(0.1))
  expect_equal(as.matrix(blk2$Q(th)), as.matrix(blk2b$Q(th)), tolerance = 1e-12)
  expect_equal(blk2$prior(th), blk2b$prior(th))

  expect_error(temporal_ar(idx, p = 0), ">= 1")
  expect_error(temporal_ar(rep(1:3, 4), p = 3), "at least 4")
})

test_that("temporal_ar(p = 3) recovers the AR field through nested Laplace", {
  skip_on_cran()
  set.seed(5)
  Tt <- 150L
  phi <- c(0.4, 0.2, -0.2); s_eps <- 0.4
  w <- numeric(Tt); w[1:3] <- rnorm(3, 0, 0.5)
  for (t in 4:Tt) w[t] <- sum(phi * w[t - 1:3]) + rnorm(1, 0, s_eps)
  d <- data.frame(t = seq_len(Tt), y = w + rnorm(Tt, 0, 0.25))

  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ latent(temporal_ar(d$t, p = 3)), data = d, family = "gaussian",
    mode = "nested_laplace", control = list(n_per_axis = 3L))))

  expect_true(is.finite(fit$log_marginal[which.max(fit$log_marginal)]))
  expect_length(fit$theta_mean, 4L)
  psi_hat <- tanh(unlist(fit$theta_mean[2:4]))
  phi_hat <- tulpa:::.arp_pacf_to_phi(psi_hat)
  # Single-realization recovery is noisy; require the right region.
  expect_lt(max(abs(phi_hat - phi)), 0.45)
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
