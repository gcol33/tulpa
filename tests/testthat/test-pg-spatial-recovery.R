# Parameter-recovery tests for the binomial Polya-Gamma spatial/temporal Gibbs
# kernels (gcol33/tulpa#143). These kernels are exported but not yet wired to a
# front door; the tests fit them directly on simulated data with a known field
# and assert recovery, so the fixes are validated rather than assumed.

chain_adj <- function(n_s) {
  nb <- lapply(seq_len(n_s), function(i) setdiff(c(i - 1L, i + 1L), c(0L, n_s + 1L)))
  list(adj_list = nb, n_neighbors = vapply(nb, length, integer(1)))
}

test_that("binomial ICAR Gibbs recovers intercept, field, and tau", {
  skip_on_cran()
  set.seed(11)
  n_s <- 30L; reps <- 20L; ntr <- 30L; b0 <- -0.4
  adj <- chain_adj(n_s)
  f <- cumsum(rnorm(n_s, 0, 0.35)); f <- f - mean(f)   # RW1-like ICAR field
  unit <- rep(seq_len(n_s), each = reps)
  y <- rbinom(length(unit), ntr, plogis(b0 + f[unit]))
  X <- matrix(1, length(unit), 1)

  fit <- cpp_pg_binomial_gibbs_spatial(
    y = as.integer(y), n = rep(ntr, length(y)), X = X,
    re_group = rep(0L, length(y)), n_re_groups = 0L,
    spatial_group = as.integer(unit), n_spatial_units = n_s,
    adj_list = adj$adj_list, n_neighbors = adj$n_neighbors,
    n_iter = 3000L, n_warmup = 1500L, thin = 1L, verbose = FALSE)

  b0_hat <- mean(fit$beta[, 1])
  f_hat  <- colMeans(fit$spatial)
  # Intercept recovered (the mean-absorbing centering keeps eta invariant), the
  # field tracks truth, and it is sum-to-zero. tau is well-identified at n = 30
  # trials (weakly identified, so railing, only under Bernoulli data).
  expect_lt(abs(b0_hat - b0), 0.15)
  expect_gt(stats::cor(f_hat, f), 0.9)
  expect_lt(abs(mean(f_hat)), 1e-6)
  expect_gt(mean(fit$tau), 3)      # not railed to the tiny-field degenerate mode
  expect_lt(mean(fit$tau), 60)
})

test_that("binomial NNGP GP Gibbs recovers intercept, field, and sigma2", {
  skip_on_cran()
  set.seed(7)
  N <- 80L; ntr <- 40L; b0 <- 0.3
  sigma2_true <- 1.0; phi_true <- 0.20
  coords <- cbind(runif(N), runif(N))
  K <- sigma2_true * exp(-as.matrix(dist(coords)) / phi_true)
  f <- as.numeric(t(chol(K + 1e-8 * diag(N))) %*% rnorm(N))
  y <- rbinom(N, ntr, plogis(b0 + f))
  X <- matrix(1, N, 1)
  nn <- 10L
  ni <- compute_nngp_neighbors(coords, nn)

  fit <- cpp_pg_binomial_gibbs_gp(
    y = as.integer(y), n = rep(ntr, N), X = X,
    re_group = rep(0L, N), n_re_groups = 0L, coords = coords,
    nn_idx = ni$nn_idx, nn_dist = ni$nn_dist,
    nn_order = as.integer((ni$nn_order %||% seq_len(N)) - 1L),
    n_spatial = N, nn = nn,
    sigma2_gp_init = 0.5, phi_gp_init = 0.5, cov_type = 0L,
    n_iter = 2500L, n_warmup = 1250L, thin = 1L, verbose = FALSE)

  # Before the fix the kernel diverged (sigma2 railed as w was treated iid, phi
  # did a data-free random walk, and the unanchored field level blew the
  # intercept up past 10). It now recovers the intercept, the field, and the
  # variance. The range phi is weakly identified from data, so it is not asserted.
  expect_lt(abs(mean(fit$beta[, 1]) - b0), 0.5)
  expect_gt(stats::cor(colMeans(fit$gp), f), 0.8)
  expect_lt(abs(mean(fit$gp)), 1e-6)                # field anchored sum-to-zero
  expect_gt(mean(fit$sigma2_gp), 0.3)               # not railed
  expect_lt(mean(fit$sigma2_gp), 6)
})

test_that("binomial multiscale NNGP GP Gibbs recovers the total field", {
  skip_on_cran()
  set.seed(9)
  N <- 80L; ntr <- 40L; b0 <- 0.2
  coords <- cbind(runif(N), runif(N))
  K <- exp(-as.matrix(dist(coords)) / 0.25)
  f <- as.numeric(t(chol(K + 1e-8 * diag(N))) %*% rnorm(N)); f <- f - mean(f)
  y <- rbinom(N, ntr, plogis(b0 + f)); X <- matrix(1, N, 1)
  nl <- 8L; nr <- 12L
  il <- compute_nngp_neighbors(coords, nl); ir <- compute_nngp_neighbors(coords, nr)

  fit <- cpp_pg_binomial_gibbs_multiscale_gp(
    y = as.integer(y), n = rep(ntr, N), X = X,
    re_group = rep(0L, N), n_re_groups = 0L, coords = coords,
    nn_idx_local = il$nn_idx, nn_dist_local = il$nn_dist,
    nn_order_local = as.integer(il$nn_order - 1L), nn_local = nl,
    nn_idx_regional = ir$nn_idx, nn_dist_regional = ir$nn_dist,
    nn_order_regional = as.integer(ir$nn_order - 1L), nn_regional = nr,
    n_spatial = N, sigma2_local_init = 0.5, phi_local_init = 0.15,
    sigma2_regional_init = 0.5, phi_regional_init = 0.6, cov_type = 0L,
    n_iter = 2000L, n_warmup = 1000L, thin = 1L, verbose = FALSE)

  # The local + regional split is only weakly identified, but their sum (the
  # total field) recovers and the intercept no longer diverges. Total variance
  # (sigma2_local + sigma2_regional) is near the truth (1.0).
  ftot <- colMeans(fit$w_local) + colMeans(fit$w_regional)
  expect_lt(abs(mean(fit$beta[, 1]) - b0), 0.5)
  expect_gt(stats::cor(ftot, f), 0.8)
  expect_gt(mean(fit$sigma2_local) + mean(fit$sigma2_regional), 0.4)
  expect_lt(mean(fit$sigma2_local) + mean(fit$sigma2_regional), 3)
})

test_that("binomial temporal AR1 Gibbs recovers the field and samples rho", {
  skip_on_cran()
  set.seed(4)
  Tt <- 40L; reps <- 25L; ntr <- 20L; b0 <- 0.1; rho_true <- 0.7
  f <- as.numeric(arima.sim(list(ar = rho_true), Tt, sd = 0.5)); f <- f - mean(f)
  tm <- rep(seq_len(Tt), each = reps)
  y <- rbinom(length(tm), ntr, plogis(b0 + f[tm]))
  X <- matrix(1, length(tm), 1)

  fit <- cpp_pg_binomial_gibbs_temporal(
    y = as.integer(y), n = rep(ntr, length(tm)), X = X,
    re_group = rep(0L, length(tm)), n_re_groups = 0L,
    time_idx = as.integer(tm), n_times = Tt,
    seasonal_period = 0L, trend_type = 0L, short_type = 1L, rho_short_init = 0.5,
    n_iter = 3000L, n_warmup = 1500L, thin = 1L, verbose = FALSE)

  # Field recovers, and rho_short is now sampled (was fixed at its init and
  # reported as a posterior) via the correct both-neighbour AR1 conditional.
  expect_gt(stats::cor(colMeans(fit$short_term), f), 0.9)
  expect_gt(stats::sd(fit$rho_short), 0.01)          # rho is sampled, not stuck
  expect_lt(abs(mean(fit$rho_short) - rho_true), 0.25)
})
