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
