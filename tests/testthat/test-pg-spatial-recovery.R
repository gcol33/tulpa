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
