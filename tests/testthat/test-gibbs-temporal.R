# Wiring + recovery for the multiscale temporal Polya-Gamma Gibbs sampler reached
# through tulpa_gibbs(temporal = ...) -> dispatch_gibbs_temporal(). Increment 5
# of the gibbs-spatial front-door work (dev_notes/plan_gibbs_spatial_frontdoor.md).

test_that("tulpa_gibbs(temporal = rw1 trend) recovers the slope on simulated data", {
  skip_if_not_slow()
  set.seed(7)
  Tt   <- 40L
  reps <- 8L
  time <- rep(seq_len(Tt), each = reps)          # 1-based per-obs time index
  N    <- length(time)

  # RW1 trend. The kernel centres it each sweep and absorbs the removed level
  # into the intercept, so eta is unchanged and both are identified.
  trend_true <- cumsum(rnorm(Tt, 0, 0.30))
  trend_true <- trend_true - mean(trend_true)

  beta_true <- c(0.1, -0.5)
  x   <- rnorm(N)                                # varies within each time block
  X   <- cbind(1, x)
  eta <- as.numeric(X %*% beta_true) + trend_true[time]
  ntr <- rep(6L, N)
  y   <- rbinom(N, ntr, plogis(eta))

  spec <- validate_temporal_multiscale(
    temporal_multiscale("t", trend = "rw1", short_term = "none"),
    data = data.frame(t = time)
  )
  fit <- tulpa_gibbs(
    y = y, n_trials = ntr, X = X, group = rep(1L, N), n_groups = 0L,
    family = "binomial", temporal = spec,
    control = list(n_iter = 3000L, warmup = 1500L)
  )

  expect_true(all(c("beta", "trend", "sigma_trend") %in% names(fit)))
  expect_equal(ncol(fit$trend), Tt)
  expect_true(all(is.finite(fit$sigma_trend)))

  # Every stored draw is centred exactly, not on average: the level is removed
  # inside the sweep rather than being left for the summary to subtract.
  expect_lt(max(abs(rowMeans(fit$trend))), 1e-8)

  # Both fixed effects recover. The intercept is the one the un-centred kernel
  # could not identify: discarding the removed mean each sweep let the field
  # level wander against it, so only the slope was assertable.
  beta_hat <- colMeans(fit$beta)
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.25)
  expect_lt(abs(beta_hat[1] - beta_true[1]), 0.2)
})

test_that("dispatch_gibbs_temporal rejects rw2, non-binomial, and spatial+temporal", {
  spec_rw2 <- validate_temporal_multiscale(
    temporal_multiscale("t", trend = "rw2"),
    data = data.frame(t = rep(1:5, 2))
  )
  n <- 10L
  expect_error(
    tulpa_gibbs(y = rbinom(n, 5L, 0.5), n_trials = rep(5L, n),
                X = cbind(1, rnorm(n)), group = rep(1L, n), n_groups = 0L,
                family = "binomial", temporal = spec_rw2,,
                control = list(n_iter = 50L, warmup = 25L)),
    "rw2"
  )

  spec_rw1 <- validate_temporal_multiscale(
    temporal_multiscale("t", trend = "rw1"),
    data = data.frame(t = rep(1:5, 2))
  )
  expect_error(
    tulpa_gibbs(y = rpois(n, 2), n_trials = rep(1L, n),
                X = cbind(1, rnorm(n)), group = rep(1L, n), n_groups = 0L,
                family = "poisson", temporal = spec_rw1,,
                control = list(n_iter = 50L, warmup = 25L)),
    "binomial"
  )

  W <- rook_adj(2L, 5L)
  expect_error(
    tulpa_gibbs(y = rbinom(n, 5L, 0.5), n_trials = rep(5L, n),
                X = cbind(1, rnorm(n)), group = rep(1L, n), n_groups = 0L,
                family = "binomial",
                spatial = list(type = "icar", adjacency = W,
                               spatial_idx = seq_len(n)),
                temporal = spec_rw1,,
                control = list(n_iter = 50L, warmup = 25L)),
    "Combined spatial \\+ temporal"
  )
})
