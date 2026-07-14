# Wiring + recovery for the spatial Polya-Gamma Gibbs samplers reached through
# tulpa_gibbs(spatial = ...) -> dispatch_gibbs_spatial(). Increment 1 of the
# gibbs-spatial front-door work (dev_notes/plan_gibbs_spatial_frontdoor.md):
# the ICAR sampler, the canonical areal case.

# rook_adj() lives in helper-spatial.R (shared with test-tulpa-spatial-frontdoor.R).

test_that("tulpa_gibbs(spatial = icar) recovers fixed effects on simulated data", {
  skip_if_not_slow()
  set.seed(42)
  nr <- nc <- 7L
  W <- rook_adj(nr, nc)
  n_units <- nr * nc

  reps  <- 4L
  unit  <- rep(seq_len(n_units), each = reps)   # per-obs spatial unit (1-based)
  N     <- length(unit)

  phi_true  <- sim_spatial_effects(W, sigma = 0.6, type = "icar")
  beta_true <- c(-0.3, 0.8)
  x  <- rnorm(N)
  X  <- cbind(1, x)

  n_groups      <- 6L
  grp           <- ((unit - 1L) %% n_groups) + 1L
  sigma_re_true <- 0.4
  b_re          <- rnorm(n_groups, 0, sigma_re_true)

  eta <- as.numeric(X %*% beta_true) + phi_true[unit] + b_re[grp]
  ntr <- rep(5L, N)
  y   <- rbinom(N, ntr, plogis(eta))

  fit <- tulpa_gibbs(
    y = y, n_trials = ntr, X = X, group = grp, n_groups = n_groups,
    family = "binomial",
    spatial = list(type = "icar", adjacency = W, spatial_idx = unit),
    n_iter = 3000L, warmup = 1500L, verbose = FALSE
  )

  # Wiring: the ICAR sampler returns its universal + spatial draws.
  expect_true(is.list(fit))
  expect_true(all(c("beta", "spatial", "tau", "sigma_re") %in% names(fit)))
  expect_equal(ncol(fit$spatial), n_units)
  expect_true(all(is.finite(fit$tau)))

  # Recovery: posterior-mean fixed effects near truth.
  beta_hat <- colMeans(fit$beta)
  expect_lt(abs(beta_hat[1] - beta_true[1]), 0.45)   # intercept
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.30)   # slope
})

test_that("tulpa_gibbs(spatial = icar, neg_binomial_2) recovers fixed effects", {
  skip_if_not_slow()
  set.seed(71)
  nr <- nc <- 7L
  W <- rook_adj(nr, nc)
  n_units <- nr * nc

  reps  <- 4L
  unit  <- rep(seq_len(n_units), each = reps)   # per-obs spatial unit (1-based)
  N     <- length(unit)

  phi_true  <- sim_spatial_effects(W, sigma = 0.5, type = "icar")
  beta_true <- c(0.8, 0.5)                       # log-mean intercept + slope
  x  <- rnorm(N)
  X  <- cbind(1, x)

  n_groups      <- 6L
  grp           <- ((unit - 1L) %% n_groups) + 1L
  sigma_re_true <- 0.3
  b_re          <- rnorm(n_groups, 0, sigma_re_true)

  r_true <- 5.0                                  # negbin dispersion (size)
  eta <- as.numeric(X %*% beta_true) + phi_true[unit] + b_re[grp]
  y   <- rnbinom(N, size = r_true, mu = exp(eta))

  fit <- tulpa_gibbs(
    y = y, n_trials = NULL, X = X, group = grp, n_groups = n_groups,
    family = "neg_binomial_2",
    spatial = list(type = "icar", adjacency = W, spatial_idx = unit),
    n_iter = 3000L, warmup = 1500L, verbose = FALSE
  )

  # Wiring: the negbin ICAR sampler returns its universal + spatial + r draws.
  expect_true(is.list(fit))
  expect_true(all(c("beta", "spatial", "tau", "sigma_re", "r") %in% names(fit)))
  expect_equal(ncol(fit$spatial), n_units)
  expect_true(all(is.finite(fit$r)))

  # Recovery: the slope is identified separately from the field; the ICAR
  # field is improper (constant null space) and the kernel re-centers it to
  # mean-zero each sweep, so the intercept and the field's overall level are
  # confounded. Assert the identifiable combination (intercept + mean field)
  # against truth + the simulated field's mean, as the GP case does.
  beta_hat <- colMeans(fit$beta)
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.25)   # slope
  level_hat  <- beta_hat[1] + mean(colMeans(fit$spatial))
  level_true <- beta_true[1] + mean(phi_true)
  expect_lt(abs(level_hat - level_true), 0.45)
  expect_gt(mean(fit$r), 1.5)                        # dispersion stays sane
})

test_that("negbin spatial Gibbs is wired for the areal ICAR field only", {
  W <- rook_adj(3L, 3L)
  expect_error(
    tulpa_gibbs(y = rnbinom(9, size = 5, mu = 2), n_trials = NULL,
                X = cbind(1, rnorm(9)), group = rep(1L, 9), n_groups = 1L,
                family = "neg_binomial_2",
                spatial = list(type = "bym2", adjacency = W,
                               spatial_idx = seq_len(9))),
    "areal ICAR field"
  )
})

test_that("tulpa_gibbs(spatial = rsr) recovers fixed effects on simulated data", {
  skip_if_not_slow()
  set.seed(11)
  nr <- nc <- 7L
  W <- rook_adj(nr, nc)
  n <- nr * nc                                   # one observation per unit

  phi_true  <- sim_spatial_effects(W, sigma = 0.5, type = "icar")
  beta_true <- c(-0.3, 0.8)
  x  <- rnorm(n)
  X  <- cbind(1, x)
  eta <- as.numeric(X %*% beta_true) + phi_true
  ntr <- rep(12L, n)
  y   <- rbinom(n, ntr, plogis(eta))

  fit <- tulpa_gibbs(
    y = y, n_trials = ntr, X = X, group = rep(1L, n), n_groups = 0L,
    family = "binomial",
    spatial = list(type = "rsr", adjacency = W, spatial_idx = seq_len(n)),
    n_iter = 3000L, warmup = 1500L, verbose = FALSE
  )

  # Wiring: the RSR sampler returns raw + projected field draws plus tau.
  expect_true(all(c("beta", "spatial", "spatial_raw", "tau") %in% names(fit)))
  expect_equal(ncol(fit$spatial), n)
  expect_true(all(is.finite(fit$tau)))

  # Recovery: RSR orthogonalises the field to X, so beta is uncontaminated.
  beta_hat <- colMeans(fit$beta)
  expect_lt(abs(beta_hat[1] - beta_true[1]), 0.45)
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.30)
})

test_that("tulpa_gibbs(spatial = gp) recovers fixed effects (one obs per location)", {
  skip_if_not_slow()
  set.seed(202)
  side <- 8L
  g <- expand.grid(lon = seq(0, 1, length.out = side),
                   lat = seq(0, 1, length.out = side))
  coords <- as.matrix(g)
  n <- nrow(coords)                              # 64 unique locations, 1 obs each

  # Smooth exponential-GP field on the raw grid.
  D     <- as.matrix(dist(coords))
  Sigma <- 0.6^2 * exp(-D / 0.30)
  w_true <- as.numeric(t(chol(Sigma + diag(1e-8, n))) %*% rnorm(n))

  beta_true <- c(-0.2, 0.7)
  x  <- rnorm(n)
  X  <- cbind(1, x)
  eta <- as.numeric(X %*% beta_true) + w_true
  ntr <- rep(20L, n)
  y   <- rbinom(n, ntr, plogis(eta))

  sp <- validate_gp(
    spatial_gp(~ lon + lat, cov = "exponential", nn = 10L),
    data = data.frame(lon = coords[, 1], lat = coords[, 2])
  )
  fit <- tulpa_gibbs(
    y = y, n_trials = ntr, X = X, group = rep(1L, n), n_groups = 0L,
    family = "binomial", spatial = sp,
    n_iter = 3000L, warmup = 1500L, verbose = FALSE
  )

  expect_true(all(c("beta", "gp", "sigma2_gp", "phi_gp") %in% names(fit)))
  expect_equal(ncol(fit$gp), n)
  expect_true(all(is.finite(fit$sigma2_gp)))
  beta_hat <- colMeans(fit$beta)
  # The slope is identified separately from the field; the intercept and the GP
  # surface's overall level are confounded (the smooth field can absorb a global
  # offset), so assert the identifiable combination -- intercept + mean field --
  # against the truth + the simulated field's own mean, not the bare intercept.
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.30)
  level_hat  <- beta_hat[1] + mean(colMeans(fit$gp))
  level_true <- beta_true[1] + mean(w_true)
  expect_lt(abs(level_hat - level_true), 0.45)
})

test_that("tulpa_gibbs(spatial = multiscale_gp) recovers fixed effects", {
  skip_if_not_slow()
  set.seed(303)
  side <- 8L
  g <- expand.grid(lon = seq(0, 1, length.out = side),
                   lat = seq(0, 1, length.out = side))
  coords <- as.matrix(g)
  n <- nrow(coords)

  D <- as.matrix(dist(coords))
  w_true <- as.numeric(t(chol(0.25^2 * exp(-D / 0.4) + diag(1e-8, n))) %*% rnorm(n))

  beta_true <- c(0.1, 0.6)
  x  <- rnorm(n)
  X  <- cbind(1, x)
  eta <- as.numeric(X %*% beta_true) + w_true
  ntr <- rep(25L, n)
  y   <- rbinom(n, ntr, plogis(eta))

  sp <- validate_gp(
    spatial_multiscale(~ lon + lat, cov = "exponential",
                       nn_local = 6L, nn_regional = 12L,
                       range_local = c(0.05, 1.5), range_regional = c(1.5, 8)),
    data = data.frame(lon = coords[, 1], lat = coords[, 2])
  )
  fit <- tulpa_gibbs(
    y = y, n_trials = ntr, X = X, group = rep(1L, n), n_groups = 0L,
    family = "binomial", spatial = sp,
    n_iter = 3000L, warmup = 1500L, verbose = FALSE
  )

  expect_true(all(c("beta", "w_local", "w_regional",
                    "sigma2_local", "sigma2_regional") %in% names(fit)))
  expect_equal(ncol(fit$w_local), n)
  expect_true(all(is.finite(colMeans(fit$w_local))))
  beta_hat <- colMeans(fit$beta)
  # As in the single-scale GP case the slope is identified; the intercept and the
  # two additive field levels are confounded, so assert the identifiable
  # combination (intercept + both field means).
  expect_lt(abs(beta_hat[2] - beta_true[2]), 0.35)
  level_hat  <- beta_hat[1] + mean(colMeans(fit$w_local)) +
                mean(colMeans(fit$w_regional))
  level_true <- beta_true[1] + mean(w_true)
  expect_lt(abs(level_hat - level_true), 0.50)
})

test_that("dispatch_gibbs_spatial rejects non-binomial and unwired types", {
  W <- rook_adj(3L, 3L)
  expect_error(
    tulpa_gibbs(y = rpois(9, 2), n_trials = rep(1L, 9), X = cbind(1, rnorm(9)),
                group = rep(1L, 9), n_groups = 1L, family = "poisson",
                spatial = list(type = "icar", adjacency = W,
                               spatial_idx = seq_len(9))),
    "binomial"
  )
  expect_error(
    tulpa_gibbs(y = rbinom(9, 1, 0.5), n_trials = rep(1L, 9),
                X = cbind(1, rnorm(9)), group = rep(1L, 9), n_groups = 1L,
                family = "binomial",
                spatial = list(type = "car_proper", adjacency = W,
                               spatial_idx = seq_len(9))),
    "not wired"
  )
})

test_that("continuous Gibbs samplers reject repeated-location designs", {
  skip_if_not_slow()
  set.seed(99)
  coords <- as.matrix(expand.grid(lon = 1:5, lat = 1:5))
  # Two observations per location -> no obs->loc map, must error.
  d <- data.frame(lon = rep(coords[, 1], 2), lat = rep(coords[, 2], 2))
  n <- nrow(d)
  sp <- validate_gp(spatial_gp(~ lon + lat, nn = 6L), data = d)
  expect_error(
    tulpa_gibbs(y = rbinom(n, 5L, 0.5), n_trials = rep(5L, n),
                X = cbind(1, rnorm(n)), group = rep(1L, n), n_groups = 0L,
                family = "binomial", spatial = sp,
                n_iter = 100L, warmup = 50L, verbose = FALSE),
    "one observation per unique location"
  )
})

test_that("negbin areal ICAR Gibbs runs and returns the draw blocks", {
  skip_on_cran()
  set.seed(9)
  n_units <- 12L
  adj <- matrix(0, n_units, n_units)
  for (i in 1:(n_units - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
  n <- n_units * 6L
  site <- rep(seq_len(n_units), each = 6L)
  x <- rnorm(n)
  y <- rnbinom(n, size = 5, mu = exp(1 + 0.3 * x))
  fit <- tulpa:::dispatch_gibbs_spatial(
    y = y, n_trials = rep(1L, n), X = cbind(1, x),
    re_group = rep(0L, n), n_re_groups = 0L,
    spatial = list(type = "icar", adjacency = adj, spatial_idx = site),
    family = "neg_binomial_2",
    iter = 200L, warmup = 100L)
  expect_true(all(c("beta", "spatial", "tau", "r") %in% names(fit)))
  expect_equal(nrow(fit$beta), 100L)
  expect_equal(ncol(fit$spatial), n_units)
  expect_true(all(is.finite(fit$beta)))
})
