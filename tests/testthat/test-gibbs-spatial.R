# Wiring + recovery for the spatial Polya-Gamma Gibbs samplers reached through
# tulpa_gibbs(spatial = ...) -> dispatch_gibbs_spatial(). Increment 1 of the
# gibbs-spatial front-door work (dev_notes/plan_gibbs_spatial_frontdoor.md):
# the ICAR sampler, the canonical areal case.

# Rook-neighbour adjacency for an nr x nc grid (1-based units, column-major).
rook_adj <- function(nr, nc) {
  n <- nr * nc
  A <- matrix(0L, n, n)
  idx <- function(r, c) (c - 1L) * nr + r
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    i <- idx(r, c)
    if (r > 1L)  A[i, idx(r - 1L, c)] <- 1L
    if (r < nr)  A[i, idx(r + 1L, c)] <- 1L
    if (c > 1L)  A[i, idx(r, c - 1L)] <- 1L
    if (c < nc)  A[i, idx(r, c + 1L)] <- 1L
  }
  A
}

test_that("tulpa_gibbs(spatial = icar) recovers fixed effects on simulated data", {
  skip_on_cran()
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
    iter = 3000L, warmup = 1500L, verbose = FALSE
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
    "not yet wired"
  )
})
