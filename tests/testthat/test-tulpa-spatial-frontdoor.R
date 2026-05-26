# End-to-end spatial routing through the tulpa() front door (Increment 2b/3 of
# the gibbs-spatial work, dev_notes/plan_gibbs_spatial_frontdoor.md). A
# spatial(col) term names the per-observation spatial-unit column; the structure
# (type, adjacency) arrives via the spatial= argument. mode = "gibbs" reaches the
# binomial Polya-Gamma areal samplers (icar/bym2), mode = "laplace" reaches
# tulpa_laplace(spatial=), mode = "auto" picks Gibbs for a binomial icar/bym2
# field. The two -- term and spec -- must be supplied together.

# rook_adj() lives in helper-spatial.R (shared with test-gibbs-spatial.R).

# A binomial areal panel on an nr x nc grid: one ICAR field on the units, a
# fixed intercept + slope, `reps` observations per unit. `region` is a factor so
# the front-door column-resolution path is exercised.
sim_areal_binomial <- function(nr = 7L, nc = 7L, reps = 4L, seed = 7L) {
  set.seed(seed)
  W <- rook_adj(nr, nc)
  n_units <- nr * nc
  unit <- rep(seq_len(n_units), each = reps)
  N <- length(unit)
  phi  <- sim_spatial_effects(W, sigma = 0.6, type = "icar")
  beta <- c(-0.3, 0.8)
  x    <- rnorm(N)
  eta  <- beta[1] + beta[2] * x + phi[unit]
  ntr  <- rep(5L, N)
  y    <- rbinom(N, ntr, plogis(eta))
  list(
    data    = data.frame(y = y, x = x, region = factor(unit), ntrials = ntr),
    W       = W, beta = beta, n_units = n_units
  )
}

test_that("tulpa(spatial = icar, mode = gibbs) recovers beta end-to-end", {
  skip_on_cran()
  s <- sim_areal_binomial()
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "icar", adjacency = s$W),
    mode = "gibbs",
    control = list(iter = 3000L, warmup = 1500L)
  )
  # Routing: the front door reached the ICAR Polya-Gamma sampler.
  expect_equal(fit$backend, "gibbs")
  expect_equal(fit$inference_tier, 1L)
  expect_true(inherits(fit, "tulpa_fit"))
  expect_true(all(c("beta", "spatial", "tau") %in% names(fit)))
  expect_equal(ncol(fit$spatial), s$n_units)
  expect_true(all(is.finite(fit$tau)))
  # Recovery: posterior-mean fixed effects near truth.
  beta_hat <- colMeans(fit$beta)
  expect_lt(abs(beta_hat[1] - s$beta[1]), 0.45)   # intercept
  expect_lt(abs(beta_hat[2] - s$beta[2]), 0.30)   # slope
})

test_that("auto picks Gibbs for a binomial icar field", {
  skip_on_cran()
  s <- sim_areal_binomial()
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "icar", adjacency = s$W),
    mode = "auto",
    control = list(iter = 1500L, warmup = 750L)
  )
  expect_equal(fit$backend, "gibbs")
  expect_equal(fit$inference_mode, "exact")
})

test_that("tulpa(spatial = bym2, mode = gibbs) wires through and returns field draws", {
  skip_on_cran()
  s <- sim_areal_binomial(reps = 3L)
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "bym2", adjacency = s$W),
    mode = "gibbs",
    control = list(iter = 1500L, warmup = 750L)
  )
  expect_equal(fit$backend, "gibbs")
  expect_equal(ncol(fit$spatial), s$n_units)
  expect_true(all(is.finite(colMeans(fit$beta))))
})

test_that("mode = laplace routes a spatial field through tulpa_laplace", {
  skip_on_cran()
  s <- sim_areal_binomial()
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "icar", adjacency = s$W),
    mode = "laplace"
  )
  expect_equal(fit$backend, "laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_false(is.null(fit$mode))   # spatial Laplace returns the joint MAP vector
  # MAP fixed effects (beta is the leading block of the mode vector).
  beta_hat <- fit$mode[1:2]
  expect_lt(abs(beta_hat[1] - s$beta[1]), 0.5)
  expect_lt(abs(beta_hat[2] - s$beta[2]), 0.35)
})

test_that("front door rejects malformed spatial specifications", {
  s <- sim_areal_binomial(reps = 1L)

  # spatial = supplied but no spatial(col) term in the formula.
  expect_error(
    tulpa(y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
          spatial = list(type = "icar", adjacency = s$W), mode = "gibbs"),
    "no spatial\\(col\\) term"
  )
  # spatial(col) term but no spatial = spec.
  expect_error(
    tulpa(y ~ x + spatial(region), data = s$data, family = "binomial",
          n_trials = s$data$ntrials, mode = "gibbs"),
    "was not supplied"
  )
  # spec missing $type.
  expect_error(
    tulpa(y ~ x + spatial(region), data = s$data, family = "binomial",
          n_trials = s$data$ntrials, spatial = list(adjacency = s$W),
          mode = "gibbs"),
    "spatial\\$type"
  )
  # temporal terms are not yet routed through tulpa().
  expect_error(
    tulpa(y ~ x + temporal(region), data = s$data, family = "binomial",
          n_trials = s$data$ntrials, mode = "gibbs"),
    "not yet routed"
  )
})

test_that("spatial Gibbs is binomial-only and rejects random slopes", {
  s <- sim_areal_binomial(reps = 1L)

  # Non-binomial under spatial Gibbs.
  expect_error(
    tulpa(y ~ x + spatial(region), data = s$data, family = "poisson",
          spatial = list(type = "icar", adjacency = s$W), mode = "gibbs"),
    "binomial"
  )
  # Random slope alongside a spatial field.
  s$data$g <- factor(rep_len(seq_len(5L), nrow(s$data)))
  expect_error(
    tulpa(y ~ x + spatial(region) + (1 + x | g), data = s$data,
          family = "binomial", n_trials = s$data$ntrials,
          spatial = list(type = "icar", adjacency = s$W), mode = "gibbs"),
    "Random-slope"
  )
})
