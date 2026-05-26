# Continuous (NNGP/GP) spatial nested routing through the tulpa() front door
# (follow-up to the areal Increment 4, dev_notes/plan_gibbs_spatial_frontdoor.md).
# A spatial_gp(~ lon + lat) spec carries the coordinate columns -- there is NO
# spatial(col) term; observations are mapped to locations from their coords.
# mode = "nested_laplace" / "structured" / "auto" integrate the GP
# hyperparameters (sigma2, phi_gp) via tulpa_nested_laplace(); the routing adds
# no math (identical to a direct call on the same prior). The (coords, nn_order,
# spatial_idx) convention reuses the production conditional path laplace_gp_at().

# Binomial GP panel: an exponential-kernel field on n_loc unique locations,
# `reps` observations per location (exercises the obs -> location map), a fixed
# intercept + slope. lon/lat are columns so the spatial_gp() coord-resolution
# path runs end to end.
sim_gp_binomial <- function(n_loc = 60L, reps = 3L, sigma2 = 0.7, phi_gp = 0.25,
                            beta = c(-0.2, 0.6), seed = 11L) {
  set.seed(seed)
  coords <- cbind(runif(n_loc), runif(n_loc))
  D <- as.matrix(dist(coords))
  K <- sigma2 * exp(-D / phi_gp) + diag(1e-6, n_loc)
  w <- as.numeric(crossprod(chol(K), rnorm(n_loc)))
  w <- w - mean(w)
  loc <- rep(seq_len(n_loc), each = reps)
  N <- length(loc)
  x <- rnorm(N)
  eta <- beta[1] + beta[2] * x + w[loc]
  ntr <- rep(4L, N)
  y <- rbinom(N, ntr, plogis(eta))
  list(
    data = data.frame(y = y, x = x,
                      lon = coords[loc, 1], lat = coords[loc, 2], ntrials = ntr),
    beta = beta, n_loc = n_loc
  )
}

test_that("mode = nested_laplace integrates an NNGP field through tulpa()", {
  skip_on_cran()
  s <- sim_gp_binomial()
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
    spatial = spatial_gp(~ lon + lat, nn = 8L),
    mode = "nested_laplace",
    control = list(keep_grid_hessians = TRUE)
  ))
  expect_equal(fit$backend, "nested_laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_s3_class(fit, "tulpa_fit")
  expect_s3_class(fit, "tulpa_nested_laplace")
  # The GP hyperparameters were integrated, not conditioned on.
  expect_named(fit$theta_mean, c("sigma2", "phi_gp"))
  expect_true(all(is.finite(fit$theta_mean)))
  expect_true(abs(sum(fit$weights) - 1) < 1e-6)

  # Recovery: marginalize beta over the (sigma2, phi_gp) grid (grid modes
  # weighted by the integration weights), never a plug-in MAP. grid_modes[[k]]
  # holds the length-p fixed-effect block.
  w  <- fit$weights
  bm <- do.call(rbind, lapply(fit$grid_modes, function(m) m[1:2]))
  beta_hat <- as.numeric(crossprod(w, bm))
  # The NNGP field has no sum-to-zero constraint, so the intercept aliases with
  # the field's overall level -> wider band by construction; the slope is clean.
  expect_lt(abs(beta_hat[1] - s$beta[1]), 0.5)   # intercept
  expect_lt(abs(beta_hat[2] - s$beta[2]), 0.3)   # slope
})

test_that("structured and auto route an NNGP field to nested_laplace", {
  skip_on_cran()
  s <- sim_gp_binomial(n_loc = 40L, reps = 2L)
  for (m in c("structured", "auto")) {
    fit <- suppressMessages(tulpa(
      y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
      spatial = spatial_gp(~ lon + lat, nn = 6L), mode = m
    ))
    expect_equal(fit$backend, "nested_laplace", info = m)
    expect_equal(fit$inference_mode, "structured", info = m)
  }
})

test_that("the NNGP front-door route is numerically identical to a direct call", {
  s <- sim_gp_binomial(n_loc = 40L, reps = 2L)
  spec <- spatial_gp(~ lon + lat, nn = 6L)
  via <- suppressMessages(tulpa(
    y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
    spatial = spec, mode = "nested_laplace"
  ))
  # Build the same prior the front door builds (validate_gp then the spec ->
  # prior converter), then call the fitter directly: the routing adds no math.
  vspec  <- suppressMessages(tulpa:::validate_gp(spec, s$data))
  prior  <- tulpa:::.spatial_spec_to_nl_prior(vspec)
  direct <- tulpa_nested_laplace(
    s$data$y, s$data$ntrials, model.matrix(y ~ x, s$data),
    prior = prior, family = "binomial"
  )
  expect_equal(via$theta_grid,   direct$theta_grid)
  expect_equal(via$log_marginal, direct$log_marginal)
  expect_equal(via$theta_mean,   direct$theta_mean)
})

test_that("a continuous field rejects a spatial(col) term", {
  s <- sim_gp_binomial(n_loc = 20L, reps = 1L)
  s$data$site <- factor(seq_len(nrow(s$data)))
  expect_error(
    suppressMessages(tulpa(
      y ~ x + spatial(site), data = s$data, family = "binomial",
      n_trials = s$data$ntrials, spatial = spatial_gp(~ lon + lat),
      mode = "nested_laplace")),
    "coordinate columns"
  )
})

test_that("hsgp / spde fields are not yet routed through tulpa()", {
  s <- sim_gp_binomial(n_loc = 20L, reps = 1L)
  expect_error(
    suppressMessages(tulpa(
      y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
      spatial = spatial_hsgp(~ lon + lat), mode = "nested_laplace")),
    "not yet routed"
  )
})

test_that("a bare list continuous spec is rejected with guidance", {
  s <- sim_gp_binomial(n_loc = 20L, reps = 1L)
  expect_error(
    suppressMessages(tulpa(
      y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
      spatial = list(type = "gp"), mode = "nested_laplace")),
    "spatial_gp"
  )
})
