# Continuous (NNGP/GP and HSGP) spatial nested routing through the tulpa() front
# door (follow-up to the areal Increment 4, dev_notes/plan_gibbs_spatial_frontdoor.md).
# A spatial_gp(~ lon + lat) / spatial_hsgp(~ lon + lat) spec carries the
# coordinate columns -- there is NO spatial(col) term; observations are mapped to
# locations from their coords. mode = "nested_laplace" / "structured" / "auto"
# integrate the GP hyperparameters (NNGP: sigma2, phi_gp; HSGP: sigma2,
# lengthscale) via tulpa_nested_laplace(); the routing adds no math (identical to
# a direct call on the same prior). The NNGP (coords, nn_order, spatial_idx)
# convention reuses the production conditional path laplace_gp_at(); the HSGP
# basis reuses setup_hsgp_2d via cpp_hsgp_basis_2d.

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

test_that("mode = laplace fits a GP field alongside a (1 | g) RE term (issue #74)", {
  s <- sim_gp_binomial(n_loc = 50L, reps = 3L)
  d <- s$data
  d$g <- factor(rep_len(seq_len(6L), nrow(d)))

  fit <- suppressMessages(tulpa(
    y ~ x + (1 | g), data = d, family = "binomial", n_trials = d$ntrials,
    spatial = spatial_gp(~ lon + lat, nn = 8L), mode = "laplace"
  ))

  expect_s3_class(fit, "tulpa_fit")
  # Latent mode is [beta (2), re (6 groups), w_gp (n_loc)]; the RE block is
  # present, so the fixed-effect slope still recovers.
  expect_length(fit$mode, 2L + 6L + s$n_loc)
  beta <- fit$mode[seq_len(2L)]
  expect_lt(abs(beta[2] - s$beta[2]), 0.35)        # slope
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

# SPDE is now routed through tulpa() (its own `spde` backend; see
# test-tulpa-spatial-spde-frontdoor.R). A field type with no front-door wiring
# yet (e.g. multiscale) must still error with guidance rather than fall through.
test_that("an unrouted spatial type errors with guidance", {
  s <- sim_gp_binomial(n_loc = 20L, reps = 1L)
  expect_error(
    suppressMessages(tulpa(
      y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
      spatial = list(type = "multiscale"), mode = "nested_laplace")),
    "Unknown spatial type"
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

# --- HSGP (Hilbert-space GP) nested routing -----------------------------
# A spatial_hsgp(~ lon + lat) spec routes through tulpa() exactly like the GP
# path: coords address the field (no spatial(col) term), and the mode integrates
# the (sigma2, lengthscale) hyperparameters via tulpa_nested_laplace(). The field
# is a sum of Laplacian basis functions evaluated at every observation; the basis
# (phi_basis + lambda_eig) is built by cpp_hsgp_basis_2d -- a thin wrapper over
# setup_hsgp_2d, the single source of truth -- so the routing adds no basis math.

test_that("mode = nested_laplace integrates an HSGP field through tulpa()", {
  skip_on_cran()
  # A smooth (long-range) field is what HSGP approximates well.
  s <- sim_gp_binomial(n_loc = 60L, reps = 3L, sigma2 = 0.8, phi_gp = 0.5,
                       beta = c(-0.2, 0.6), seed = 21L)
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
    spatial = spatial_hsgp(~ lon + lat, m = 8L),
    mode = "nested_laplace",
    control = list(keep_grid_hessians = TRUE)
  ))
  expect_equal(fit$backend, "nested_laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_s3_class(fit, "tulpa_fit")
  expect_s3_class(fit, "tulpa_nested_laplace")
  # The HSGP hyperparameters were integrated, not conditioned on.
  expect_named(fit$theta_mean, c("sigma2", "lengthscale"))
  expect_true(all(is.finite(fit$theta_mean)))
  expect_true(abs(sum(fit$weights) - 1) < 1e-6)

  # Recovery: marginalize beta over the (sigma2, lengthscale) grid (grid modes
  # weighted by the integration weights), never a plug-in MAP.
  w  <- fit$weights
  bm <- do.call(rbind, lapply(fit$grid_modes, function(m) m[1:2]))
  beta_hat <- as.numeric(crossprod(w, bm))
  # Like NNGP, the HSGP field has no sum-to-zero constraint, so the intercept
  # aliases the field's overall level -> wide band; the slope is clean.
  expect_lt(abs(beta_hat[1] - s$beta[1]), 0.6)   # intercept
  expect_lt(abs(beta_hat[2] - s$beta[2]), 0.3)   # slope
})

test_that("structured and auto route an HSGP field to nested_laplace", {
  skip_on_cran()
  s <- sim_gp_binomial(n_loc = 40L, reps = 2L)
  for (m in c("structured", "auto")) {
    fit <- suppressMessages(tulpa(
      y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
      spatial = spatial_hsgp(~ lon + lat, m = 6L), mode = m
    ))
    expect_equal(fit$backend, "nested_laplace", info = m)
    expect_equal(fit$inference_mode, "structured", info = m)
  }
})

test_that("the HSGP front-door route is numerically identical to a direct call", {
  s <- sim_gp_binomial(n_loc = 40L, reps = 2L)
  spec <- spatial_hsgp(~ lon + lat, m = 6L)
  via <- suppressMessages(tulpa(
    y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
    spatial = spec, mode = "nested_laplace"
  ))
  # Build the same prior the front door builds (validate_hsgp then the spec ->
  # prior converter), then call the fitter directly: the routing adds no math.
  vspec  <- tulpa:::validate_hsgp(spec, s$data)
  prior  <- tulpa:::.spatial_spec_to_nl_prior(vspec)
  direct <- tulpa_nested_laplace(
    s$data$y, s$data$ntrials, model.matrix(y ~ x, s$data),
    prior = prior, family = "binomial"
  )
  expect_equal(via$theta_grid,   direct$theta_grid)
  expect_equal(via$log_marginal, direct$log_marginal)
  expect_equal(via$theta_mean,   direct$theta_mean)
})

test_that("an HSGP spec rejects a spatial(col) term and a bare list", {
  s <- sim_gp_binomial(n_loc = 20L, reps = 1L)
  s$data$site <- factor(seq_len(nrow(s$data)))
  # Continuous field + a spatial(col) term is contradictory.
  expect_error(
    suppressMessages(tulpa(
      y ~ x + spatial(site), data = s$data, family = "binomial",
      n_trials = s$data$ntrials, spatial = spatial_hsgp(~ lon + lat),
      mode = "nested_laplace")),
    "coordinate columns"
  )
  # A bare list (type = "hsgp") is not a spatial_hsgp() spec.
  expect_error(
    suppressMessages(tulpa(
      y ~ x, data = s$data, family = "binomial", n_trials = s$data$ntrials,
      spatial = list(type = "hsgp"), mode = "nested_laplace")),
    "spatial_hsgp"
  )
})

test_that("cpp_hsgp_basis_2d matches the make_hsgp_basis_2d oracle", {
  # Independent check that the exported basis builder agrees with the
  # hand-written 2D HSGP basis used in test-nested-laplace-gp.R. Mirrors
  # setup_hsgp_2d: L = c * range / 2 (floored at 0.1), centred coords,
  # phi_j(x) = (1/sqrt(L)) sin(pi j (x + L)/(2L)), lambda = (pi j / 2L)^2.
  oracle <- function(coords, m, cc) {
    x <- coords[, 1]; y <- coords[, 2]
    L1 <- max(cc * (max(x) - min(x)) / 2, 0.1)
    L2 <- max(cc * (max(y) - min(y)) / 2, 0.1)
    xs <- x - (max(x) + min(x)) / 2
    ys <- y - (max(y) + min(y)) / 2
    M <- m * m
    phi <- matrix(0, nrow(coords), M); lam <- numeric(M)
    for (j1 in seq_len(m)) for (j2 in seq_len(m)) {
      idx <- (j1 - 1) * m + j2
      lam[idx] <- (pi * j1 / (2 * L1))^2 + (pi * j2 / (2 * L2))^2
      phi[, idx] <- ((1 / sqrt(L1)) * sin(pi * j1 * (xs + L1) / (2 * L1))) *
                    ((1 / sqrt(L2)) * sin(pi * j2 * (ys + L2) / (2 * L2)))
    }
    list(phi = phi, lam = lam)
  }
  set.seed(3)
  coords <- cbind(runif(30), runif(30))
  got <- tulpa:::cpp_hsgp_basis_2d(coords, 5L, 1.5)
  want <- oracle(coords, 5L, 1.5)
  expect_equal(unname(got$lambda_eig), want$lam, tolerance = 1e-10)
  expect_equal(unname(got$phi_basis), unname(want$phi), tolerance = 1e-10)
})
