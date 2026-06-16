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

test_that("mode = laplace threads offset() through a spatial field (gcol33/tulpa#72)", {
  skip_on_cran()
  s  <- sim_areal_binomial()
  d  <- s$data
  sp <- list(type = "icar", adjacency = s$W)
  cc <- 0.5
  base <- tulpa(y ~ x + spatial(region), data = d, family = "binomial",
                n_trials = d$ntrials, spatial = sp, mode = "laplace")

  d0 <- d; d0$o <- 0
  zero <- tulpa(y ~ x + spatial(region) + offset(o), data = d0,
                family = "binomial", n_trials = d$ntrials, spatial = sp,
                mode = "laplace")
  # offset = 0 reproduces the no-offset spatial Laplace fit exactly.
  expect_equal(zero$mode, base$mode, tolerance = 1e-8)

  dc <- d; dc$o <- cc
  con <- tulpa(y ~ x + spatial(region) + offset(o), data = dc,
               family = "binomial", n_trials = d$ntrials, spatial = sp,
               mode = "laplace")
  # A constant offset c is absorbed by the intercept (shifts by -c); the slope
  # and centered field are unchanged to the weak intercept-prior tolerance.
  expect_equal(con$mode[1], base$mode[1] - cc, tolerance = 1e-2)
  expect_equal(con$mode[2], base$mode[2], tolerance = 1e-2)
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
  # a temporal() formula term is rejected in favour of the `temporal=` argument.
  expect_error(
    tulpa(y ~ x + temporal(region), data = s$data, family = "binomial",
          n_trials = s$data$ntrials, mode = "gibbs"),
    "drop the temporal"
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

# --- RSR front-door routing (Increment 5) --------------------------------
# spatial_rsr() wraps an areal spec and flags $rsr; tulpa() routes it as its own
# gibbs-only areal type (auto picks Gibbs for binomial), building the unit-level
# projector from restrict_to so the field is orthogonal to the covariates.

test_that("tulpa() routes an RSR field to the binomial Gibbs sampler (auto)", {
  skip_on_cran()
  s <- sim_areal_binomial()
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = spatial_rsr(spatial_car(s$W, level = "obs"), restrict_to = ~ x),
    mode = "auto",
    control = list(iter = 3000L, warmup = 1500L)
  )
  # Routing: auto picked the (Tier 1) RSR Polya-Gamma sampler, not nested/plain.
  expect_equal(fit$backend, "gibbs")
  expect_equal(fit$inference_mode, "exact")
  expect_true(all(c("beta", "spatial", "spatial_raw", "tau") %in% names(fit)))
  expect_equal(ncol(fit$spatial), s$n_units)
  # Recovery: RSR orthogonalises the field to x, so the slope is uncontaminated.
  beta_hat <- colMeans(fit$beta)
  expect_lt(abs(beta_hat[2] - s$beta[2]), 0.30)
})

test_that("tulpa() rejects a non-binomial RSR field", {
  s <- sim_areal_binomial(reps = 2L)
  s$data$count <- rpois(nrow(s$data), 3)
  expect_error(
    tulpa(count ~ x + spatial(region), data = s$data, family = "poisson",
          spatial = spatial_rsr(spatial_car(s$W, level = "obs"), restrict_to = ~ x),
          mode = "auto", control = list(iter = 50L, warmup = 25L)),
    "binomial"
  )
})

# --- Nested-Laplace spatial routing (Increment 4) ------------------------
# mode = "nested_laplace" / "structured" / "auto" (non-binomial-Gibbs) route an
# areal field through tulpa_nested_laplace(), which INTEGRATES the spatial
# hyperparameter -- the designed Tier 2 path, distinct from the conditional
# tulpa_laplace(spatial=) fit (mode = "laplace") that fixes it. The routing
# layer must add no math: the front-door grid/log-marginal are identical to a
# direct tulpa_nested_laplace() call on the same prior.

test_that("mode = nested_laplace integrates an areal field through tulpa()", {
  skip_on_cran()
  s <- sim_areal_binomial()
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "icar", adjacency = s$W),
    mode = "nested_laplace",
    control = list(keep_grid_hessians = TRUE)
  )
  expect_equal(fit$backend, "nested_laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_s3_class(fit, "tulpa_fit")
  expect_s3_class(fit, "tulpa_nested_laplace")
  # The spatial precision tau was integrated, not conditioned on.
  expect_true(all(is.finite(fit$theta_mean)))
  expect_true(all(is.finite(fit$weights)) && abs(sum(fit$weights) - 1) < 1e-8)

  # Recovery: marginalize beta over the tau grid (grid modes weighted by the
  # integration weights), never a plug-in MAP (CLAUDE.md "Marginalize Derived
  # Quantities"). grid_modes[[k]] holds the length-p fixed-effect block.
  w  <- fit$weights
  bm <- do.call(rbind, lapply(fit$grid_modes, function(m) m[1:2]))
  beta_hat <- as.numeric(crossprod(w, bm))
  expect_lt(abs(beta_hat[1] - s$beta[1]), 0.45)   # intercept
  expect_lt(abs(beta_hat[2] - s$beta[2]), 0.30)   # slope
})

test_that("the nested spatial route is numerically identical to a direct call", {
  s <- sim_areal_binomial(reps = 3L)
  via <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "icar", adjacency = s$W),
    mode = "nested_laplace"
  )
  # Build the same areal prior the front door builds, then call the fitter
  # directly: the routing layer adds no math.
  spec <- list(type = "icar", adjacency = s$W,
               spatial_idx = tulpa:::.resolve_unit_index(
                 s$data$region, "region", nrow(s$W)))
  prior <- tulpa:::.spatial_spec_to_nl_prior(spec)
  direct <- tulpa_nested_laplace(
    y = s$data$y, n_trials = s$data$ntrials,
    X = model.matrix(y ~ x, s$data),
    prior = prior, family = "binomial"
  )
  expect_equal(via$theta_grid,   direct$theta_grid)
  expect_equal(via$log_marginal, direct$log_marginal)
  expect_equal(via$theta_mean,   direct$theta_mean)
})

test_that("mode = structured routes an areal field to nested_laplace", {
  s <- sim_areal_binomial(reps = 2L)
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "icar", adjacency = s$W),
    mode = "structured"
  )
  expect_equal(fit$backend, "nested_laplace")
  expect_equal(fit$inference_tier, 2L)
})

test_that("auto integrates a non-binomial areal field via nested Laplace", {
  # A non-binomial areal field is not the binomial Gibbs case, so auto picks
  # the nested-Laplace Tier 2 path (not the conditional Laplace at fixed tau,
  # and not HMC). y in 0..5 are valid Poisson counts.
  s <- sim_areal_binomial(reps = 2L)
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "poisson",
    spatial = list(type = "icar", adjacency = s$W),
    mode = "auto"
  )
  expect_equal(fit$backend, "nested_laplace")
  expect_equal(fit$inference_mode, "structured")
})

test_that("nested_laplace bym2 packs the scale factor and wires through", {
  s <- sim_areal_binomial(reps = 2L)
  fit <- tulpa(
    y ~ x + spatial(region), data = s$data, family = "binomial",
    n_trials = s$data$ntrials,
    spatial = list(type = "bym2", adjacency = s$W, scale_factor = 1.0),
    mode = "nested_laplace"
  )
  expect_equal(fit$backend, "nested_laplace")
  # BYM2 integrates a 2D (sigma, rho) grid.
  expect_equal(ncol(fit$theta_grid), 2L)
  expect_true(all(is.finite(fit$theta_mean)))
})

test_that("a continuous type given with an areal spatial(col) term is rejected", {
  # A continuous field is addressed by coordinates, not a unit column, so a
  # spatial(col) term alongside a continuous type is contradictory. (Genuinely
  # continuous routing -- gp/nngp via spatial_gp() -- lives in
  # test-tulpa-spatial-gp-frontdoor.R; hsgp/spde "not yet routed" too.)
  s <- sim_areal_binomial(reps = 1L)
  expect_error(
    tulpa(y ~ x + spatial(region), data = s$data, family = "binomial",
          n_trials = s$data$ntrials, spatial = list(type = "gp"),
          mode = "nested_laplace"),
    "coordinate columns"
  )
})
