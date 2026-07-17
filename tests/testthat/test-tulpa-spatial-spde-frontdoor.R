# SPDE (Matern via FEM mesh) nested routing through the tulpa() front door
# (dev_notes/plan_gibbs_spatial_frontdoor.md, increment 4 -- the final
# continuous family). SPDE is coordinate-addressed like gp/nngp/hsgp -- no
# spatial(col) term -- but carries its OWN nested-Laplace engine: fit_spde()
# rebuilds the Matern precision Q(range, sigma) per node via the FEM Q-builder
# and integrates (range, sigma) with a CCD / grid design in R, not the generic
# registry grid that tulpa_nested_laplace drives. So a nested mode selects the
# generic nested_laplace backend, and tulpa() redirects an SPDE field to the
# dedicated `spde` backend (fit_spde). The routing adds no math: it is identical
# to a direct fit_spde() call on the same spec.

# Build an SPDE spec (via fmesher) plus a Poisson panel with a fixed
# intercept + slope and a Matern field on the mesh nodes. A tight PC prior on
# sigma keeps the joint (range, sigma) posterior mode interior -- the SPDE
# marginal likelihood drifts upward in sigma at the Laplace mode otherwise, the
# same reason test-spde-ccd.R tightens prior_sigma.
make_spde_panel <- function(n_obs = 250L, range_true = 0.3, sigma_true = 0.6,
                            beta = c(2.0, 0.5), seed = 11L) {
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.4), cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                              prior_range = c(0.3, 0.5),
                              prior_sigma = c(0.6, 0.05))
  w <- as.numeric(rnorm(spec$n_mesh, 0, sigma_true)); w <- w - mean(w)
  x <- rnorm(n_obs)
  eta <- beta[1] + beta[2] * x + as.numeric(spec$A %*% w)
  y <- rpois(n_obs, lambda = exp(eta))
  list(data = data.frame(y = y, x = x), spec = spec, beta = beta)
}

test_that("mode = nested_laplace integrates an SPDE field through tulpa()", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  s <- make_spde_panel()
  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = s$data, family = "poisson",
    spatial = s$spec, mode = "nested_laplace"
  )))
  # Redirected from the generic nested_laplace backend to the SPDE engine.
  expect_equal(fit$backend, "spde")
  expect_equal(fit$inference_tier, 2L)
  expect_s3_class(fit, "tulpa_fit")
  # (range, sigma) were integrated, not conditioned on.
  expect_true(fit$nested$method %in% c("ccd", "grid"))
  expect_equal(sum(fit$nested$weights), 1, tolerance = 1e-8)
  expect_true(is.finite(fit$nested$range_mean) && fit$nested$range_mean > 0)
  expect_true(is.finite(fit$nested$sigma_mean) && fit$nested$sigma_mean > 0)
  # Recovery (when CCD found an interior mode): beta at the joint posterior mode.
  # On a degenerate Hessian CCD falls back to the grid and drops `beta`, which
  # is correct behaviour -- only assert recovery when the mode is usable.
  if (identical(fit$nested$method, "ccd")) {
    expect_lt(abs(fit$beta[1] - s$beta[1]), 0.5)   # intercept
    expect_lt(abs(fit$beta[2] - s$beta[2]), 0.3)   # slope
  }
})

test_that("mode = laplace fits an SPDE field alongside a (1 | g) RE term (issue #74)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  s <- make_spde_panel()
  d <- s$data
  d$g <- factor(rep_len(seq_len(8L), nrow(d)))

  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x + (1 | g), data = d, family = "poisson",
    spatial = s$spec, mode = "laplace"
  )))

  expect_s3_class(fit, "tulpa_fit")
  # Latent mode is [beta (2), re (8 groups), w_mesh (n_mesh)].
  expect_length(fit$mode, 2L + 8L + s$spec$n_mesh)
  expect_lt(abs(fit$mode[2] - s$beta[2]), 0.35)   # slope recovers under the RE block
})

test_that("structured and auto route an SPDE field to the spde backend", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  s <- make_spde_panel(n_obs = 150L, seed = 3L)
  for (m in c("structured", "auto")) {
    fit <- suppressWarnings(suppressMessages(tulpa(
      y ~ x, data = s$data, family = "poisson", spatial = s$spec, mode = m
    )))
    expect_equal(fit$backend, "spde", info = m)
    expect_equal(fit$inference_mode, "structured", info = m)
    expect_equal(fit$inference_tier, 2L, info = m)
  }
})

test_that("the SPDE front-door route is numerically identical to a direct fit_spde()", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  s <- make_spde_panel(n_obs = 150L, seed = 7L)
  via <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = s$data, family = "poisson", spatial = s$spec,
    mode = "nested_laplace"
  )))
  # The front door builds X = model.matrix(~ x) and calls fit_spde() with the
  # default (ccd) backend -- so a direct call on the same spec must agree
  # bit-for-bit (fit_spde() is deterministic: no RNG in the integration).
  X <- model.matrix(~ x, data = s$data)
  direct <- suppressWarnings(fit_spde(
    y = s$data$y, X = X, spatial = s$spec, family = "poisson",
    control = list(method = "ccd")
  ))
  expect_equal(via$nested$method, direct$nested$method)
  expect_equal(via$nested$range_mean, direct$nested$range_mean)
  expect_equal(via$nested$sigma_mean, direct$nested$sigma_mean)
  expect_equal(via$nested$weights, direct$nested$weights)
  expect_equal(via$log_marginal, direct$log_marginal)
  expect_equal(via$beta, direct$beta)
})

test_that("mode = laplace conditions an SPDE field at fixed hyperparameters", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  s <- make_spde_panel(n_obs = 120L, seed = 9L)
  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = s$data, family = "poisson", spatial = s$spec, mode = "laplace"
  )))
  # Conditional fit: stays on the tulpa_laplace path (no hyperparameter grid).
  expect_equal(fit$backend, "laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_null(fit$nested)
  expect_true(all(is.finite(fit$mode)))
})

test_that("the SPDE front door routes a gaussian (geostatistical) response to fit_spde()", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  s <- make_spde_panel(n_obs = 150L, seed = 21L)
  dat <- s$data
  dat$yc <- as.numeric(scale(dat$y))     # a continuous response on the same design
  via <- suppressWarnings(suppressMessages(tulpa(
    yc ~ x, data = dat, family = "gaussian", spatial = s$spec,
    mode = "nested_laplace"
  )))
  X <- model.matrix(~ x, data = dat)
  direct <- suppressWarnings(fit_spde(
    y = dat$yc, X = X, spatial = s$spec, family = "gaussian",
    control = list(method = "ccd")
  ))
  # Continuous-field geostatistics: gaussian routes through the same fit_spde()
  # engine, so the front door is numerically identical to a direct call.
  expect_true(is.finite(via$nested$range_mean) && via$nested$range_mean > 0)
  expect_true(is.finite(via$nested$sigma_mean) && via$nested$sigma_mean > 0)
  expect_equal(via$nested$range_mean, direct$nested$range_mean)
  expect_equal(via$nested$sigma_mean, direct$nested$sigma_mean)
  expect_equal(via$beta, direct$beta)
})

test_that("an SPDE spec rejects a spatial(col) term, a bad dimension, RE, and a bad family", {
  skip_if_not_installed("fmesher")
  s <- make_spde_panel(n_obs = 100L, seed = 13L)
  d <- s$data
  d$region <- rep(seq_len(10L), length.out = nrow(d))

  # A coordinate-addressed field must not carry a spatial(col) term.
  expect_error(
    suppressMessages(tulpa(y ~ x + spatial(region), data = d, family = "poisson",
                           spatial = s$spec, mode = "nested_laplace")),
    "coordinate columns"
  )
  # Dimension mismatch: spec built on 100 obs, fit on 50 rows.
  expect_error(
    suppressMessages(tulpa(y ~ x, data = d[seq_len(50L), ], family = "poisson",
                           spatial = s$spec, mode = "nested_laplace")),
    "projector matrix A"
  )
  # A single random-intercept `(1 | g)` alongside the SPDE field IS supported
  # (see test-spde-re.R); a random SLOPE is not.
  d$g <- factor(rep(seq_len(5L), length.out = nrow(d)))
  expect_error(
    suppressMessages(tulpa(y ~ x + (1 + x | g), data = d, family = "poisson",
                           spatial = s$spec, mode = "nested_laplace")),
    "slope|random-intercept"
  )
  # The SPDE kernel supports binomial / poisson / neg_binomial_2 / gaussian; a
  # valid family outside that set (beta, on a (0,1) response) is rejected.
  d$p <- runif(nrow(d), 0.1, 0.9)
  expect_error(
    suppressMessages(tulpa(p ~ x, data = d, family = "beta",
                           spatial = s$spec, mode = "nested_laplace")),
    "SPDE supports family"
  )
})

test_that("generic accessors work on an auto-mode SPDE fit (coef/vcov/confint/summary)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  s <- make_spde_panel(n_obs = 150L, seed = 17L)
  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = s$data, family = "poisson", spatial = s$spec, mode = "auto"
  )))
  expect_equal(fit$backend, "spde")

  # The nested SPDE return used to carry no mode/H_beta/means, so every
  # coefficient-facing generic errored on the front-door route.
  cf <- coef(fit)
  expect_length(cf, 2L)
  expect_lt(abs(unname(cf[2]) - s$beta[2]), 0.4)

  V <- vcov(fit)
  expect_equal(dim(V), c(2L, 2L))
  expect_true(all(is.finite(diag(V))) && all(diag(V) > 0))

  ci <- confint(fit)
  expect_equal(nrow(ci), 2L)
  expect_true(all(ci[, 1] < ci[, 2]))

  expect_no_error(summary(fit))
})
