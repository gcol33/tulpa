# test-tulpa-entry-nested.R
# Contract for the nested-Laplace front door: a `latent(tgmrf(...))` formula
# routed through tulpa() must (a) select the nested_laplace backend, (b) be
# numerically identical to a direct tulpa_nested_laplace() call (the routing
# layer adds no math -- single source of truth), (c) recover parameters on the
# front-door path, and (d) fail loudly on incompatible backends / RE structure.

# A periodic-AR1 GMRF block (sigma, rho) over `n` latent slots, one obs per
# slot. Same closed form as the tgmrf recovery fixture.
make_ar1_block <- function(n) {
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1]); rho <- tanh(theta[2]); tau <- 1 / sigma^2
      d <- rep(tau * (1 + rho^2), n)
      o <- rep(-tau * rho, n - 1L)
      M <- Matrix::bandSparse(n, k = c(-1L, 0L, 1L), diagonals = list(o, d, o))
      M[1, n] <- -tau * rho; M[n, 1] <- -tau * rho
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior  = function(theta) 0,
    init   = c(log_sigma = 0, atanh_rho = atanh(0.5)),
    bounds = list(lower = c(log(0.3), atanh(0.0)),
                  upper = c(log(3.0), atanh(0.95))),
    name   = "ar1"
  )
}

sim_ar1_pois <- function(seed, n, sigma = 0.8, rho = 0.6, b0 = 0.2, b1 = 0.4) {
  set.seed(seed)
  z <- numeric(n); z[1] <- rnorm(1, 0, sigma)
  for (t in 2:n) z[t] <- rho * z[t - 1] + rnorm(1, 0, sigma * sqrt(1 - rho^2))
  x <- rnorm(n)
  y <- rpois(n, exp(b0 + b1 * x + z))
  data.frame(y = y, x = x)
}

test_that("tulpa() auto-selects nested_laplace when a latent block is present", {
  d   <- sim_ar1_pois(101, 60L)
  blk <- make_ar1_block(60L)
  fit <- tulpa(y ~ x + latent(blk), d, family = "poisson", mode = "auto")

  expect_s3_class(fit, "tulpa_fit")
  expect_s3_class(fit, "tulpa_nested_laplace")
  expect_equal(fit$backend, "nested_laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_equal(fit$inference_mode, "structured")
  # Block hyperparameters were integrated, not conditioned on.
  expect_length(fit$theta_mean, 2L)
  expect_true(all(is.finite(fit$theta_mean)))

  # mode = "structured" and the explicit backend name route the same way.
  expect_equal(
    tulpa(y ~ x + latent(blk), d, family = "poisson", mode = "structured")$backend,
    "nested_laplace")
  expect_equal(
    tulpa(y ~ x + latent(blk), d, family = "poisson", mode = "nested_laplace")$backend,
    "nested_laplace")
})

test_that("the formula route is numerically identical to a direct call", {
  # The routing layer must add no math: same prior, same X, same family ->
  # bit-identical grid, log-marginal, and posterior moments.
  d   <- sim_ar1_pois(102, 60L)
  blk <- make_ar1_block(60L)

  via_formula <- tulpa(y ~ x + latent(blk), d, family = "poisson",
                       mode = "nested_laplace")
  direct <- tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, 60L),
    X = model.matrix(y ~ x, d),
    prior = blk, family = "poisson"
  )

  expect_equal(via_formula$theta_grid,   direct$theta_grid)
  expect_equal(via_formula$log_marginal, direct$log_marginal)
  expect_equal(via_formula$theta_mean,   direct$theta_mean)
  expect_equal(via_formula$theta_sd,     direct$theta_sd)
})

test_that("a single random-intercept term threads through the nested path", {
  set.seed(103)
  n <- 60L
  z <- numeric(n); z[1] <- rnorm(1, 0, 0.8)
  for (t in 2:n) z[t] <- 0.6 * z[t - 1] + rnorm(1, 0, 0.8 * sqrt(1 - 0.36))
  g  <- factor(rep(seq_len(6), each = 10))
  y  <- rpois(n, exp(0.1 + rnorm(6, 0, 0.4)[g] + z))
  d  <- data.frame(y = y, g = g)
  blk <- make_ar1_block(n)

  fit <- tulpa(y ~ 1 + (1 | g) + latent(blk), d, family = "poisson",
               mode = "auto", sigma_re = 0.4)
  expect_equal(fit$backend, "nested_laplace")

  # Equivalent to passing re_idx / n_re_groups / sigma_re directly.
  direct <- tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, n), X = model.matrix(y ~ 1, d),
    prior = blk, re_idx = as.integer(g), n_re_groups = nlevels(g),
    sigma_re = 0.4, family = "poisson"
  )
  expect_equal(fit$log_marginal, direct$log_marginal)
})

test_that("latent blocks fail loudly on a non-nested backend", {
  d   <- sim_ar1_pois(104, 40L)
  blk <- make_ar1_block(40L)
  for (m in c("laplace", "mala", "exact")) {
    err <- expect_error(
      tulpa(y ~ x + latent(blk), d, family = "poisson", mode = m))
    expect_match(conditionMessage(err), "nested-Laplace|latent prior block")
  }
})

test_that("nested_laplace without a latent block errors", {
  d <- sim_ar1_pois(105, 40L)
  err <- expect_error(
    tulpa(y ~ x, d, family = "poisson", mode = "nested_laplace"))
  expect_match(conditionMessage(err), "at least one .latent")
})

test_that("the joint engine cannot be driven from a single-response formula", {
  d   <- sim_ar1_pois(106, 40L)
  blk <- make_ar1_block(40L)
  err <- expect_error(
    tulpa(y ~ x + latent(blk), d, family = "poisson",
          mode = "nested_laplace_joint"))
  expect_match(conditionMessage(err), "multiple response arms")
})

test_that("more than one random-intercept term alongside a block errors", {
  set.seed(107)
  n  <- 60L
  g  <- factor(rep(seq_len(6), each = 10))
  g2 <- factor(rep(seq_len(5), times = 12))
  d  <- data.frame(y = rpois(n, 2), g = g, g2 = g2)
  blk <- make_ar1_block(n)
  err <- expect_error(
    tulpa(y ~ (1 | g) + (1 | g2) + latent(blk), d, family = "poisson",
          mode = "auto", sigma_re = 0.4))
  expect_match(conditionMessage(err), "at most one random-intercept")
})

test_that("nested_laplace and nested_laplace_joint are registered Tier 2", {
  for (b in c("nested_laplace", "nested_laplace_joint")) {
    expect_true(b %in% INFERENCE_TIERS$structured$backends, info = b)
    expect_equal(get_backend_tier(b)$tier, 2L, info = b)
    expect_equal(BACKEND_REGISTRY[[b]]$input, "nested", info = b)
  }
  # Joint is registered (a known model-package engine) but reachable as a
  # named function; the single-response formula guard is what blocks it.
  expect_true(backend_is_reachable("nested_laplace"))
  expect_true(backend_is_reachable("nested_laplace_joint"))
})

# --- Recovery on the front-door path -------------------------------------
# Per CLAUDE.md "statistical code needs recovery tests": certify that the
# *formula* route recovers parameters, not just that it dispatches. The
# nested kernel itself is recovery-tested in test-nested-laplace-recovery.R
# and test-tgmrf-recovery.R; here we confirm those guarantees survive the
# routing layer end-to-end. Slow (multi-seed Laplace fits) -> off CRAN.
test_that("tulpa(latent(...)) recovers beta across seeds", {
  skip_on_cran()
  n        <- 60L
  b0_true  <- 0.2
  b1_true  <- 0.4
  n_seeds  <- 12L
  bhat <- matrix(NA_real_, n_seeds, 2L, dimnames = list(NULL, c("b0", "b1")))

  for (s in seq_len(n_seeds)) {
    d   <- sim_ar1_pois(2000L + s, n, b0 = b0_true, b1 = b1_true)
    blk <- make_ar1_block(n)
    fit <- tulpa(y ~ x + latent(blk), d, family = "poisson",
                 mode = "auto", control = list(max_iter = 80L, tol = 1e-7))
    # Fixed-effect posterior mean: marginalize beta over the hyperparameter
    # grid (grid modes weighted by integration weights), never a plug-in MAP.
    gm <- fit$grid_modes
    if (is.null(gm)) {
      fit <- tulpa(y ~ x + latent(blk), d, family = "poisson",
                   mode = "nested_laplace",
                   control = list(max_iter = 80L, tol = 1e-7,
                                  keep_grid_hessians = TRUE))
      gm <- fit$grid_modes
    }
    w  <- fit$weights
    b_cells <- do.call(rbind, lapply(gm, function(m) m[1:2]))
    bhat[s, ] <- as.numeric(crossprod(w, b_cells))
  }

  # The slope b1 is cleanly identified; the intercept b0 aliases with the
  # latent field's overall level (the AR1 field has a non-trivial empirical
  # mean at n = 60), so it carries a wider recovery band by construction.
  expect_lt(abs(stats::median(bhat[, "b1"]) - b1_true), 0.15)
  expect_lt(abs(stats::median(bhat[, "b0"]) - b0_true), 0.30)
})
