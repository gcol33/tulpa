# test-spde-nuts-joint.R
# Parameter recovery for joint NUTS over (log_kappa, log_tau, z, beta,
# log_phi) on simulated SPDE Gaussian data. Companion to the per-seed
# smoke tests in test-spde-nuts.R, which condition on fixed (range, sigma);
# this file exercises the joint path landed across commits 49a30ad ->
# 1a3f643 (NC z->w transform + PC prior on (range, sigma) in
# (log_kappa, log_tau)).
#
# The CRITICAL "Statistical Code Needs Recovery Tests, Not Smoke Tests"
# rule (CLAUDE.md) demands per-seed bias bounds + 95% CI coverage. Joint
# NUTS is currently expensive (sparse Cholesky every gradient call), so
# we ship two tiers:
#
#   * "slim recovery" -- 3 seeds on a coarse mesh, modest iter count.
#     Runs in devtools::test() under NOT_CRAN=true; skipped on CRAN.
#     Verifies the joint path is calibrated within wide tolerances and
#     catches structural regressions (sign flips, dropped priors).
#
#   * "full recovery" -- 10 seeds on a fine mesh, longer chains, tight
#     tolerances. Gated behind Sys.getenv("TULPA_FULL_RECOVERY") == "true"
#     because a single run is multi-hour. Use this before any release or
#     after touching the SPDE log-post, PC prior, or NC transform.
#
# Helpers below are shared between tiers.

helper_make_spde_spec <- function(coords, max_edge = c(0.2, 0.5),
                                  cutoff = 0.08, nu = 1,
                                  prior_range = c(0.5, 0.5),
                                  prior_sigma = c(1, 0.5)) {
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = max_edge,
                              cutoff = cutoff)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spatial_spde_custom(
    C = fem$c0, G = fem$g1, A = A, nu = nu,
    prior_range = prior_range, prior_sigma = prior_sigma
  )
}

# Simulate a Matern SPDE field on a mesh at a target (range, sigma).
# Builds the precision Q from the FEM matrices and draws w ~ N(0, Q^{-1})
# via the inner sparse-Cholesky path. Integer alpha = nu + 1 only (fractional
# nu is gated, gcol33/tulpa#71):
#   Q = tau^2 (kappa^2 C0 + G1) diag(1/C0) (kappa^2 C0 + G1)
# Returns the field evaluated at every mesh node.
simulate_spde_field <- function(spec, range_true, sigma_true, seed) {
  set.seed(seed)
  nu    <- spec$nu
  kappa <- sqrt(8 * nu) / range_true
  tau   <- 1 / (sqrt(4 * pi) * kappa * sigma_true)

  C0  <- Matrix::Diagonal(x = spec$C0_diag)
  G1  <- Matrix::sparseMatrix(i = spec$G1_i + 1L, p = spec$G1_p,
                              x = spec$G1_x, dims = c(spec$n_mesh, spec$n_mesh),
                              index1 = TRUE)

  K <- (kappa^2) * C0 + G1
  Q <- (tau^2) * Matrix::crossprod(K, Matrix::solve(C0, K))

  L_chol <- Matrix::Cholesky(Matrix::forceSymmetric(Q), LDL = FALSE,
                             perm = FALSE)
  z      <- rnorm(spec$n_mesh)
  w      <- as.numeric(Matrix::solve(L_chol, z, system = "Lt"))
  w
}

run_one_replicate <- function(seed, range_true, sigma_true, beta0_true,
                              beta1_true, sigma_obs_true, n_obs,
                              n_iter, n_warmup,
                              mesh_max_edge = c(0.20, 0.5),
                              mesh_cutoff   = 0.08,
                              adapt_delta   = 0.9,
                              nu            = 1) {
  set.seed(seed * 17L + 1L)
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec   <- helper_make_spde_spec(
    coords, max_edge = mesh_max_edge, cutoff = mesh_cutoff, nu = nu,
    prior_range = c(0.1, 0.05),   # broad: P(range < 0.1) = 0.05
    prior_sigma = c(3.0, 0.05)    # broad: P(sigma > 3)   = 0.05
  )

  w_true <- simulate_spde_field(spec, range_true, sigma_true,
                                seed = seed * 17L + 2L)
  w_true <- w_true - mean(w_true)
  x_cov  <- runif(n_obs, -1, 1)
  X      <- cbind(1, x_cov)
  eta    <- beta0_true + beta1_true * x_cov + as.numeric(spec$A %*% w_true)
  y      <- eta + rnorm(n_obs, 0, sigma_obs_true)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "gaussian",
    joint  = TRUE,
    prior_range  = c(0.1, 0.05),
    prior_sigma  = c(3.0, 0.05),
    log_phi_init = log(sigma_obs_true),
    control = list(n_iter = n_iter, n_warmup = n_warmup,
                   adapt_delta = adapt_delta, seed = as.integer(seed))
  )

  list(
    beta = colMeans(fit$draws[, c("beta[1]", "beta[2]"), drop = FALSE]),
    beta_ci = list(
      b0 = unname(stats::quantile(fit$draws[, "beta[1]"], c(0.025, 0.975))),
      b1 = unname(stats::quantile(fit$draws[, "beta[2]"], c(0.025, 0.975)))
    ),
    range_hat = mean(fit$range_draws),
    range_ci  = unname(stats::quantile(fit$range_draws, c(0.025, 0.975))),
    sigma_hat = mean(fit$sigma_draws),
    sigma_ci  = unname(stats::quantile(fit$sigma_draws, c(0.025, 0.975))),
    phi_mean  = fit$phi_summary[["mean"]],
    accept    = mean(fit$accept_prob),
    divergent = sum(fit$divergent),
    n_iter    = nrow(fit$draws),
    n_mesh    = spec$n_mesh
  )
}

# Shared assertion block. `tier_tag` is just a label baked into expect_*
# info strings so failure messages point at the right tier.
assert_recovery <- function(reps,
                            range_true, sigma_true,
                            beta0_true, beta1_true, sigma_obs_true,
                            tol_beta, tol_log_rs, tol_log_phi,
                            tol_bias_beta, tol_bias_log_rs,
                            cover_min, accept_min, divergent_frac,
                            tier_tag) {
  n_seeds <- length(reps)

  beta0_hats   <- vapply(reps, function(r) r$beta[1], numeric(1))
  beta1_hats   <- vapply(reps, function(r) r$beta[2], numeric(1))
  range_hats   <- vapply(reps, function(r) r$range_hat, numeric(1))
  sigma_hats   <- vapply(reps, function(r) r$sigma_hat, numeric(1))
  phi_hats     <- vapply(reps, function(r) r$phi_mean, numeric(1))
  accept_means <- vapply(reps, function(r) r$accept, numeric(1))
  diverg_sum   <- vapply(reps, function(r) r$divergent, numeric(1))
  n_iter_each  <- vapply(reps, function(r) r$n_iter, integer(1))

  # --- Per-seed bias bounds ---
  expect_true(all(abs(beta0_hats - beta0_true) < tol_beta),
              info = sprintf("[%s] beta0 errs: %s", tier_tag,
                             paste(round(beta0_hats - beta0_true, 3),
                                   collapse = ", ")))
  expect_true(all(abs(beta1_hats - beta1_true) < tol_beta),
              info = sprintf("[%s] beta1 errs: %s", tier_tag,
                             paste(round(beta1_hats - beta1_true, 3),
                                   collapse = ", ")))
  expect_true(all(abs(log(range_hats / range_true)) < tol_log_rs),
              info = sprintf("[%s] range log-errs: %s", tier_tag,
                             paste(round(log(range_hats / range_true), 3),
                                   collapse = ", ")))
  expect_true(all(abs(log(sigma_hats / sigma_true)) < tol_log_rs),
              info = sprintf("[%s] sigma log-errs: %s", tier_tag,
                             paste(round(log(sigma_hats / sigma_true), 3),
                                   collapse = ", ")))
  expect_true(all(abs(log(phi_hats / sigma_obs_true)) < tol_log_phi),
              info = sprintf("[%s] phi log-errs: %s", tier_tag,
                             paste(round(log(phi_hats / sigma_obs_true), 3),
                                   collapse = ", ")))

  # --- Across-seed bias ---
  expect_lt(abs(mean(beta0_hats) - beta0_true),                tol_bias_beta)
  expect_lt(abs(mean(beta1_hats) - beta1_true),                tol_bias_beta)
  expect_lt(abs(mean(log(range_hats / range_true))),           tol_bias_log_rs)
  expect_lt(abs(mean(log(sigma_hats / sigma_true))),           tol_bias_log_rs)

  # --- 95% CI coverage ---
  covers_beta0 <- vapply(reps, function(r)
    r$beta_ci$b0[1] <= beta0_true && beta0_true <= r$beta_ci$b0[2],
    logical(1))
  covers_beta1 <- vapply(reps, function(r)
    r$beta_ci$b1[1] <= beta1_true && beta1_true <= r$beta_ci$b1[2],
    logical(1))
  covers_range <- vapply(reps, function(r)
    r$range_ci[1] <= range_true && range_true <= r$range_ci[2],
    logical(1))
  covers_sigma <- vapply(reps, function(r)
    r$sigma_ci[1] <= sigma_true && sigma_true <= r$sigma_ci[2],
    logical(1))

  expect_gte(sum(covers_beta0), cover_min)
  expect_gte(sum(covers_beta1), cover_min)
  expect_gte(sum(covers_range), cover_min)
  expect_gte(sum(covers_sigma), cover_min)

  # --- Sampler health (lenient: only that nothing degenerated outright) ---
  expect_true(all(accept_means > accept_min),
              info = sprintf("[%s] accept rates: %s", tier_tag,
                             paste(round(accept_means, 3), collapse = ", ")))
  expect_true(all(diverg_sum < divergent_frac * n_iter_each),
              info = sprintf("[%s] divergent counts (of %s iter): %s",
                             tier_tag,
                             paste(n_iter_each, collapse = ", "),
                             paste(diverg_sum, collapse = ", ")))
}

# ---------------------------------------------------------------------------
# Slim recovery: cheap calibration check, runs under devtools::test().
# Coarse mesh (~60 nodes) keeps per-iter cost ~0.4s; 3 seeds * 500 iter
# total ~ a few minutes. Tolerances are intentionally wider than the full
# tier because n_seeds = 3 makes coverage rate a coarse estimator.
# ---------------------------------------------------------------------------
test_that("joint NUTS recovers (range, sigma, beta) -- slim tier", {
  skip_if_not_slow()
  skip_if_not_installed("fmesher")
  skip_if_not_installed("Matrix")

  range_true     <- 0.4
  sigma_true     <- 1.0
  beta0_true     <- 0.5
  beta1_true     <- -0.8
  sigma_obs_true <- 0.3
  n_obs          <- 150L
  n_seeds        <- 3L
  n_iter         <- 300L
  n_warmup       <- 200L

  reps <- vector("list", n_seeds)
  for (k in seq_len(n_seeds)) {
    reps[[k]] <- run_one_replicate(
      seed           = k,
      range_true     = range_true,
      sigma_true     = sigma_true,
      beta0_true     = beta0_true,
      beta1_true     = beta1_true,
      sigma_obs_true = sigma_obs_true,
      n_obs          = n_obs,
      n_iter         = n_iter,
      n_warmup       = n_warmup,
      mesh_max_edge  = c(0.3, 0.7),
      mesh_cutoff    = 0.15,
      adapt_delta    = 0.9
    )
  }

  assert_recovery(
    reps,
    range_true = range_true, sigma_true = sigma_true,
    beta0_true = beta0_true, beta1_true = beta1_true,
    sigma_obs_true = sigma_obs_true,
    tol_beta        = 0.40,   # wider than full -- short chains, small n
    tol_log_rs      = 1.20,   # range/sigma weakly identified from n=3
    tol_log_phi     = 0.60,
    tol_bias_beta   = 0.25,
    tol_bias_log_rs = 0.60,
    cover_min       = 2L,     # 2/3 CIs cover -- weakest meaningful bound
    accept_min      = 0.30,
    divergent_frac  = 0.30,   # short warmup leaves divergences
    tier_tag        = "slim"
  )
})

# ---------------------------------------------------------------------------
# Full recovery: 10 seeds, finer mesh, longer chains, tight tolerances.
# Gated -- a single run is multi-hour. Run before release or after any
# change to the SPDE log-post / PC prior / NC transform.
#
#   NOT_CRAN=true TULPA_FULL_RECOVERY=true Rscript -e \
#     'testthat::test_file("tests/testthat/test-spde-nuts-joint.R")'
# ---------------------------------------------------------------------------
test_that("joint NUTS recovers (range, sigma, beta) -- full tier", {
  skip_if_not_slow()
  skip_if_not_installed("fmesher")
  skip_if_not_installed("Matrix")
  if (!identical(Sys.getenv("TULPA_FULL_RECOVERY"), "true")) {
    skip("Set TULPA_FULL_RECOVERY=true to run the multi-hour full recovery test")
  }

  range_true     <- 0.4
  sigma_true     <- 1.0
  beta0_true     <- 0.5
  beta1_true     <- -0.8
  sigma_obs_true <- 0.3
  n_obs          <- 250L
  n_seeds        <- 10L
  n_iter         <- 600L
  n_warmup       <- 400L

  reps <- vector("list", n_seeds)
  for (k in seq_len(n_seeds)) {
    reps[[k]] <- run_one_replicate(
      seed           = k,
      range_true     = range_true,
      sigma_true     = sigma_true,
      beta0_true     = beta0_true,
      beta1_true     = beta1_true,
      sigma_obs_true = sigma_obs_true,
      n_obs          = n_obs,
      n_iter         = n_iter,
      n_warmup       = n_warmup,
      mesh_max_edge  = c(0.20, 0.5),
      mesh_cutoff    = 0.08,
      adapt_delta    = 0.9
    )
  }

  assert_recovery(
    reps,
    range_true = range_true, sigma_true = sigma_true,
    beta0_true = beta0_true, beta1_true = beta1_true,
    sigma_obs_true = sigma_obs_true,
    tol_beta        = 0.25,
    tol_log_rs      = 0.80,
    tol_log_phi     = 0.40,
    tol_bias_beta   = 0.10,
    tol_bias_log_rs = 0.25,
    cover_min       = 7L,     # 7/10 = 70%
    accept_min      = 0.40,
    divergent_frac  = 0.10,
    tier_tag        = "full"
  )
})

test_that("joint NUTS round-trip: output structure matches contract", {
  skip_if_not_installed("fmesher")
  skip_if_not_slow()
  set.seed(42)
  n_obs  <- 80
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec   <- helper_make_spde_spec(
    coords, max_edge = c(0.3, 0.7), cutoff = 0.15,
    prior_range = c(0.1, 0.05), prior_sigma = c(3.0, 0.05)
  )
  x_cov <- runif(n_obs, -1, 1)
  X     <- cbind(1, x_cov)
  y     <- x_cov + rnorm(n_obs, 0, 0.3)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "gaussian",
    joint  = TRUE,
    control = list(n_iter = 100L, n_warmup = 50L, seed = 1L)
  )

  expect_true(fit$joint_hypers)
  expect_true(fit$joint)
  expect_true("log_kappa" %in% colnames(fit$draws))
  expect_true("log_tau"   %in% colnames(fit$draws))
  expect_true(any(grepl("^z\\[", colnames(fit$draws))))
  expect_false(any(grepl("^w\\[", colnames(fit$draws))))
  expect_true(is.matrix(fit$w_draws))
  expect_equal(ncol(fit$w_draws), spec$n_mesh)
  expect_equal(nrow(fit$w_draws), nrow(fit$draws))
  expect_true(is.numeric(fit$range_draws))
  expect_true(is.numeric(fit$sigma_draws))
  expect_length(fit$range_draws, nrow(fit$draws))
  expect_length(fit$sigma_draws, nrow(fit$draws))
  expect_true(all(fit$range_draws > 0))
  expect_true(all(fit$sigma_draws > 0))
  expect_named(fit$range_summary, c("mean", "median", "q05", "q95"))
  expect_named(fit$sigma_summary, c("mean", "median", "q05", "q95"))
})

# ---------------------------------------------------------------------------
# Fractional alpha (fractional nu) builds a valid spec now that the operator-
# based rational SPDE is wired through fit_spde() (gcol33/tulpa#71). The NUTS
# sampler stays integer-only, so the rejection lives at the fit call.
# ---------------------------------------------------------------------------
test_that("fractional nu is rejected for the joint-NUTS SPDE path (gcol33/tulpa#71)", {
  skip_if_not_installed("fmesher")
  set.seed(2026)
  coords <- cbind(runif(40), runif(40))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.30, 0.7), cutoff = 0.15,
                                nu = 0.5)
  y <- rnorm(40)
  X <- matrix(1.0, nrow = 40, ncol = 1)
  expect_error(
    tulpa_nuts_spde(y = y, X = X, spatial = spec, family = "gaussian",
                    joint = TRUE, control = list(n_iter = 50L, n_warmup = 25L)),
    "fractional"
  )
})

test_that("joint NUTS rejects PC anchors out of range", {
  skip_if_not_installed("fmesher")
  set.seed(2)
  coords <- cbind(runif(20), runif(20))
  spec   <- helper_make_spde_spec(coords, max_edge = c(0.3, 0.6), cutoff = 0.15)
  y <- rnorm(20)
  X <- matrix(1.0, nrow = 20, ncol = 1)
  expect_error(
    tulpa_nuts_spde(y = y, X = X, spatial = spec, family = "gaussian",
                    joint = TRUE, prior_range = c(-0.1, 0.05),
                    control = list(n_iter = 50L, n_warmup = 25L)),
    "prior_range"
  )
  expect_error(
    tulpa_nuts_spde(y = y, X = X, spatial = spec, family = "gaussian",
                    joint = TRUE, prior_sigma = c(1.0, 1.5),
                    control = list(n_iter = 50L, n_warmup = 25L)),
    "prior_sigma"
  )
})
