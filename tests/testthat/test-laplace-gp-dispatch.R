# test-laplace-gp-dispatch.R
# Pin the dispatch contract for tulpa_laplace(spatial = spatial_gp(...)).
# Builds a tiny pre-validated GP spec by hand so the test does not depend
# on the full validate_gp(data) chain or any mesh package.

# Helper: build a synthetic NNGP neighbor structure for a small problem.
make_synthetic_gp_spec <- function(n_obs = 50, k = 5L, seed = 11L) {
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  ord <- order(coords[, 1], coords[, 2])
  coords_ord <- coords[ord, , drop = FALSE]

  nn_idx  <- matrix(0L, n_obs, k)
  nn_dist <- matrix(0,  n_obs, k)
  for (i in 2:n_obs) {
    d <- sqrt((coords_ord[1:(i - 1), 1] - coords_ord[i, 1])^2 +
              (coords_ord[1:(i - 1), 2] - coords_ord[i, 2])^2)
    nc <- min(length(d), k)
    sel <- order(d)[seq_len(nc)]
    nn_idx[i, seq_len(nc)]  <- sel
    nn_dist[i, seq_len(nc)] <- d[sel]
  }

  spec <- spatial_gp(coords = ~ x1 + x2, cov = "exponential", nn = k)
  # Manually populate the fields validate_gp() would fill.
  spec$n_obs        <- n_obs
  spec$n_spatial    <- n_obs
  spec$n_unique     <- n_obs
  spec$obs_to_loc   <- seq_len(n_obs)
  spec$unique_coords <- coords_ord
  spec$coords_matrix <- coords_ord
  spec$nn           <- k
  spec$neighbor_info <- list(
    nn_idx       = nn_idx,
    nn_dist      = nn_dist,
    nn_order     = ord,
    nn_order_inv = order(ord)
  )
  list(spec = spec, n_obs = n_obs, coords = coords_ord, ord = ord)
}

test_that("dispatch_laplace_spatial routes GP specs (no longer errors)", {
  ss <- make_synthetic_gp_spec()
  X  <- matrix(1.0, ss$n_obs, 1)
  y  <- rbinom(ss$n_obs, 1, 0.5)

  msg <- tryCatch({
    dispatch_laplace_spatial(
      y = y, n_trials = rep(1L, ss$n_obs), X = X,
      re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
      spatial = ss$spec, family = "binomial", phi = 1.0,
      max_iter = 5L, tol = 1e-3, n_threads = 1L
    )
    NA_character_
  }, error = function(e) conditionMessage(e))
  if (!is.na(msg)) {
    expect_false(grepl("not yet supported", msg, fixed = TRUE))
  }
})

test_that("NNGP Laplace rejects an additional iid RE block with a clear error", {
  ss <- make_synthetic_gp_spec()
  X  <- matrix(1.0, ss$n_obs, 1)
  y  <- rbinom(ss$n_obs, 1, 0.5)

  expect_error(
    dispatch_laplace_spatial(
      y = y, n_trials = rep(1L, ss$n_obs), X = X,
      re_idx = sample.int(3, ss$n_obs, replace = TRUE),
      n_re_groups = 3L, sigma_re = 1.0,
      spatial = ss$spec, family = "binomial", phi = 1.0,
      max_iter = 5L, tol = 1e-3, n_threads = 1L
    ),
    "does not yet support an additional iid RE block"
  )
})

test_that("laplace_gp_at refuses an unvalidated spec with a clear error", {
  raw_spec <- spatial_gp(coords = ~ x1 + x2)
  expect_error(
    laplace_gp_at(
      y = rbinom(10, 1, 0.5), n_trials = rep(1L, 10),
      X = matrix(1, 10, 1), spatial = raw_spec
    ),
    "neighbor_info is NULL"
  )
})

test_that("gp_cov_type_for_laplace maps cov + nu correctly", {
  expect_equal(gp_cov_type_for_laplace(list(cov = "exponential")), 0L)
  expect_equal(gp_cov_type_for_laplace(list(cov = "matern", nu = 1.5)), 1L)
  expect_equal(gp_cov_type_for_laplace(list(cov = "matern", nu = 2.5)), 2L)

  # Reject unsupported covariances with an actionable error.
  expect_error(gp_cov_type_for_laplace(list(cov = "matern", nu = 0.5)),
               "Matern with nu in")
  expect_error(gp_cov_type_for_laplace(list(cov = "gaussian")),
               "exponential.*matern")
  expect_error(gp_cov_type_for_laplace(list(cov = "spherical")),
               "exponential.*matern")
})

test_that("tulpa_laplace(spatial = spatial_gp(...)) runs end-to-end", {
  ss <- make_synthetic_gp_spec(n_obs = 80)
  set.seed(101)
  X  <- cbind(1, rnorm(ss$n_obs))
  beta_true <- c(-0.3, 0.5)
  eta <- as.numeric(X %*% beta_true)
  y  <- rbinom(ss$n_obs, 1, plogis(eta))

  fit <- tulpa_laplace(
    y = y, n_trials = rep(1L, ss$n_obs), X = X,
    re_list = list(),
    family = "binomial",
    spatial = ss$spec,
    max_iter = 50L, tol = 1e-6, n_threads = 1L,
    return_hessian = TRUE
  )

  # mode = c(beta, spatial_effects) — spatial_effects has length n_spatial.
  expect_length(fit$mode, ncol(X) + ss$spec$n_spatial)
  expect_true(is.finite(fit$log_marginal))
  # Hessian skipped for GP (eta = X*beta + w, not X*beta + Z*u).
  expect_null(fit$H_beta)
})
