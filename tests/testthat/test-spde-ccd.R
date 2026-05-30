# test-spde-ccd.R
# Tests for the CCD-based nested-Laplace integration in fit_spde():
# joint posterior mode-find + CCD design oriented by local Hessian.

helper_make_spde_for_ccd <- function(n_obs, range_true, sigma_true, seed = 1L,
                                     max_edge = c(0.15, 0.4), cutoff = 0.05,
                                     prior_range = c(0.3, 0.5),
                                     prior_sigma = c(1, 0.01)) {
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  # Route through raw fmesher because tulpaMesh::tulpa_mesh(cutoff > 0)
  # currently emits a mesh with zero triangles (gcol33/tulpaMesh#2),
  # which collapses the SPDE precision to the orphan ridge and makes
  # log_marginal flat in (range, sigma).
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = max_edge,
                              cutoff = cutoff)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  # Tighter PC priors than the spatial_spde() defaults: the SPDE
  # marginal likelihood is monotonic in sigma even at the Laplace mode,
  # so the joint posterior only has a finite maximum when the PC prior
  # on sigma is strong enough to dominate the upward drift. Default
  # P(sigma > 1) = 0.5 leaves a heavy tail; a 1% upper-tail prior keeps
  # the mode interior for typical data.
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                              prior_range = prior_range,
                              prior_sigma = prior_sigma)

  w_true <- as.numeric(rnorm(spec$n_mesh, 0, sigma_true))
  w_true <- w_true - mean(w_true)
  eta <- as.numeric(spec$A %*% w_true)
  list(coords = coords, spec = spec, eta = eta, w_true = w_true)
}

test_that("fit_spde(method='ccd') returns a usable nested-Laplace result", {
  skip_if_not_installed("fmesher")

  # Poisson with reasonably high counts plus tight PC priors keep the
  # joint posterior over (range, sigma) well-defined. The CCD path
  # converges to an interior mode when the Hessian is well-conditioned;
  # otherwise it falls back to the rectangular grid (correct behaviour).
  d <- helper_make_spde_for_ccd(n_obs = 250, range_true = 0.3,
                                 sigma_true = 0.6, seed = 11L,
                                 prior_sigma = c(0.6, 0.05))
  y <- rpois(length(d$eta), lambda = exp(2.0 + d$eta))
  X <- matrix(1, nrow = length(y), ncol = 1)

  fit <- suppressWarnings(
    fit_spde(y, X, d$spec, family = "poisson", method = "ccd")
  )

  expect_true(fit$nested$method %in% c("ccd", "grid"))
  expect_true(all(is.finite(fit$nested$weights)))
  expect_equal(sum(fit$nested$weights), 1, tolerance = 1e-8)
  expect_true(fit$nested$range_mean > 0 && is.finite(fit$nested$range_mean))
  expect_true(fit$nested$sigma_mean > 0 && is.finite(fit$nested$sigma_mean))
})

test_that("fit_spde(method='ccd') falls back to grid on degenerate data", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords,
                              max.edge = c(0.15, 0.4), cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1)

  y <- rbinom(n_obs, 1, 0.4)        # noise only, no spatial structure
  X <- matrix(1, nrow = n_obs, ncol = 1)

  expect_warning(
    fit <- fit_spde(y, X, spec, family = "binomial",
                    n_trials = rep(1L, n_obs), n_grid = 3L,
                    method = "ccd"),
    "rectangular grid"
  )
  expect_identical(fit$nested$method, "grid")
  expect_true(all(is.finite(fit$nested$weights)))
})

test_that("fit_spde(method='ccd') matches grid integration on shared mode", {
  skip_if_not_installed("fmesher")

  d <- helper_make_spde_for_ccd(n_obs = 300, range_true = 0.3,
                                 sigma_true = 0.5, seed = 5L,
                                 prior_sigma = c(0.6, 0.05))
  y <- rpois(length(d$eta), lambda = exp(2.0 + d$eta))
  X <- matrix(1, nrow = length(y), ncol = 1)

  fit_c <- suppressWarnings(
    fit_spde(y, X, d$spec, family = "poisson", method = "ccd")
  )
  fit_g <- fit_spde(y, X, d$spec, family = "poisson",
                    method = "grid", n_grid = 7L)

  expect_identical(fit_g$nested$method, "grid")
  # Both posterior means must be finite. When CCD does not fall back to
  # grid, posterior means should agree to within ~1 decade on log-scale
  # (CCD is 9 nodes, grid is 49 — different integration rules).
  expect_true(all(is.finite(c(fit_c$nested$range_mean,
                              fit_c$nested$sigma_mean,
                              fit_g$nested$range_mean,
                              fit_g$nested$sigma_mean))))
  if (identical(fit_c$nested$method, "ccd")) {
    expect_lt(abs(log(fit_c$nested$range_mean) - log(fit_g$nested$range_mean)), 1.0)
    expect_lt(abs(log(fit_c$nested$sigma_mean) - log(fit_g$nested$sigma_mean)), 1.0)
  }
})

test_that("fit_spde reports an outer Pareto-k-hat over (range, sigma)", {
  skip_if_not_installed("fmesher")
  d <- helper_make_spde_for_ccd(n_obs = 250, range_true = 0.3, sigma_true = 0.6,
                                seed = 11L, prior_sigma = c(0.6, 0.05))
  y <- rpois(length(d$eta), lambda = exp(2.0 + d$eta))
  X <- matrix(1, nrow = length(y), ncol = 1)

  fit <- suppressWarnings(
    fit_spde(y, X, d$spec, family = "poisson", method = "ccd", k_samples = 150L))
  # Both hyperparameters are positive (log transform), so k-hat is well-defined;
  # value is data-dependent -- assert plumbing + ESS range.
  expect_true(is.finite(fit$pareto_k))
  expect_true(is.finite(fit$pareto_k_is_ess))
  expect_gt(fit$pareto_k_is_ess, 0)
  expect_equal(fit$pareto_k_scope, "outer (range, sigma) Gaussian proposal")

  off <- suppressWarnings(
    fit_spde(y, X, d$spec, family = "poisson", method = "ccd", diagnose_k = FALSE))
  expect_true(is.na(off$pareto_k))                  # gated off
})
