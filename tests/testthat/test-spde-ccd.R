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
  expect_true(all(is.finite(c(fit_c$nested$range_mean,
                              fit_c$nested$sigma_mean,
                              fit_g$nested$range_mean,
                              fit_g$nested$sigma_mean))))
  if (identical(fit_c$nested$method, "ccd")) {
    # Compare an axis only where the grid mode is interior to the grid span.
    # The grid box is fixed at prior_mode * [0.3, 3]; when the SPDE marginal is
    # only weakly identified in an axis its mode lands on the box edge, so the
    # grid integrates a censored region and is not a valid reference for the
    # wider, mode-centred CCD design. On interior axes the 9-node CCD and the
    # 49-node grid should agree to within ~1 decade on the log scale.
    g_interior <- function(best, grid)
      best > min(grid) * (1 + 1e-6) && best < max(grid) * (1 - 1e-6)
    if (g_interior(fit_g$nested$range_best, fit_g$nested$range_grid))
      expect_lt(abs(log(fit_c$nested$range_mean) - log(fit_g$nested$range_mean)), 1.0)
    if (g_interior(fit_g$nested$sigma_best, fit_g$nested$sigma_grid))
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

test_that("fit_spde_nested_ccd builds the CCD design at a clean interior mode", {
  # A textbook-concave log-marginal in (log range, log sigma) with an interior
  # maximum; combined with the PC prior, obj has a clean interior minimum, so
  # the precision optimHess(obj) is PD and the CCD design must engage rather
  # than fall back to the rectangular grid. Drives the integrator directly so
  # the branch is exercised without depending on a particular mesh/data
  # realisation landing interior.
  r0 <- 0.3; s0 <- 0.6
  spde_log_marginal <- function(r, s) {
    lm <- -0.5 * ((log(r) - log(r0))^2 / 0.25 + (log(s) - log(s0))^2 / 0.25)
    list(log_marginal = lm, n_iter = rep(5L, length(r)))
  }
  fit_spde_single <- function(range, sigma)
    list(mode = NULL, beta = NULL, spatial_effects = NULL, log_det_Q = NA_real_)
  sp <- list(prior_range = c(0.3, 0.5), prior_sigma = c(0.6, 0.05))

  res <- fit_spde_nested_ccd(spde_log_marginal, fit_spde_single, sp,
                             spatial = list(), diagnose_k = FALSE)

  expect_identical(res$nested$method, "ccd")
  expect_equal(res$nested$n_points, 9L)
  expect_equal(sum(res$nested$weights), 1, tolerance = 1e-8)
  expect_true(is.finite(res$range) && res$range > 0)
  expect_true(is.finite(res$sigma) && res$sigma > 0)
})
