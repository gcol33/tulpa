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
  # i.i.d. mesh noise carries no real spatial range, so these plumbing fits lean
  # on a tighter-than-default PC prior to pin a well-conditioned mode on data the
  # likelihood barely constrains. (Genuine-field recovery -- where the SPDE
  # marginal's 0.5 log|Q| normalizer makes the (range, sigma) mode interior on
  # its own -- is exercised separately below with broad priors.)
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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()
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

# --------------------------------------------------------------------------- #
# (range, sigma) recovery against truth, cross-checked vs the NUTS-joint       #
# integrator (gcol33/tulpa#98).                                                #
#                                                                              #
# `helper_make_spde_for_ccd` draws the mesh field as i.i.d. mesh noise, so it  #
# carries NO spatial range -- the plumbing tests above use it, never recovery. #
# This block simulates a GENUINE Matern SPDE field at a known (range, sigma)   #
# (the same precision the kernels assume) and checks the marginalized          #
# hyperparameter posterior against that truth.                                 #
#                                                                              #
# The deterministic Tier-2 path (fit_spde CCD) recovers (range, sigma) and     #
# agrees with the Tier-1 NUTS-joint integrator on the shared simulator. The    #
# SPDE Laplace marginal carries the GMRF prior normalizer 0.5 log|Q(theta)|    #
# (the Occam factor that bends the marginal down at large sigma), so the       #
# (range, sigma) posterior is interior-peaked rather than railing -- the same  #
# theta-dependent normalizer NUTS carries through log p(x|theta). Before the   #
# normalizer was added (see src/spde_logdet.h) sigma railed to the tens on     #
# this exact field while NUTS recovered ~0.31 / ~0.76; the two integrators now #
# match. Gaussian obs (exact Laplace) make this a clean cross-integrator gate. #
# --------------------------------------------------------------------------- #

# Exact Matern SPDE field on the spec's mesh at (range_true, sigma_true): draws
# w ~ N(0, Q^{-1}) with Q = tau^2 (kappa^2 C0 + G1) C0^{-1} (kappa^2 C0 + G1),
# kappa = sqrt(8 nu)/range, tau = 1/(sqrt(4 pi) kappa sigma) (nu = 1).
.spde_recover_field <- function(spec, range_true, sigma_true, seed) {
  set.seed(seed)
  nu    <- spec$nu
  kappa <- sqrt(8 * nu) / range_true
  tau   <- 1 / (sqrt(4 * pi) * kappa * sigma_true)
  C0 <- Matrix::Diagonal(x = spec$C0_diag)
  G1 <- Matrix::sparseMatrix(i = spec$G1_i + 1L, p = spec$G1_p, x = spec$G1_x,
                             dims = c(spec$n_mesh, spec$n_mesh), index1 = TRUE)
  K <- (kappa^2) * C0 + G1
  Q <- (tau^2) * Matrix::crossprod(K, Matrix::solve(C0, K))
  L <- Matrix::Cholesky(Matrix::forceSymmetric(Q), LDL = FALSE, perm = FALSE)
  as.numeric(Matrix::solve(L, rnorm(spec$n_mesh), system = "Lt"))
}

# 95% CI on (range, sigma) from the CCD log-scale mode + its Hessian Gaussian
# approximation (hessian_logsc is the neg-log-posterior Hessian = precision of
# N(theta_hat, .) in (log range, log sigma)).
.spde_ccd_ci <- function(nested) {
  V  <- solve(nested$hessian_logsc)
  sd <- sqrt(pmax(diag(V), 0))
  list(range = exp(log(nested$range_mode) + c(-1.96, 1.96) * sd[1]),
       sigma = exp(log(nested$sigma_mode) + c(-1.96, 1.96) * sd[2]))
}

test_that("SPDE CCD recovers (range, sigma) on a genuine field (#98)", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  skip_if_not_installed("Matrix")

  # Multi-seed recovery + CI coverage on a GENUINE Matern field (unlike
  # helper_make_spde_for_ccd's i.i.d. mesh noise). Gaussian obs (Laplace exact)
  # with broad priors, so recovery is driven by the SPDE marginal, not the PC
  # prior. Cross-checks the Tier-2 CCD against the Tier-1 NUTS-joint truth
  # (range ~ 0.31 / sigma ~ 0.76 for a true 0.35 / 0.80 field).
  range_true <- 0.35; sigma_true <- 0.8; sigma_obs <- 0.3
  n_obs <- 300L; n_seed <- 12L

  rr <- ss <- numeric(n_seed)
  cov_r <- cov_s <- logical(n_seed); used_ccd <- logical(n_seed)
  for (s in seq_len(n_seed)) {
    set.seed(s * 31L + 7L)
    coords <- cbind(runif(n_obs), runif(n_obs))
    mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.08)
    fem  <- fmesher::fm_fem(mesh)
    A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
    spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                                prior_range = c(0.1, 0.05),
                                prior_sigma = c(3.0, 0.05))
    w  <- .spde_recover_field(spec, range_true, sigma_true, seed = s * 31L + 8L)
    w  <- w - mean(w)
    xc <- runif(n_obs, -1, 1); X <- cbind(1, xc)
    y  <- 1.0 + 0.5 * xc + as.numeric(spec$A %*% w) + rnorm(n_obs, 0, sigma_obs)

    fit <- suppressWarnings(fit_spde(y, X, spec, family = "gaussian",
                                     phi = sigma_obs, method = "ccd",
                                     diagnose_k = FALSE))
    rr[s] <- fit$nested$range_mean
    ss[s] <- fit$nested$sigma_mean
    used_ccd[s] <- identical(fit$nested$method, "ccd")
    if (used_ccd[s]) {
      ci <- .spde_ccd_ci(fit$nested)
      cov_r[s] <- ci$range[1] <= range_true && range_true <= ci$range[2]
      cov_s[s] <- ci$sigma[1] <= sigma_true && sigma_true <= ci$sigma[2]
    }
  }

  # No seed rails: sigma stays O(1), not the tens it hit without the prior
  # normalizer. range stays inside half a decade of the truth.
  expect_true(all(ss < 3 * sigma_true),
              info = sprintf("sigma_hats: %s", paste(round(ss, 3), collapse = ", ")))
  expect_true(all(abs(log(rr / range_true)) < 0.8),
              info = sprintf("range log-errs: %s",
                             paste(round(log(rr / range_true), 3), collapse = ", ")))
  expect_true(all(abs(log(ss / sigma_true)) < 0.6),
              info = sprintf("sigma log-errs: %s",
                             paste(round(log(ss / sigma_true), 3), collapse = ", ")))

  # Across-seed bias: the CCD weighted mean tracks the truth (calibration ~ -0.14
  # / +0.02 in log on this design; gate at 0.3 / 0.2 with headroom).
  expect_lt(abs(mean(log(rr / range_true))), 0.30)
  expect_lt(abs(mean(log(ss / sigma_true))), 0.20)

  # 95% CI coverage (Gaussian-mode approx) over the seeds that engaged the CCD
  # design. Calibration ~ 11/12; gate at a 0.70 floor.
  n_ccd <- sum(used_ccd)
  expect_gt(n_ccd, 0)
  expect_gte(sum(cov_r[used_ccd]), ceiling(0.70 * n_ccd))
  expect_gte(sum(cov_s[used_ccd]), ceiling(0.70 * n_ccd))
})
