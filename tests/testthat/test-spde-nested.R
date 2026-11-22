# test-spde-nested.R
# fit_spde()'s nested-Laplace integration over (range, sigma): the field is an
# `spde` block of the joint door, which places the outer grid at the outer mode
# and scores it by its outer Pareto-k-hat.

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

# The Poisson plumbing fixture the CCD blocks below share: one data set, fit
# once per control setting. `diagnose_k = FALSE` is the fit the blocks that do
# not read k-hat use; the diagnostic runs after the integration from its own
# preserved seed, so it does not move the moments those blocks compare.
.ccd_poisson_data <- fixture_once(function() {
  d <- helper_make_spde_for_ccd(n_obs = 250, range_true = 0.3,
                                sigma_true = 0.6, seed = 11L,
                                prior_sigma = c(0.6, 0.05))
  y <- rpois(length(d$eta), lambda = exp(2.0 + d$eta))
  list(y = y, X = matrix(1, nrow = length(y), ncol = 1), spec = d$spec)
})

.ccd_poisson_fit <- fixture_once(function(...) {
  d <- .ccd_poisson_data()
  suppressWarnings(
    fit_spde(d$y, d$X, d$spec, family = "poisson",
             control = list(progress = FALSE, ...))
  )
})

test_that("fit_spde() integrates (range, sigma) on the joint door", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  fit <- .ccd_poisson_fit(k_samples = 150L)
  rec <- fit$nested
  expect_identical(rec$backend, "nested_laplace_joint")
  expect_identical(tulpa:::.hyper_axis_bare(colnames(rec$theta_grid)),
                   c("range", "sigma"))
  expect_true(all(is.finite(rec$weights)))
  expect_equal(sum(rec$weights), 1, tolerance = 1e-8)
  expect_true(rec$range_mean > 0 && is.finite(rec$range_mean))
  expect_true(rec$sigma_mean > 0 && is.finite(rec$sigma_mean))
  # The fit's mode is the inner mode of the cell of highest posterior density.
  expect_identical(fit$range, rec$range_best)
  expect_identical(fit$sigma, rec$sigma_best)
  expect_equal(fit$mode, unname(rec$modes[rec$best_idx, ]))
})

test_that("fit_spde() returns a fit on data with no spatial structure", {
  # Noise data leaves the likelihood nothing to identify a mode from, and the
  # priors are nearly flat on both axes.
  skip_if_not_installed("fmesher")
  skip_on_cran()

  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords,
                              max.edge = c(0.15, 0.4), cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                              prior_range = c(1e-6, 1e-6),
                              prior_sigma = c(1e6, 0.9999))
  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)
  fit <- suppressWarnings(fit_spde(y, X, spec, family = "binomial",
                                   n_trials = rep(1L, n_obs),
                                   control = list(progress = FALSE)))
  expect_true(all(is.finite(fit$nested$weights)))
  expect_true(is.finite(fit$beta))
})

test_that("fit_spde reports an outer Pareto-k-hat over (range, sigma)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  fit <- .ccd_poisson_fit(k_samples = 150L)
  expect_true(is.finite(fit$pareto_k))
  expect_true(is.finite(fit$pareto_k_is_ess))
  expect_gt(fit$pareto_k_is_ess, 0)
  expect_identical(fit$pareto_k, fit$nested$pareto_k)

  off <- .ccd_poisson_fit(diagnose_k = FALSE)
  expect_true(is.na(off$pareto_k))
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
# The deterministic Tier-2 path (fit_spde) recovers (range, sigma) and     #
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

skip_on_cran()

test_that("fit_spde recovers (range, sigma) on a genuine field (#98)", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  skip_if_not_installed("Matrix")

  # Multi-seed recovery + 95% interval coverage on a GENUINE Matern field.
  # Gaussian obs (Laplace exact) with broad priors, so recovery is driven by the
  # SPDE marginal, not the PC prior. The intervals are spatial_range()'s, read
  # off the outer grid.
  range_true <- 0.35; sigma_true <- 0.8; sigma_obs <- 0.3
  n_obs <- 300L; n_seed <- 12L

  rr <- ss <- numeric(n_seed)
  cov_r <- cov_s <- logical(n_seed)
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
                                     phi = sigma_obs^2,
                                     control = list(diagnose_k = FALSE,
                                                    progress = FALSE)))
    rr[s] <- fit$nested$range_mean
    ss[s] <- fit$nested$sigma_mean
    sr <- spatial_range(fit)
    cov_r[s] <- sr["range", "q2.5"] <= range_true && range_true <= sr["range", "q97.5"]
    cov_s[s] <- sr["sigma", "q2.5"] <= sigma_true && sigma_true <= sr["sigma", "q97.5"]
  }

  expect_true(all(ss < 3 * sigma_true),
              info = sprintf("sigma_hats: %s", paste(round(ss, 3), collapse = ", ")))
  expect_true(all(abs(log(rr / range_true)) < 0.8),
              info = sprintf("range log-errs: %s",
                             paste(round(log(rr / range_true), 3), collapse = ", ")))
  expect_true(all(abs(log(ss / sigma_true)) < 0.6),
              info = sprintf("sigma log-errs: %s",
                             paste(round(log(ss / sigma_true), 3), collapse = ", ")))
  expect_lt(abs(mean(log(rr / range_true))), 0.30)
  expect_lt(abs(mean(log(ss / sigma_true))), 0.20)
  expect_gte(sum(cov_r), ceiling(0.70 * n_seed))
  expect_gte(sum(cov_s), ceiling(0.70 * n_seed))
})
