# Fixed-hyperparameter (single-point) Laplace for the NNGP and SPDE fields
# (gcol33/tulpa#277). Both exports are one-cell runs of the SAME single arm
# (make_single_arm) + LatentBlock factory (make_nngp_block / make_spde_block) +
# joint multi-block driver their nested integrators grid over, so the mode and
# the log-marginal must equal the nested kernel evaluated at that single cell.
#
# The equivalence here is EXACT, not approximate: the single fit and the nested
# kernel run the same driver over the same block, so the assertions below use
# `tolerance = 0` deliberately. A tolerance would hide exactly the drift a
# second Newton implementation reintroduces, which is what this file exists to
# prevent -- the precedent is test-laplace-spatial-car-proper-hsgp.R.

# ---------------------------------------------------------------- NNGP -------

# A Vecchia ordering with its nearest-predecessor neighbour lists: site i
# conditions on its `nn` closest predecessors in the natural order, which is the
# layout both entries expect (nn_order 0-based, nn_idx 1-based into that order).
.gp_fixture <- function(ng, seed, nn = 5L, p = 2L) {
  set.seed(seed)
  coords <- cbind(runif(ng), runif(ng))
  fld <- as.numeric(scale(sin(3 * coords[, 1]) + coords[, 2]))
  y <- rbinom(ng, 1, plogis(fld))
  X <- if (p == 1L) matrix(1, ng, 1) else cbind(1, rnorm(ng))
  d <- as.matrix(dist(coords))
  nn_idx <- matrix(0L, ng, nn)
  nn_dist <- matrix(0, ng, nn)
  for (i in seq_len(ng)) {
    prev <- seq_len(i - 1L)
    if (!length(prev)) next
    k <- min(nn, length(prev))
    o <- order(d[i, prev])[seq_len(k)]
    nn_idx[i, seq_len(k)] <- as.integer(prev[o])
    nn_dist[i, seq_len(k)] <- d[i, prev[o]]
  }
  list(y = as.numeric(y), n = rep(1L, ng), X = X, re_idx = rep(0, ng),
       n_re_groups = 0L, sigma_re = 1.0, coords = coords,
       nn_idx = nn_idx, nn_dist = nn_dist,
       nn_order = as.integer(seq_len(ng) - 1L),
       n_spatial = ng, nn = nn, cov_type = 0L, family = "binomial")
}

.expect_gp_equiv <- function(args, sigma2 = 0.9, phi_gp = 0.4, extra = list()) {
  sp <- do.call(tulpa:::cpp_laplace_fit_gp,
                c(args, list(sigma2_gp = sigma2, phi_gp = phi_gp), extra))
  ns_args <- args
  ns_args$spatial_idx <- extra$obs_to_loc_nullable %||%
    seq_len(length(args$y))
  extra$obs_to_loc_nullable <- NULL
  ns <- do.call(tulpa:::cpp_nested_laplace_nngp,
                c(ns_args, list(sigma2_grid = sigma2, phi_gp_grid = phi_gp),
                  extra))
  expect_equal(as.numeric(sp$mode), as.numeric(ns$modes[1, ]), tolerance = 0)
  expect_equal(sp$log_marginal, as.numeric(ns$log_marginal)[1], tolerance = 0)
  expect_equal(sp$log_det_Q, as.numeric(ns$log_det_Q)[1], tolerance = 0)
  expect_equal(sp$n_iter, as.integer(ns$n_iter)[1])
  expect_equal(sp$converged, as.logical(ns$converged)[1])
  sp
}

test_that("GP single-point == nested NNGP at one (sigma2, phi_gp) cell", {
  skip_on_cran()
  # Both sides of the dense/sparse Newton threshold (SPARSE_THRESHOLD = 200).
  # The NNGP block scatters its prior into the sparse builder only, so
  # blocks_require_sparse() pins BOTH entries to the sparse path at every n_x --
  # which is what makes the small-field case exact rather than merely close.
  for (ng in c(60L, 150L, 260L)) {
    fit <- .expect_gp_equiv(.gp_fixture(ng, 11L + ng))
    expect_true(fit$converged)
  }
})

test_that("GP single-point == nested NNGP across neighbour-set sizes", {
  skip_on_cran()
  # The dense scatter's disagreement with the sparse one grew with nn and
  # diverged outright at nn = 8 (a 300-iteration non-convergence returning
  # log_marginal = NaN, against a 23-iteration convergence on the sparse path).
  # Sweeping nn keeps that regime covered.
  for (nn in c(2L, 5L, 8L)) {
    fit <- .expect_gp_equiv(.gp_fixture(150L, 99L, nn = nn))
    expect_true(fit$converged)
    expect_true(is.finite(fit$log_marginal))
  }
})

test_that("GP single-point == nested NNGP with an RE block and an offset", {
  skip_on_cran()
  a <- .gp_fixture(120L, 7L)
  ng <- length(a$y)
  a$re_idx <- as.numeric(rep(1:6, length.out = ng))
  a$n_re_groups <- 6L
  a$sigma_re <- 0.8
  set.seed(21)
  off <- rnorm(ng, 0, 0.2)
  .expect_gp_equiv(a, extra = list(offset_nullable = off))
})

test_that("GP single-point == nested NNGP when coordinates repeat", {
  skip_on_cran()
  # obs_to_loc attaches several observations to one field node; the nested
  # entry calls the same map spatial_idx. Identity mapping is the p == n case.
  a <- .gp_fixture(80L, 33L)
  n_loc <- length(a$y)
  set.seed(34)
  extra_rows <- sample.int(n_loc, 40L, replace = TRUE)
  keep <- c(seq_len(n_loc), extra_rows)
  a$y <- a$y[keep]
  a$n <- a$n[keep]
  a$X <- a$X[keep, , drop = FALSE]
  a$re_idx <- rep(0, length(keep))
  .expect_gp_equiv(a, extra = list(obs_to_loc_nullable = as.integer(keep)))
})


# ---------------------------------------------------------------- SPDE -------

.spde_fixture <- function(n_obs = 200L, seed = 42L, p = 1L) {
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.5),
                              cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  w_true <- rnorm(mesh$n, 0, 0.3)
  w_true <- w_true - mean(w_true)
  y <- rbinom(n_obs, 1, plogis(-0.5 + as.numeric(A %*% w_true)))
  X <- if (p == 1L) matrix(1, n_obs, 1) else cbind(1, rnorm(n_obs))
  list(y = as.numeric(y), n_trials = rep(1L, n_obs), X = X,
       re_idx = rep(0, n_obs), n_re_groups = 0L, sigma_re = 1.0,
       A_x = A@x, A_i = A@i, A_p = A@p,
       n_obs = n_obs, n_mesh = mesh$n,
       C0_diag = Matrix::diag(fem$c0),
       G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
       family = "binomial")
}

# The single fit is handed (kappa, tau); the nested entry is handed
# (range, sigma) and converts. Computing the pair with the SAME expressions the
# block uses makes the two grids the same doubles, so the comparison is of the
# solvers rather than of two roundings.
.spde_kt <- function(range, sigma, nu) {
  kappa <- sqrt(8 * nu) / range
  list(kappa = kappa, tau = 1 / (sqrt(4 * pi) * kappa * sigma))
}

test_that("SPDE single-point == nested SPDE at one (range, sigma) cell", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  a <- .spde_fixture()
  nu <- 1
  for (rg in list(c(0.3, 0.5), c(0.6, 1.1))) {
    kt <- .spde_kt(rg[1], rg[2], nu)
    sp <- do.call(tulpa:::cpp_laplace_fit_spde,
                  c(a, list(kappa = kt$kappa, tau_spde = kt$tau, alpha = 2L)))
    ns <- do.call(tulpa:::cpp_nested_laplace_spde,
                  c(a, list(range_grid = rg[1], sigma_grid = rg[2], nu = nu)))
    expect_equal(as.numeric(sp$mode), as.numeric(ns$modes[1, ]), tolerance = 0)
    expect_equal(sp$log_marginal, as.numeric(ns$log_marginal)[1], tolerance = 0)
    expect_equal(sp$log_det_Q, as.numeric(ns$log_det_Q)[1], tolerance = 0)
    expect_true(sp$converged)
  }
})

test_that("SPDE single-point == nested SPDE with covariates and an RE block", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  a <- .spde_fixture(n_obs = 180L, seed = 8L, p = 2L)
  a$re_idx <- as.numeric(rep(1:5, length.out = a$n_obs))
  a$n_re_groups <- 5L
  a$sigma_re <- 0.7
  kt <- .spde_kt(0.4, 0.8, 1)
  sp <- do.call(tulpa:::cpp_laplace_fit_spde,
                c(a, list(kappa = kt$kappa, tau_spde = kt$tau, alpha = 2L)))
  ns <- do.call(tulpa:::cpp_nested_laplace_spde,
                c(a, list(range_grid = 0.4, sigma_grid = 0.8, nu = 1)))
  expect_equal(as.numeric(sp$mode), as.numeric(ns$modes[1, ]), tolerance = 0)
  expect_equal(sp$log_marginal, as.numeric(ns$log_marginal)[1], tolerance = 0)
})

test_that("the SPDE single fit returns a mode its own score confirms", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  # The mesh field is sum-to-zero centred after the Newton loop. Centring it
  # without moving the constant into the intercept shifts eta away from the mode
  # the loop found: the fit then reported a beta score of ~0.47 at a converged
  # solve. The driver folds the removed constant into the arm intercept, so eta
  # -- and with it the reported mode -- is the one the loop settled on.
  a <- .spde_fixture(n_obs = 160L, seed = 15L, p = 2L)
  kt <- .spde_kt(0.35, 0.6, 1)
  sp <- do.call(tulpa:::cpp_laplace_fit_spde,
                c(a, list(kappa = kt$kappa, tau_spde = kt$tau, alpha = 2L,
                          max_iter = 300L, tol = 1e-10)))
  expect_true(sp$converged)

  p <- ncol(a$X)
  mesh <- sp$mode[(p + 1):(p + a$n_mesh)]
  expect_lt(abs(mean(mesh)), 1e-8)          # the field really is centred

  # d(log lik)/d beta at the reported mode, less the kernel's weak ridge
  # (DEFAULT_TAU_BETA = 1e-4). The mesh score carries the sum-to-zero
  # multiplier, so only the beta block is a clean stationarity statement.
  A <- Matrix::sparseMatrix(i = a$A_i, p = a$A_p, x = a$A_x,
                            dims = c(a$n_obs, a$n_mesh), index1 = FALSE)
  eta <- as.numeric(a$X %*% sp$mode[seq_len(p)] + A %*% mesh)
  # The Newton loop stops on its step, so this residual settles at a floor set
  # by conditioning rather than at zero: it is ~2e-06 here. The bound is what
  # separates a settled mode from the uncompensated centring, which left this
  # same quantity at ~0.47 -- five orders of magnitude up.
  g_beta <- as.numeric(crossprod(a$X, a$y - plogis(eta))) -
    1e-4 * sp$mode[seq_len(p)]
  expect_lt(max(abs(g_beta)), 1e-4)
})
