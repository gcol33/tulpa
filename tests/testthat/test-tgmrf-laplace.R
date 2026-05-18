# ---------------------------------------------------------------------------
# tgmrf P2 — Laplace adapter via the multi-block nested-Laplace driver.
#
# The hard exit gate: a user-defined AR1 Q built through tgmrf() must match
# the built-in AR1 fitter bit-for-bit (numerical tolerance) at every grid
# point, because under matching parameterisations the two paths assemble
# the same Newton system. A divergence here means the precomputed-Q +
# generic block factory is producing different Hessians / log-marginals
# from the hand-written AR1 add_prior / log_prior_ar1.
#
# Also: a Poisson sim recovery sanity check (modes finite, posterior means
# in a sane band).
# ---------------------------------------------------------------------------

# Build a tgmrf with the same precision form as tulpa's built-in AR1.
#   Q_ii = tau (i = 0 or n-1) ; tau * (1 + rho^2) (interior)
#   Q_{i, i+1} = Q_{i+1, i} = -tau * rho
make_ar1_tgmrf <- function(n, theta_grid_matrix, bounds = NULL) {
  blk <- tgmrf(
    Q = function(theta) {
      tau <- exp(theta[1]); rho <- tanh(theta[2])
      d <- c(tau, rep(tau * (1 + rho^2), n - 2L), tau)
      o <- rep(-tau * rho, n - 1L)
      M <- Matrix::bandSparse(n, k = c(-1L, 0L, 1L),
                              diagonals = list(o, d, o))
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior  = function(theta) 0,    # flat prior on theta so log_marginal at
                                   # each grid point matches built-in AR1
                                   # (which uses an implicit flat prior).
    init   = c(log_tau = 0, atanh_rho = atanh(0.5)),
    bounds = bounds,
    name   = "ar1_user"
  )
  if (!is.null(theta_grid_matrix)) {
    blk$theta_grid_built <- theta_grid_matrix
  }
  blk
}

test_that("tgmrf AR1 matches built-in AR1 modes up to z-block recentering", {
  # Both paths assemble the same Newton system (same Q, same likelihood)
  # and converge to the same MAP. Built-in AR1 then applies a sum-to-zero
  # *post-hoc* center_effects on the z block (a relabel that leaves the
  # internal Newton state alone but shifts the reported z by -mean(z_MAP)
  # without adjusting beta — see laplace_newton.h L112). tgmrf carries no
  # `center` callback because user-defined Q is generally full-rank and a
  # mean shift IS a real change there. The equivalence test therefore
  # compares:
  #   * beta exactly (the centering does not touch beta)
  #   * z up to the mean-zero offset (subtract mean(z_tgmrf) and assert
  #     pointwise match with z_builtin).
  set.seed(2026)
  n <- 25L
  tau_true <- 1.5; rho_true <- 0.6
  z <- numeric(n)
  z[1] <- rnorm(1, 0, 1 / sqrt(tau_true * (1 - rho_true^2)))
  for (t in 2:n) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, 1 / sqrt(tau_true))
  X <- matrix(1, nrow = n, ncol = 1L)
  eta <- 0.4 + z
  y <- rpois(n, exp(eta))

  tau_grid <- c(0.8, 1.2, 2.0)
  rho_grid <- c(0.2, 0.5, 0.7)
  gr <- expand.grid(tau = tau_grid, rho = rho_grid)

  builtin <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = list(
      list(type = "ar1",
           temporal_idx = seq_len(n),
           n_times = n,
           tau_grid = gr$tau,
           rho_grid = gr$rho)
    ),
    family = "poisson",
    max_iter = 100L, tol = 1e-10
  )

  theta_grid_matrix <- cbind(log_tau = log(gr$tau),
                             atanh_rho = atanh(gr$rho))
  blk <- make_ar1_tgmrf(n, theta_grid_matrix)
  user <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = list(blk),
    family = "poisson",
    max_iter = 100L, tol = 1e-10
  )

  expect_equal(dim(user$modes), dim(builtin$modes))

  # beta: column 1 of each modes matrix.
  expect_equal(user$modes[, 1], builtin$modes[, 1], tolerance = 1e-8)

  # z columns: 2..n+1. Subtract row-wise mean from tgmrf and compare.
  z_cols <- 2:(n + 1L)
  user_z <- user$modes[, z_cols, drop = FALSE]
  user_z_centered <- user_z - rowMeans(user_z)
  expect_equal(user_z_centered, builtin$modes[, z_cols, drop = FALSE],
               tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("tgmrf AR1 recovers theta on a Poisson sim", {
  set.seed(7)
  n <- 80L
  tau_true <- 2.0; rho_true <- 0.7
  z <- numeric(n)
  z[1] <- rnorm(1, 0, 1 / sqrt(tau_true * (1 - rho_true^2)))
  for (t in 2:n) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, 1 / sqrt(tau_true))
  X <- matrix(1, nrow = n, ncol = 1L)
  eta <- 0.5 + z
  y <- rpois(n, exp(eta))

  blk <- make_ar1_tgmrf(
    n,
    theta_grid_matrix = NULL,                  # use bounds-based default
    bounds = list(lower = c(log(0.3),  atanh(0.0)),
                  upper = c(log(8.0),  atanh(0.95)))
  )
  fit <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = list(blk),
    family = "poisson",
    max_iter = 100L, tol = 1e-8
  )
  expect_true(all(is.finite(fit$log_marginal)))
  # Posterior weight should sit in a sensible band around the truth in the
  # block's parameterisation. Single-seed sanity, not a strict recovery
  # threshold (the multi-seed recovery test is the user-driven validation
  # follow-on per generic-todo.md P9).
  expect_true(abs(fit$theta_mean[1] - log(tau_true)) < 2.0)
  expect_true(abs(tanh(fit$theta_mean[2]) - rho_true) < 0.6)
})

test_that("tulpa_nested_laplace accepts a bare tgmrf object as prior", {
  set.seed(11)
  n <- 30L
  z <- as.numeric(arima.sim(list(ar = 0.5), n = n, sd = 1))
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(0.2 + z))
  blk <- make_ar1_tgmrf(
    n, theta_grid_matrix = NULL,
    bounds = list(lower = c(log(0.5), atanh(0)),
                  upper = c(log(3),   atanh(0.9)))
  )
  fit_bare <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = blk,                     # bare tgmrf, NOT list(blk)
    family = "poisson",
    max_iter = 60L, tol = 1e-7
  )
  fit_list <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = list(blk),
    family = "poisson",
    max_iter = 60L, tol = 1e-7
  )
  expect_equal(fit_bare$log_marginal, fit_list$log_marginal, tolerance = 1e-10)
  expect_equal(fit_bare$theta_mean, fit_list$theta_mean, tolerance = 1e-10)
})

test_that("tgmrf inside a multi-block prior composes with an iid block", {
  # Smoke check: tgmrf + iid on a Gaussian-latent Poisson sim. The joint
  # Cartesian grid is small (5 tgmrf cells * 3 sigma_iid points = 15) and
  # the goal here is purely to exercise the precompute -> joint-grid copy
  # path and confirm the inner Newton stays finite.
  set.seed(3)
  n <- 30L
  tau_true <- 1.0; rho_true <- 0.4
  z <- numeric(n)
  z[1] <- rnorm(1, 0, 1 / sqrt(tau_true * (1 - rho_true^2)))
  for (t in 2:n) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, 1 / sqrt(tau_true))
  obs_noise <- rnorm(n, 0, 0.2)
  eta <- 0.0 + z + obs_noise
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(eta))

  blk_tgmrf <- make_ar1_tgmrf(
    n, theta_grid_matrix = NULL,
    bounds = list(lower = c(log(0.5), atanh(0)),
                  upper = c(log(3),   atanh(0.9)))
  )

  iid_blk <- list(
    type    = "iid",
    obs_idx = seq_len(n),
    n_units = n,
    sigma_grid = c(0.1, 0.3, 0.6)
  )

  fit <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = list(blk_tgmrf, iid_blk),
    family = "poisson",
    max_iter = 80L, tol = 1e-7
  ))
  expect_true(all(is.finite(fit$log_marginal)))
  # 5x5 tgmrf grid * 3 sigma grid = 75 cells.
  expect_equal(nrow(fit$theta_grid), 75L)
})
