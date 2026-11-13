# Proper hyperpriors define the posterior on the unbounded theta support.

skip_on_cran()

make_ar1_tgmrf_imh <- function(n) {
  tgmrf(
    Q = function(theta) {
      tau <- exp(theta[1]); rho <- tanh(theta[2])
      d <- c(tau, rep(tau * (1 + rho^2), n - 2L), tau)
      o <- rep(-tau * rho, n - 1L)
      M <- Matrix::bandSparse(n, k = c(-1L, 0L, 1L),
                              diagonals = list(o, d, o))
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior = function(theta) sum(stats::dnorm(theta, 0, 1, log = TRUE)),
    init = c(log_tau = 0, atanh_rho = atanh(0.5)),
    bounds = list(lower = c(log(0.3), atanh(0)),
                  upper = c(log(3),   atanh(0.95))),
    name = "ar1_imh"
  )
}

test_that("tulpa_tgmrf_imh produces finite draws + healthy acceptance", {
  skip_if_not_slow()
  set.seed(101)
  n <- 30L
  z <- as.numeric(arima.sim(list(ar = 0.5), n = n, sd = 1))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.2 + z))
  blk <- make_ar1_tgmrf_imh(n)

  fit <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson", mode = "imh",
    n_iter = 400L, warmup = 200L,
    pilot_axis_points = 5L
  )
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(ncol(fit$draws_theta), 2L)
  expect_equal(colnames(fit$draws_theta), c("log_tau", "atanh_rho"))
  expect_true(all(is.finite(fit$draws)))
  expect_true(all(is.finite(fit$log_prob)))
  # Acceptance should not collapse — the Laplace proposal is well-matched
  # to the AR1 hyperparameter posterior. Anything below 5% would signal
  # the FD Hessian or the mode is broken.
  expect_gt(fit$mean_accept, 0.05)
  expect_equal(fit$inference_mode, "exact")
  expect_equal(fit$inference_tier, 1L)
})

test_that("pseudo-marginal theta means stay near a Laplace fit on a moderate-data fixture", {
  skip_if_not_slow()
  # The grid comparison includes quadrature and inner Laplace error.
  set.seed(202)
  n <- 50L
  z <- as.numeric(arima.sim(list(ar = 0.6), n = n, sd = 1.0))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.3 + z))
  blk <- make_ar1_tgmrf_imh(n)

  grid <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = blk, family = "poisson",
    control = list(max_iter = 80L, tol = 1e-8)
  )
  fit <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson", mode = "imh",
    n_iter = 1500L, warmup = 500L,
    pilot_axis_points = 5L
  )
  # This comparison checks a broad posterior region; the grid retains
  # both quadrature and inner Laplace error.
  grid_mean <- as.numeric(grid$theta_mean)
  imh_mean  <- as.numeric(fit$means_theta)
  expect_lt(abs(grid_mean[1] - imh_mean[1]), 0.5)
  expect_lt(abs(grid_mean[2] - imh_mean[2]), 0.5)
})
