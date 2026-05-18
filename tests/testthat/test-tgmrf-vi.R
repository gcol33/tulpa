# ---------------------------------------------------------------------------
# tgmrf P5 -- Pathfinder VI over the block's hyperparameter vector.
#
# tulpa_tgmrf_vi() is the structured-tier sibling of the Tier-1 MCMC
# adapters (tulpa_tgmrf_imh, tulpa_tgmrf_nuts). Same Laplace body for
# (beta, z) | theta; the outer integration is an L-BFGS Gaussian fit on
# log_marginal(theta). No bias correction -- the Gaussian fit IS the
# approximation.
#
# These tests check (a) plumbing, (b) that VI lands in the same band as
# the Tier-1 samplers up to the structured approximation gap.
# ---------------------------------------------------------------------------

make_vi_ar1_block <- function(n) {
  tgmrf(
    Q = function(theta) {
      tau <- exp(theta[1]); rho <- tanh(theta[2])
      d <- c(tau, rep(tau * (1 + rho^2), n - 2L), tau)
      o <- rep(-tau * rho, n - 1L)
      M <- Matrix::bandSparse(n, k = c(-1L, 0L, 1L),
                              diagonals = list(o, d, o))
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior = function(theta) 0,
    init = c(log_tau = 0, atanh_rho = atanh(0.5)),
    bounds = list(lower = c(log(0.3), atanh(0)),
                  upper = c(log(3),   atanh(0.95))),
    name = "ar1_vi"
  )
}

test_that("tulpa_tgmrf_vi returns Gaussian draws with sensible moments", {
  set.seed(101)
  n <- 25L
  z <- as.numeric(arima.sim(list(ar = 0.5), n = n, sd = 1))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.2 + z))
  blk <- make_vi_ar1_block(n)

  fit <- tulpa_tgmrf_vi(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson",
    n_draws = 500L
  )

  expect_s3_class(fit, "tulpa_tgmrf_vi")
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(ncol(fit$draws), 2L)
  expect_equal(colnames(fit$draws), c("log_tau", "atanh_rho"))
  expect_true(all(is.finite(fit$draws)))
  expect_equal(fit$inference_mode, "structured")
  expect_equal(fit$inference_tier, 2L)
  expect_equal(fit$backend, "tgmrf_vi")
  expect_true(is.finite(fit$elbo))
  expect_true(fit$converged)
  expect_named(fit$mode_theta, c("log_tau", "atanh_rho"))
  expect_equal(dim(fit$cov), c(2L, 2L))
  # Variational SDs come from the inverse Hessian -- positive, finite,
  # and not absurdly large.
  expect_true(all(fit$sds > 0))
  expect_true(all(fit$sds < 5))
  expect_lt(abs(fit$means[1]), 3)
  expect_lt(abs(fit$means[2]), 3)
})

test_that("VI mode and IMH posterior mean agree up to the structured-tier gap", {
  skip_on_cran()
  set.seed(2031)
  n <- 30L
  z <- as.numeric(arima.sim(list(ar = 0.55), n = n, sd = 1))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.3 + z))
  blk <- make_vi_ar1_block(n)

  fit_imh <- tulpa_tgmrf_imh(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson",
    n_iter = 1500L, warmup = 500L
  )
  fit_vi <- tulpa_tgmrf_vi(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson",
    n_draws = 800L
  )

  # VI mode should land near IMH posterior mean for a near-Gaussian
  # target. Tolerance covers (a) IMH MC error and (b) VI underdispersion
  # bias.
  expect_lt(abs(fit_imh$means[1] - fit_vi$mode_theta[1]), 0.8)
  expect_lt(abs(fit_imh$means[2] - fit_vi$mode_theta[2]), 1.8)
})

test_that("tulpa_tgmrf_vi rejects non-tgmrf block argument", {
  expect_error(
    tulpa_tgmrf_vi(
      y = 1:5, n_trials = rep(1L, 5L),
      X = matrix(1, 5L, 1L),
      block = list(),
      family = "poisson"
    ),
    "tgmrf"
  )
})
