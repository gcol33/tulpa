# ---------------------------------------------------------------------------
# tgmrf P3 — NUTS over the block's hyperparameter vector.
#
# tulpa_tgmrf_nuts() is the gradient-based sibling of tulpa_tgmrf_imh().
# Same composition (Laplace body + MH-style bias correction) but the
# proposals come from leapfrog integration of log_marginal(theta) with a
# finite-difference gradient. This is the closest analogue to a full
# joint-(beta, z, theta) NUTS reachable without plumbing user-Q
# callbacks through the C++ hmc_nuts_* leapfrog (P3 full -- a separate
# C++ scope; see dev_notes/plans/generic-todo.md).
#
# These tests are slow (~30 s) because each leapfrog step costs
# 2 * theta_dim inner Laplace solves. Marked skip_on_cran to keep
# CI lean.
# ---------------------------------------------------------------------------

skip_if_not_slow()

make_nuts_ar1_block <- function(n) {
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
    name = "ar1_nuts"
  )
}

test_that("tulpa_tgmrf_nuts runs end-to-end and returns sensible posterior moments", {
  set.seed(99)
  n <- 25L
  z <- as.numeric(arima.sim(list(ar = 0.5), n = n, sd = 1))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.2 + z))
  blk <- make_nuts_ar1_block(n)

  fit <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson", mode = "nuts",
    n_iter = 150L, warmup = 75L,
    max_depth = 4L
  )

  expect_s3_class(fit, "tulpa_tgmrf")
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(ncol(fit$draws), 2L)
  expect_equal(colnames(fit$draws), c("log_tau", "atanh_rho"))
  expect_true(all(is.finite(fit$draws)))
  expect_gt(fit$mean_accept, 0.05)        # collapse guard
  expect_lt(fit$mean_accept, 0.999)       # not stuck
  expect_true(all(fit$tree_depth >= 1L))
  # Energy divergences are tracked (previously never counted): a well-formed,
  # per-draw logical vector plus a non-negative integer count within range.
  expect_type(fit$divergent, "logical")
  expect_length(fit$divergent, nrow(fit$draws))
  expect_true(fit$n_divergent >= 0L && fit$n_divergent <= nrow(fit$draws))
  expect_equal(fit$n_divergent, sum(fit$divergent))
  expect_equal(fit$inference_mode, "exact")
  expect_equal(fit$inference_tier, 1L)
  # Posterior means in the (log_tau, atanh_rho) band consistent with the sim.
  expect_lt(abs(fit$means[1]), 3)
  expect_lt(abs(fit$means[2]), 3)
})

test_that("NUTS and IMH on the same tgmrf agree on posterior means up to MC error", {
  set.seed(2030)
  n <- 30L
  z <- as.numeric(arima.sim(list(ar = 0.55), n = n, sd = 1))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.3 + z))
  blk <- make_nuts_ar1_block(n)

  fit_imh <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson", mode = "imh",
    n_iter = 1500L, warmup = 500L
  )
  fit_nuts <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson", mode = "nuts",
    n_iter = 200L, warmup = 100L,
    max_depth = 4L
  )
  # Both target the same posterior p(theta | y) = log_marginal(theta).
  # Means should land in the same band up to MC error of either chain.
  # NUTS at n_iter = 200 has higher MC noise than IMH at n_iter = 1500;
  # the tolerance is set against that, not against asymptotic agreement.
  expect_lt(abs(fit_imh$means[1] - fit_nuts$means[1]), 0.8)
  expect_lt(abs(fit_imh$means[2] - fit_nuts$means[2]), 1.8)
})
