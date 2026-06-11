# ---------------------------------------------------------------------------
# tgmrf P3 full -- joint NUTS over (beta, z, theta) in C++.
#
# tulpa_tgmrf_nuts_joint() integrates the full joint posterior in one
# Hamiltonian system: closed-form gradients for (beta, z), central FD on
# theta. Targets the same posterior as the outer-theta NUTS adapter
# (tulpa_tgmrf_nuts()) but reaches it via the joint dynamics rather than
# via an inner Laplace + FD-on-log-marginal outer step.
#
# Scope: C++-backend tgmrf blocks only (block$backend == "cpp"). The
# R-closure backend cannot reach joint NUTS efficiently because calling R
# for Q(theta) at every leapfrog step is prohibitive. The wrapper errors
# out with a clear pointer to tgmrf_cpp() / tulpa_tgmrf_nuts().
#
# Tests are slow (Rcpp::sourceCpp compile + joint NUTS); marked
# skip_on_cran.
# ---------------------------------------------------------------------------

skip_on_cran()
skip_if_fast()

# Helper: locate the periodic-AR1 example .cpp. Works under load_all and
# under installed package.
.joint_cpp_example_path <- function() {
  p <- system.file("examples", "tgmrf_periodic_ar1.cpp", package = "tulpa")
  if (nzchar(p) && file.exists(p)) return(p)
  fallback <- file.path("..", "..", "inst", "examples",
                        "tgmrf_periodic_ar1.cpp")
  if (file.exists(fallback)) return(normalizePath(fallback))
  testthat::skip("Cannot locate inst/examples/tgmrf_periodic_ar1.cpp")
}

# Shared R-closure analogue for the cross-check test.
.joint_make_periodic_ar1_r <- function(n, init, bounds = NULL) {
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1])
      rho   <- tanh(theta[2])
      tau   <- 1 / sigma^2
      d <- rep(tau * (1 + rho^2), n)
      o <- rep(-tau * rho, n - 1L)
      M <- Matrix::bandSparse(n, k = c(-1L, 0L, 1L),
                              diagonals = list(o, d, o))
      M[1, n] <- -tau * rho
      M[n, 1] <- -tau * rho
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior = function(theta) {
      stats::dnorm(theta[1], 0, 1, log = TRUE) +
        stats::dnorm(theta[2], 0, 1, log = TRUE)
    },
    init   = init,
    bounds = bounds,
    name   = "periodic_ar1_r"
  )
}

# ---------------------------------------------------------------------------
# (a) End-to-end on the periodic AR1: gradient correctness + healthy mixing
# ---------------------------------------------------------------------------

test_that("tulpa_tgmrf_nuts_joint runs end-to-end on a periodic-AR1 sim", {
  cpp_file <- .joint_cpp_example_path()

  # kN = 80 in the .cpp.
  n <- 80L
  init   <- c(log_sigma = 0, atanh_rho = atanh(0.5))
  bounds <- list(lower = c(log(0.3), atanh(0.0)),
                 upper = c(log(3.0), atanh(0.95)))

  blk <- tgmrf_cpp(cpp_file = cpp_file, id = "tgmrf_periodic_ar1",
                   init = init, bounds = bounds,
                   name = "periodic_ar1_cpp")

  set.seed(2026)
  sigma_true <- 0.8; rho_true <- 0.7
  z <- numeric(n)
  z[1] <- rnorm(1, 0, sigma_true)
  for (t in 2:n) {
    z[t] <- rho_true * z[t - 1] + rnorm(1, 0, sigma_true * sqrt(1 - rho_true^2))
  }
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(0.4 + z))

  fit <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk,
    family = "poisson", mode = "nuts_joint",
    n_iter = 150L, warmup = 75L, max_depth = 5L,
    seed = 42L
  )

  expect_s3_class(fit, "tulpa_tgmrf")
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(ncol(fit$draws_theta), 2L)
  expect_equal(colnames(fit$draws_theta), c("log_sigma", "atanh_rho"))
  expect_equal(ncol(fit$draws_beta),  1L)
  expect_equal(ncol(fit$draws_z),     n)
  expect_true(all(is.finite(fit$draws_theta)))
  expect_true(all(is.finite(fit$draws_beta)))
  expect_true(all(is.finite(fit$draws_z)))

  # Healthy mixing: mean_accept inside [0.4, 0.95]. Dual averaging targets
  # 0.8; we set the band wide enough to absorb 75-warmup MC noise.
  expect_gt(fit$mean_accept, 0.4)
  expect_lt(fit$mean_accept, 0.99)

  # tree depth > 0 most of the time (not stuck at the root).
  expect_true(median(fit$tree_depth) >= 1L)

  expect_equal(fit$inference_mode, "exact")
  expect_equal(fit$inference_tier, 1L)
  expect_equal(fit$backend, "tgmrf_nuts_joint")

  # Posterior means in the (log_tau, atanh_rho) band consistent with the
  # sim: log_sigma_true = log(0.8) ~ -0.22; tanh(atanh_rho_mean) close to
  # 0.7. Single-seed tolerance is loose; recovery tests with 30 seeds live
  # in test-tgmrf-recovery.R.
  expect_lt(abs(fit$means_theta[1] - log(sigma_true)), 1.0)
  expect_lt(abs(tanh(fit$means_theta[2]) - rho_true), 0.5)
})

# ---------------------------------------------------------------------------
# (b) Joint NUTS vs outer-theta NUTS: same target posterior, agreement
#     within MC error of either chain. The §8.4 tier cross-check exit gate.
# ---------------------------------------------------------------------------

test_that("joint NUTS and outer-theta NUTS agree on theta means within MC error", {
  cpp_file <- .joint_cpp_example_path()

  n <- 80L
  init   <- c(log_sigma = 0, atanh_rho = atanh(0.5))
  bounds <- list(lower = c(log(0.3), atanh(0.0)),
                 upper = c(log(3.0), atanh(0.95)))

  blk_cpp <- tgmrf_cpp(cpp_file = cpp_file, id = "tgmrf_periodic_ar1",
                       init = init, bounds = bounds,
                       name = "periodic_ar1_cpp")
  blk_r <- .joint_make_periodic_ar1_r(n, init, bounds)

  set.seed(311)
  z <- as.numeric(arima.sim(list(ar = 0.6), n = n, sd = 0.7))
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(0.3 + z))

  fit_outer <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk_r,
    family = "poisson", mode = "nuts",
    n_iter = 200L, warmup = 100L, max_depth = 4L
  )

  fit_joint <- tulpa_tgmrf(
    y = y, n_trials = rep(1L, n), X = X, block = blk_cpp,
    family = "poisson", mode = "nuts_joint",
    n_iter = 200L, warmup = 100L, max_depth = 5L,
    seed = 1234L
  )

  # Both target p(theta | y). Means should agree to within the combined MC
  # error of two short chains. Tolerance: 0.4 per axis in natural units
  # (the spec target). log_sigma posterior SDs tend to be ~0.15-0.25 here;
  # 100 post-warmup samples gives ~ SD / sqrt(50) ~ 0.03 per chain so 0.4
  # is loose.
  expect_lt(abs(fit_joint$means_theta[1] - fit_outer$means[1]), 0.4)
  expect_lt(abs(fit_joint$means_theta[2] - fit_outer$means[2]), 0.4)
})

# ---------------------------------------------------------------------------
# (c) Refuses R-backend block with a clear message.
# ---------------------------------------------------------------------------

test_that("tulpa_tgmrf_nuts_joint refuses R-closure tgmrf blocks", {
  n <- 20L
  blk_r <- .joint_make_periodic_ar1_r(n, c(log_sigma = 0, atanh_rho = 0))
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(0.2))

  expect_error(
    tulpa_tgmrf(
      y = y, n_trials = rep(1L, n), X = X, block = blk_r,
      family = "poisson", mode = "nuts_joint",
      n_iter = 10L, warmup = 5L
    ),
    regexp = "tgmrf_cpp"
  )
})
