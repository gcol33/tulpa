# ---------------------------------------------------------------------------
# tgmrf P7 -- C++ fast backend (tgmrf_cpp()) equivalence with the
# R-closure path (tgmrf()).
#
# The hard exit gate (R-vs-C++ equivalence, dev_notes/plans/generic-todo.md section 8.5):
# defining the same periodic-AR1 block in R and in C++ and fitting both
# with tulpa_nested_laplace() on identical data and identical outer grid
# must agree on log_marginal and theta_mean to within a tight numerical
# tolerance. Both paths feed the same precomputed CSC triples into the
# multi-block Newton driver -- the only place they diverge is the source
# of Q(theta_k). Any discrepancy here means the C++ kernel is producing a
# different precision matrix from the R closure.
#
# Tests are skipped on CRAN because Rcpp::sourceCpp compilation is slow
# (~30-60 s the first time, < 1 s after the cache warms).
# ---------------------------------------------------------------------------

# Shared periodic-AR1 R closure -- matches the kernel in
# inst/examples/tgmrf_periodic_ar1.cpp exactly.
make_periodic_ar1_r <- function(n, init,
                                bounds = NULL,
                                theta_grid = NULL) {
  blk <- tgmrf(
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
  if (!is.null(theta_grid)) blk$theta_grid_built <- theta_grid
  blk
}

# Locate the example .cpp -- works both under devtools::load_all (source
# tree) and under installed package (system.file).
.tgmrf_cpp_example_path <- function() {
  p <- system.file("examples", "tgmrf_periodic_ar1.cpp", package = "tulpa")
  if (nzchar(p) && file.exists(p)) return(p)
  fallback <- file.path("..", "..", "inst", "examples",
                        "tgmrf_periodic_ar1.cpp")
  if (file.exists(fallback)) return(normalizePath(fallback))
  testthat::skip("Cannot locate inst/examples/tgmrf_periodic_ar1.cpp")
}

test_that("tgmrf_cpp() compiles, registers, and fits a periodic-AR1 sim", {
  skip_on_cran()
  skip_if_fast()
  cpp_file <- .tgmrf_cpp_example_path()

  # kN = 80 in the .cpp.
  n <- 80L
  init <- c(log_sigma = 0, atanh_rho = atanh(0.5))
  bounds <- list(lower = c(log(0.3), atanh(0.0)),
                 upper = c(log(3.0), atanh(0.95)))

  blk <- tgmrf_cpp(
    cpp_file = cpp_file,
    id       = "tgmrf_periodic_ar1",
    init     = init,
    bounds   = bounds,
    name     = "periodic_ar1_cpp"
  )

  expect_s3_class(blk, "tgmrf")
  expect_s3_class(blk, "tulpa_latent_block")
  expect_identical(blk$backend, "cpp")
  expect_identical(blk$cpp_id, "tgmrf_periodic_ar1")
  expect_identical(blk$n_latent, n)
  expect_identical(blk$theta_dim, 2L)
  expect_identical(blk$theta_names, c("log_sigma", "atanh_rho"))

  set.seed(2026)
  sigma_true <- 0.8; rho_true <- 0.7
  z <- numeric(n)
  z[1] <- rnorm(1, 0, sigma_true)
  for (t in 2:n) {
    z[t] <- rho_true * z[t - 1] + rnorm(1, 0, sigma_true * sqrt(1 - rho_true^2))
  }
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(0.4 + z))

  fit <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = blk, family = "poisson",
    control = list(max_iter = 100L, tol = 1e-8)
  )
  expect_true(all(is.finite(fit$log_marginal)))
  # Posterior weight should sit in a sensible band around the truth. Single
  # seed -- not a strict recovery threshold (see test-tgmrf-recovery.R).
  expect_true(abs(fit$theta_mean[1] - log(sigma_true)) < 2.0)
  expect_true(abs(tanh(fit$theta_mean[2]) - rho_true) < 0.6)
})

test_that("tgmrf_cpp() and tgmrf() agree to numerical tol on the same fit", {
  skip_on_cran()
  skip_if_fast()
  cpp_file <- .tgmrf_cpp_example_path()

  n <- 80L
  init <- c(log_sigma = 0, atanh_rho = atanh(0.5))

  # Shared outer grid (same theta_grid_built for both paths so they hit the
  # same joint-grid rows).
  grid <- as.matrix(expand.grid(
    log_sigma = seq(log(0.5), log(2.0), length.out = 3L),
    atanh_rho = seq(atanh(0.1), atanh(0.8), length.out = 3L)
  ))

  blk_r <- make_periodic_ar1_r(n, init, theta_grid = grid)
  blk_c <- tgmrf_cpp(
    cpp_file = cpp_file,
    id       = "tgmrf_periodic_ar1",
    init     = init,
    name     = "periodic_ar1_cpp"
  )
  blk_c$theta_grid_built <- grid

  set.seed(42)
  z <- as.numeric(arima.sim(list(ar = 0.6), n = n, sd = 0.7))
  X <- matrix(1, nrow = n, ncol = 1L)
  y <- rpois(n, exp(0.2 + z))

  fit_r <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = blk_r, family = "poisson",
    control = list(max_iter = 100L, tol = 1e-10)
  )
  fit_c <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = blk_c, family = "poisson",
    control = list(max_iter = 100L, tol = 1e-10)
  )

  # The two paths feed bit-identical CSC triples into the same Newton
  # driver. Tolerance is set at 1e-6 to absorb double-rounding through the
  # two different Q-construction code paths (R bandSparse + setSparse vs.
  # C++ triplet assembly) -- the inner Laplace approximation itself is
  # deterministic.
  expect_equal(fit_c$log_marginal, fit_r$log_marginal, tolerance = 1e-6)
  expect_equal(fit_c$theta_mean,   fit_r$theta_mean,   tolerance = 1e-6,
               ignore_attr = TRUE)
  expect_equal(fit_c$theta_sd,     fit_r$theta_sd,     tolerance = 1e-6,
               ignore_attr = TRUE)
  expect_equal(fit_c$modes,        fit_r$modes,        tolerance = 1e-6,
               ignore_attr = TRUE)
})
