# Direct-call fitters get the same tulpa_fit contract as tulpa()-dispatched
# ones (gcol33/tulpa#102): the tulpa_fit class (so the generic S3 methods
# dispatch), the fixed-effect layout, and an explicit posterior-provenance tag
# (draws_kind) the chain-vs-iid diagnostic gate reads. Before the shared
# .finalize_fit() helper, a directly-called fitter returned a bare/under-tagged
# list: generic S3 fell through to the base default, and an iid fit was treated
# as a chain so mcmc_diagnostics() reported a vacuous Rhat ~ 1 (a false
# convergence pass).

test_that(".finalize_fit attaches class, layout, and provenance", {
  # Bare list gains the tulpa_fit class and a registry-derived draws_kind.
  f <- .finalize_fit(list(means = 1:2), backend = "laplace",
                     n_fixed = 2L, fixed_names = c("a", "b"))
  expect_s3_class(f, "tulpa_fit")
  expect_equal(f$backend, "laplace")
  expect_equal(f$draws_kind, "iid")      # laplace emits "iid" in the registry
  expect_equal(f$n_fixed, 2L)
  expect_equal(f$fixed_names, c("a", "b"))

  # A value already on the fit wins over both the argument and the registry.
  g <- .finalize_fit(list(draws_kind = "chain", n_fixed = 99L),
                     backend = "laplace", draws_kind = "iid", n_fixed = 2L)
  expect_equal(g$draws_kind, "chain")
  expect_equal(g$n_fixed, 99L)

  # A non-registry backend must carry an explicit draws_kind (the tgmrf_* case).
  h <- .finalize_fit(list(), backend = "tgmrf_vi", draws_kind = "iid")
  expect_equal(h$draws_kind, "iid")

  # extra_class is prepended; idempotent re-finalize does not duplicate it.
  k <- .finalize_fit(list(), backend = "tgmrf_vi", draws_kind = "iid",
                     extra_class = "tulpa_tgmrf")
  expect_identical(class(k), c("tulpa_tgmrf", "tulpa_fit"))
  k2 <- .finalize_fit(k, backend = "tgmrf_vi", extra_class = "tulpa_tgmrf")
  expect_identical(class(k2), c("tulpa_tgmrf", "tulpa_fit"))

  # A non-list is returned unchanged.
  expect_identical(.finalize_fit(42), 42)
})

test_that("directly-called tulpa_laplace() is classed and provenance-tagged", {
  skip_on_cran()
  set.seed(11)
  n <- 60L
  X <- cbind(1, rnorm(n))
  colnames(X) <- c("(Intercept)", "x")
  eta <- X %*% c(-0.3, 0.8)
  y <- rbinom(n, 1L, plogis(eta))

  fit <- tulpa_laplace(y = y, n_trials = rep(1L, n), X = X, family = "binomial")

  # Generic S3 dispatch is available outside tulpa().
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$n_fixed, 2L)
  expect_equal(fit$fixed_names, c("(Intercept)", "x"))

  # Provenance gate: Laplace emits iid draws, so mcmc_diagnostics() returns the
  # approximation-reliability table (a `laplace_diagnostics`), not a chain Rhat
  # table that would read as a vacuous single-chain convergence pass.
  expect_equal(fit$draws_kind, "iid")
  d <- suppressMessages(mcmc_diagnostics(fit))
  expect_s3_class(d, "laplace_diagnostics")
})

test_that("directly-called VI fit does not fake a convergence pass", {
  skip_on_cran()
  skip_if_not(exists("tulpa_tgmrf"))
  set.seed(12)
  n <- 25L
  z <- as.numeric(arima.sim(list(ar = 0.5), n = n, sd = 1))
  X <- matrix(1, n, 1L)
  y <- rpois(n, exp(0.2 + z))
  blk <- tgmrf(
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

  fit <- tulpa_tgmrf(y = y, n_trials = rep(1L, n), X = X, block = blk,
                     family = "poisson", mode = "vi", n_draws = 300L)

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$draws_kind, "iid")
  # An iid VI fit routes to the reliability table, not a chain Rhat pass.
  expect_s3_class(suppressMessages(mcmc_diagnostics(fit)), "laplace_diagnostics")
})
