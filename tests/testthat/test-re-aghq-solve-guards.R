# Failure signals on the compiled AGHQ entry points: the per-group solve status
# cpp_aghq_blups reports, and the validation of the tensor quadrature grid.
# Structural (tier 1) -- no fit, no optimizer, just the C++ boundary.

# R-closure oracle over `ng` d-dimensional groups whose data information at any b
# is negH_of(g) * I and whose score is 0, so the mode-find stops at b = 0 on
# every group and the penalized precision is exactly negH_of(g) * I + Sigma^-1.
.aghq_flat_oracle <- function(ng, d = 1L, negH_of = function(g) 1) {
  build <- function(theta) list(
    grad_hess = function(g, b) list(logL = 0, grad = rep(0, d),
                                    negH = diag(negH_of(g), d)),
    node_ll = function(g, B) rep(0, nrow(B)))
  cpp_aghq_make_rclosure_oracle(build, ng, d, 1L)
}

test_that("cpp_aghq_blups reports a group whose penalized precision does not factor", {
  ng  <- 4L
  # Group 2's information is negative enough that negH + Sigma^-1 = -9 is not PD
  # at par = c(theta = 0, log-SD = 0), i.e. Sigma = 1.
  orc <- .aghq_flat_oracle(ng, 1L, function(g) if (g == 2L) -10 else 1)
  bl  <- cpp_aghq_blups(c(0, 0), orc, 1L, FALSE)

  expect_length(bl$group_ok, ng)
  expect_false(bl$group_ok[2L])
  expect_true(all(bl$group_ok[-2L]))

  expect_true(all(is.na(bl$bhat[2L, ])))
  expect_true(all(is.na(bl$bvar[2L, ])))
  expect_true(all(is.na(bl$bcov[2L, , ])))

  # The groups that did solve are untouched: negH + P = 2, so C = 1/2.
  expect_equal(unname(bl$bhat[1L, 1L]), 0, tolerance = 1e-10)
  expect_equal(unname(bl$bvar[1L, 1L]), 0.5, tolerance = 1e-10)
  expect_equal(unname(bl$bvar[3L, 1L]), 0.5, tolerance = 1e-10)

  # The objective declines at the same parameter, through the same solve.
  expect_equal(cpp_aghq_objective(c(0, 0), orc, 1L, FALSE, 5L, 1.0), -1e10)
  expect_false(cpp_aghq_objective_grad(c(0, 0), orc, 1L, FALSE, 5L, 1.0)$ok)
})

test_that("cpp_aghq_blups solves every group when each precision factors", {
  orc <- .aghq_flat_oracle(3L)
  bl  <- cpp_aghq_blups(c(0, 0), orc, 1L, FALSE)
  expect_true(all(bl$group_ok))
  expect_false(anyNA(bl$bhat))
  expect_false(anyNA(bl$bvar))
  expect_false(anyNA(bl$bcov))
})

test_that("the reported variance tracks a tiny log-SD instead of one constant", {
  # Sigma is L L' in log-Cholesky coordinates plus a PD jitter. The jitter is
  # RELATIVE to each diagonal entry (gcol33/tulpa#595), so it stays negligible
  # at every representable scale: with negH = 1 the posterior variance is
  # 1 / (1 + 1 / Sigma), and at a Sigma far below 1 that is Sigma itself.
  # An ABSOLUTE 1e-10 jitter made this one constant across ten orders of
  # magnitude of true variance.
  orc <- .aghq_flat_oracle(2L)
  for (log_sd in c(-100, -300)) {
    sigma <- exp(2 * log_sd)
    expect_gt(sigma, 0)                       # representable, so no backstop
    bl <- cpp_aghq_blups(c(0, log_sd), orc, 1L, FALSE)
    expect_true(all(bl$group_ok))
    expect_false(any(bl$sigma_jitter_floored))
    expect_equal(unname(bl$bvar[1L, 1L]), 1 / (1 + 1 / sigma),
                 tolerance = 1e-9, info = paste("log_sd", log_sd))
  }
  # Two different degenerate values now give two different answers, which is
  # the whole of what the absolute floor destroyed.
  v100 <- cpp_aghq_blups(c(0, -100), orc, 1L, FALSE)$bvar[1L, 1L]
  v300 <- cpp_aghq_blups(c(0, -300), orc, 1L, FALSE)$bvar[1L, 1L]
  expect_gt(v100 / v300, 1e100)
})

test_that("a log-SD past underflow takes the backstop and says so", {
  # Below about log-SD -372 the square underflows to exactly zero, which is the
  # one case a relative jitter cannot serve: there the absolute backstop IS the
  # covariance, and the reported variance is not a function of the parameter.
  # That is reported rather than inherited.
  jitter   <- 1e-10
  expected <- 1 / (1 + 1 / jitter)

  orc <- .aghq_flat_oracle(2L)
  for (log_sd in c(-400, -500, -700)) {
    expect_identical(exp(2 * log_sd), 0)      # the premise
    bl <- cpp_aghq_blups(c(0, log_sd), orc, 1L, FALSE)
    expect_true(all(bl$group_ok))
    expect_true(all(bl$sigma_jitter_floored), info = paste("log_sd", log_sd))
    expect_equal(unname(bl$bvar[1L, 1L]), expected, tolerance = 1e-12,
                 info = paste("log_sd", log_sd))
  }
  # The flag is per-coordinate and is FALSE on an ordinary fit, so a reader can
  # tell "the variance really is small" from "the jitter is what you are
  # reading" without knowing the parameter.
  ok <- cpp_aghq_blups(c(0, 0), orc, 1L, FALSE)
  expect_length(ok$sigma_jitter_floored, 1L)
  expect_false(any(ok$sigma_jitter_floored))
})

test_that("only the underflowed coordinate of a diagonal block is flagged", {
  # Two uncorrelated coordinates, one healthy and one past underflow: the flag
  # names the coordinate, not the fit.
  orc <- .aghq_flat_oracle(2L, 2L)
  bl  <- cpp_aghq_blups(c(0, 0, -500), orc, 2L, FALSE)
  expect_identical(as.logical(bl$sigma_jitter_floored), c(FALSE, TRUE))
  expect_equal(unname(bl$bvar[1L, 1L]), 0.5, tolerance = 1e-10)
  expect_equal(unname(bl$bvar[1L, 2L]), 1 / (1 + 1e10), tolerance = 1e-12)
})

test_that("a non-finite par declines through the per-group solve", {
  # The covariance-level guard cannot be reached from a finite par, so this is
  # the channel a diverged coordinate actually leaves its signal on.
  orc <- .aghq_flat_oracle(2L)
  bl  <- cpp_aghq_blups(c(0, NaN), orc, 1L, FALSE)
  expect_false(any(bl$group_ok))
  expect_true(all(is.na(bl$bhat)))
  expect_true(all(is.na(bl$bvar)))
})

test_that("the AGHQ tensor grid refuses a node count past the cap", {
  # 9 nodes on 7 axes is 4782969 nodes, past the 1048576 cap; the message names
  # the per-axis counts that produced it. The grid is built before the oracle is
  # touched, so this never reaches a group.
  orc7 <- .aghq_flat_oracle(1L, 7L)
  expect_error(cpp_aghq_objective(rep(0, 8), orc7, 7L, FALSE, 9L, 1.0),
               "1048576", fixed = TRUE)
  expect_error(cpp_aghq_objective(rep(0, 8), orc7, 7L, FALSE, 9L, 1.0),
               "9 x 9 x 9 x 9 x 9 x 9 x 9", fixed = TRUE)
  expect_error(cpp_aghq_objective_grad(rep(0, 8), orc7, 7L, FALSE, 9L, 1.0),
               "exceeds", fixed = TRUE)
})

test_that("a tensor grid inside the cap still builds", {
  orc6 <- .aghq_flat_oracle(1L, 6L)                 # 5^6 = 15625 nodes
  expect_true(is.finite(cpp_aghq_objective(rep(0, 7), orc6, 6L, FALSE, 5L, 1.0)))
})

test_that("the AGHQ node-count request is validated before it is broadcast", {
  orc <- .aghq_flat_oracle(2L)
  expect_error(cpp_aghq_objective(c(0, 0), orc, 1L, FALSE, 0L, 1.0), "n_quad")
  expect_error(cpp_aghq_objective(c(0, 0), orc, 1L, FALSE, -3L, 1.0), "n_quad")
  # Length neither 1 nor one entry per covariance block.
  expect_error(cpp_aghq_objective(c(0, 0), orc, 1L, FALSE, c(3L, 3L), 1.0),
               "length")
})

# ---------------------------------------------------------------------------
# The solve status reaches the CALLER (gcol33/tulpa#605). `cpp_aghq_blups`
# already reports it per group; what is tested here is that
# `tulpa_re_aghq()` hands it on, so a consumer conditions its per-group reads
# on a logical vector instead of parsing indices out of a warning message.
# ---------------------------------------------------------------------------

# Small binomial GLMM with a random intercept -- one fit, a few groups.
.aghq_group_ok_fixture <- function(seed = 4L, ng = 12L, n_per = 6L) {
  set.seed(seed)
  N <- ng * n_per
  g <- rep(seq_len(ng), each = n_per)
  x <- rnorm(N); X <- cbind(1, x); nt <- rep(3L, N)
  u <- rnorm(ng, 0, 0.7)
  y <- rbinom(N, nt, plogis(0.3 + 0.7 * x + u[g]))
  l1pe <- function(z) ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
  site <- function(theta) {
    eta_fixed <- as.numeric(X %*% theta)
    list(eta_re = eta_fixed,
         deriv = function(rows, eta) {
           p <- plogis(eta)
           list(logL = y[rows] * eta - nt[rows] * l1pe(eta),
                d1 = y[rows] - nt[rows] * p, d2 = -nt[rows] * p * (1 - p))
         },
         lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
  }
  list(ng = ng, n = N, idx = g, site = site)
}

.aghq_group_ok_fit <- function(fx) {
  tulpa_re_aghq(
    theta0   = c(0, 0),
    re_terms = list(list(idx = fx$idx, n_groups = fx$ng, n_coefs = 1L)),
    Sigma0   = list(matrix(0.25, 1, 1)),
    make_site = fx$site, n_obs = fx$n, n_quad = 3L)
}

test_that("tulpa_re_aghq reports group_ok on a fit whose every group solved", {
  skip_on_cran()
  fx  <- .aghq_group_ok_fixture()
  fit <- expect_silent(.aghq_group_ok_fit(fx))

  expect_type(fit$group_ok, "logical")
  expect_length(fit$group_ok, fx$ng)
  expect_true(all(fit$group_ok))
  # The flag is what the NA rows would mean, so an all-TRUE fit carries none.
  expect_false(anyNA(fit$blup[[1L]]))
  expect_false(anyNA(fit$blup_var[[1L]]))
  expect_false(any(vapply(fit$blup_cov_g, anyNA, logical(1))))
})

test_that("the optimizer's evaluation counts reach the caller", {
  skip_on_cran()
  # The joint driver is one stats::optim call, so a consumer reporting the work
  # behind a fit has nothing to read unless the counts are handed on -- which is
  # why tulpaObs reported NA iterations on a fit it declared converged
  # (gcol33/tulpaObs#281). BFGS counts EVALUATIONS, and they are passed through
  # verbatim under optim's own names rather than relabelled as iterations.
  fx  <- .aghq_group_ok_fixture()
  fit <- .aghq_group_ok_fit(fx)
  expect_true(fit$converged)
  expect_true(all(c("function", "gradient") %in% names(fit$counts)))
  expect_gt(fit$counts[["function"]], 0)
  expect_true(all(is.finite(fit$counts)))
})

test_that("a failed group is a field on the fit, not only a warning", {
  skip_on_cran()
  # Every group of the fixture above solves, and a group that does NOT solve
  # takes the objective to its -1e10 penalty at the same parameter -- so the
  # optimizer never returns a point carrying one. The status is injected at the
  # extractor instead, which is exactly the boundary the return value crosses;
  # what the compiled side does with a genuinely unsolvable group is pinned by
  # the first test in this file.
  bad  <- 2L
  real <- cpp_aghq_blups
  fail_one <- function(par, oracle, nc, full) {
    bl <- real(par, oracle, nc, full)
    bl$group_ok[bad] <- FALSE
    bl$bhat[bad, ] <- NA_real_
    bl$bvar[bad, ] <- NA_real_
    bl$bcov[bad, , ] <- NA_real_
    bl
  }
  testthat::local_mocked_bindings(cpp_aghq_blups = fail_one, .package = "tulpa")

  fx <- .aghq_group_ok_fixture()
  expect_warning(fit <- .aghq_group_ok_fit(fx),
                 "per-group posterior solve failed for 1 of 12 groups")

  expect_length(fit$group_ok, fx$ng)
  expect_false(fit$group_ok[bad])
  expect_true(all(fit$group_ok[-bad]))

  # The flag explains exactly the NA entries, and no others: a caller dropping
  # the groups it marks drops every unusable row and keeps every usable one.
  expect_true(all(is.na(fit$blup[[1L]][bad, ])))
  expect_false(anyNA(fit$blup[[1L]][-bad, , drop = FALSE]))
  expect_true(all(is.na(fit$blup_var[[1L]][bad, ])))
  expect_true(all(is.na(fit$blup_cov_g[[bad]])))
  expect_false(any(vapply(fit$blup_cov_g[-bad], anyNA, logical(1))))
})

# ---------------------------------------------------------------------------
# The failure sentinel is not an objective value (gcol33/tulpa#606).
# cpp_aghq_objective() reports a large finite penalty where a group's solve
# failed, so that stats::optim rejects the point. A fit built ON that value
# reports the sentinel as a log-likelihood, derives its covariance from the
# finite-difference curvature of the penalty, and -- since `reltol` is relative
# to |f| = 1e10 -- can declare convergence after one hair of a step.
# ---------------------------------------------------------------------------

test_that("the sentinel R tests against is the one the objective reports", {
  # No literal on the R side: the value is the compiled producer's.
  orc <- .aghq_flat_oracle(4L, 1L, function(g) if (g == 2L) -10 else 1)
  expect_identical(cpp_aghq_objective(c(0, 0), orc, 1L, FALSE, 5L, 1.0),
                   cpp_aghq_fail_penalty())
  expect_identical(cpp_aghq_objective_grad(c(0, 0), orc, 1L, FALSE, 5L, 1.0)$f,
                   cpp_aghq_fail_penalty())
  # And a solvable configuration is on the other side of the predicate.
  expect_true(.aghq_is_fail(cpp_aghq_fail_penalty()))
  expect_true(.aghq_is_fail(NaN))
  expect_false(.aghq_is_fail(cpp_aghq_objective(c(0, 0), .aghq_flat_oracle(3L),
                                                1L, FALSE, 5L, 1.0)))
})

test_that("tulpa_re_aghq refuses a start point the objective is undefined at", {
  skip_on_cran()
  # Group 2's information is negative enough that its penalized precision does
  # not factor at any covariance the optimizer reaches, so every evaluation of
  # the objective is the sentinel.
  ng  <- 6L
  orc <- .aghq_flat_oracle(ng, 1L, function(g) if (g == 2L) -1e6 else 1)
  run <- function(...) tulpa_re_aghq(
    theta0 = log(10), re_terms = list(list(n_groups = ng, n_coefs = 1L)),
    Sigma0 = list(matrix(0.25, 1, 1)), oracle = orc, n_quad = 3L, ...)

  expect_warning(fit <- run(), "not defined at the starting parameters")
  expect_null(fit)
  # The refusal names the group, so a caller is told what to change.
  expect_warning(run(), "1 of 6 groups (2)", fixed = TRUE)

  # The ridge is what made this reachable in the wild: with one, the penalty
  # surface is no longer flat in every coordinate, so the singular-Hessian
  # guard downstream does not catch it. This is the tulpaObs configuration
  # (`nmix_laplace_re()` passes theta_prior_sd = 100).
  expect_warning(fit <- run(theta_prior_sd = 100),
                 "not defined at the starting parameters")
  expect_null(fit)

  # The contrast: a fixture whose groups all solve is fit, and what it reports
  # as a marginal likelihood is on the other side of the predicate. (The flat
  # oracle above cannot serve as that arm -- with every group repaired its
  # objective is constant in theta, which the singular-Hessian guard refuses
  # for its own reasons.)
  fx <- .aghq_group_ok_fixture()
  ok <- .aghq_group_ok_fit(fx)
  expect_false(is.null(ok))
  expect_false(.aghq_is_fail(ok$log_marginal))
  expect_true(all(ok$group_ok))
})

test_that("agq_fit errors rather than reporting the sentinel as a log-likelihood", {
  skip_on_cran()
  set.seed(7)
  ng <- 8L; npg <- 5L; n <- ng * npg
  g <- rep(seq_len(ng), each = npg)
  X <- cbind(1, rnorm(n))
  y <- rpois(n, exp(0.4 + 0.3 * X[, 2]))

  # The built-in GLMM families keep every per-group precision PD, so the
  # sentinel is unreachable through the data; it is injected at the objective,
  # which is the value optim reports back through opt$value.
  testthat::local_mocked_bindings(
    cpp_aghq_objective = function(par, oracle, nc, full, n_quad, lkj_eta)
      cpp_aghq_fail_penalty(),
    .package = "tulpa")

  expect_error(
    agq_fit(y = y, X = X, group = g, n_groups = ng, family = "poisson",
            n_quad = 1L),
    "failure sentinel")
})
