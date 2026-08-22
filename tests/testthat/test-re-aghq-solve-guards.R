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
