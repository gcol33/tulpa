# BRASIL best-rational-approximation engine (Hofreither 2021), the coefficient
# source for the fractional-SPDE rational approximation (gcol33/tulpa#71).
# These exercise the algorithm directly; it is not yet wired into any fitter.

test_that("BRASIL converges to an equioscillating minimax approximation of x^beta", {
  skip_on_cran()
  for (cfg in list(list(b = 0.5, iv = c(1, 100), m = 3),
                   list(b = 0.5, iv = c(0.5, 50), m = 4),
                   list(b = 0.25, iv = c(1, 1000), m = 4),
                   list(b = 0.75, iv = c(1, 200), m = 3))) {
    f <- function(x) x^cfg$b
    res <- tulpa:::.brasil(f, cfg$iv, cfg$m, tol = 1e-5)
    expect_true(res$converged)
    # Equioscillation: the largest and smallest local-error peaks agree.
    expect_lt(res$deviation, 1e-4)
    # Uniform error on a dense grid is at the converged peak level.
    xs <- seq(cfg$iv[1], cfg$iv[2], length.out = 4000)
    err <- max(abs(f(xs) - tulpa:::.bary_eval(res$br, xs)))
    expect_lt(err, 5 * res$error)
  }
})

test_that("BRASIL error decreases as the rational degree grows", {
  skip_on_cran()
  f <- function(x) x^0.5
  iv <- c(1, 100)
  errs <- vapply(2:5, function(m) tulpa:::.brasil(f, iv, m, tol = 1e-6)$error,
                 numeric(1))
  expect_true(all(diff(errs) < 0))   # strictly decreasing
})

test_that("BRASIL zeros and poles are real, negative, and reconstruct the rational", {
  skip_on_cran()
  f <- function(x) x^0.5
  iv <- c(0.5, 50)
  res <- tulpa:::.brasil(f, iv, 4, tol = 1e-6)
  z <- tulpa:::.bary_zeros(res$br)
  p <- tulpa:::.bary_poles(res$br)
  expect_equal(length(z), 4L)
  expect_equal(length(p), 4L)
  # Roots of the minimax approximation of x^beta on a positive interval are
  # real, negative, and interlacing (poles < zeros < 0 < interval).
  expect_true(all(p < 0)); expect_true(all(z < 0))
  # Reconstruct r(x) = scale * prod(x - z) / prod(x - p) and match the
  # barycentric evaluation everywhere -- confirms the pole / zero extraction.
  xm <- mean(iv)
  scale <- tulpa:::.bary_eval(res$br, xm) * prod(xm - p) / prod(xm - z)
  xs <- seq(iv[1], iv[2], length.out = 2000)
  rpz <- vapply(xs, function(x) scale * prod(x - z) / prod(x - p), numeric(1))
  expect_lt(max(abs(rpz - tulpa:::.bary_eval(res$br, xs))), 1e-8)
})

test_that(".poly_from_roots builds prod(x - r) in ascending powers", {
  # (x - 1)(x + 2) = x^2 + x - 2  ->  c(-2, 1, 1)
  expect_equal(tulpa:::.poly_from_roots(c(1, -2)), c(-2, 1, 1))
})
