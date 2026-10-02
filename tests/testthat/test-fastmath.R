# tulpa::fastmath::exp / log (src/fastmath.h, ported from Arm Optimized
# Routines) feed the binomial and Poisson likelihood kernels. Each is within
# about 0.52 ulp of the true value, as is a good system libm, so the two may
# differ by an ulp; past two, the port is wrong somewhere.

ulps_apart <- function(a, b) {
  scale <- pmax(abs(b) * .Machine$double.eps, 2^-1074)
  abs(a - b) / scale
}

test_that("fastmath exp matches the system exp to within two ulps", {
  set.seed(1)
  x <- c(runif(2e4, -745, 709.7), runif(2e4, -30, 30), rnorm(2e4, 0, 1e-3),
         runif(2e3, -745.1, -708), runif(2e3, 700, 709.78),
         -.Machine$double.xmin, 2^-60, -2^-60, 1e-300, -1)
  a <- cpp_test_fastmath(x, "exp")
  b <- exp(x)
  expect_true(all(is.finite(a) == is.finite(b)))
  ok <- is.finite(b)
  expect_lte(max(ulps_apart(a[ok], b[ok])), 2)
})

test_that("fastmath exp returns the special values the C library does", {
  x <- c(0, -0, Inf, -Inf, NaN, 710, 1e308, -746, -1e308)
  a <- cpp_test_fastmath(x, "exp")
  expect_identical(a[1:4], c(1, 1, Inf, 0))
  expect_true(is.nan(a[5]))
  expect_identical(a[6:9], c(Inf, Inf, 0, 0))
})

test_that("fastmath log matches the system log to within two ulps", {
  set.seed(2)
  x <- c(exp(runif(2e4, -700, 700)), runif(2e4, 0.9, 1.1),
         1 + rnorm(2e4, 0, 1e-6), runif(2e4, 1, 2),
         2^-1074 * c(1, 3, 1e6), .Machine$double.xmax, 2^-1022)
  a <- cpp_test_fastmath(x, "log")
  b <- log(x)
  expect_lte(max(ulps_apart(a, b)), 2)
})

test_that("fastmath log returns the special values the C library does", {
  a <- cpp_test_fastmath(c(1, 0, -0, Inf, -1, -Inf, NaN), "log")
  expect_identical(a[1:4], c(0, -Inf, -Inf, Inf))
  expect_true(all(is.nan(a[5:7])))
})
