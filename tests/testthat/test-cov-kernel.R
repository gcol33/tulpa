# The isotropic covariance kernel's phi-derivative matches its value function
# (gcol33/tulpa#142 A5).
#
# dcov_dphi existed in two copies that had drifted: the GP copy returned half
# the true Gaussian derivative, and neither implemented SPHERICAL, so a
# spherical fit took its covariance from one kernel and its gradient from
# another. Both are user-reachable -- spatial_gp() advertises cov = "gaussian"
# and "spherical". Checking each derivative against a numerical derivative of
# the value it differentiates catches exactly this class of drift.

COV <- c(exponential = 0L, matern32 = 1L, gaussian = 2L, spherical = 3L)

test_that("dcov_dphi matches a numerical derivative of compute_cov", {
  h <- 1e-6
  for (nm in names(COV)) {
    ct <- COV[[nm]]
    for (sigma2 in c(0.5, 1.0, 3.0)) {
      for (phi in c(0.3, 1.0, 2.5)) {
        # stay off the spherical kink at d == phi
        for (d in c(0.05, 0.25, 0.7, 1.3, 3.0)) {
          if (identical(nm, "spherical") && abs(d - phi) < 1e-2) next
          num <- (cpp_test_compute_cov(d, sigma2, phi + h, ct) -
                  cpp_test_compute_cov(d, sigma2, phi - h, ct)) / (2 * h)
          got <- cpp_test_dcov_dphi(d, sigma2, phi, ct)
          expect_equal(got, num, tolerance = 1e-4,
                       info = paste(nm, "sigma2:", sigma2, "phi:", phi, "d:", d))
        }
      }
    }
  }
})

test_that("each covariance kernel matches its closed form", {
  sigma2 <- 2.0; phi <- 1.5
  for (d in c(0.1, 0.8, 1.4, 2.2)) {
    expect_equal(cpp_test_compute_cov(d, sigma2, phi, COV[["exponential"]]),
                 sigma2 * exp(-d / phi), tolerance = 1e-12)
    u <- sqrt(3) * d / phi
    expect_equal(cpp_test_compute_cov(d, sigma2, phi, COV[["matern32"]]),
                 sigma2 * (1 + u) * exp(-u), tolerance = 1e-8)
    expect_equal(cpp_test_compute_cov(d, sigma2, phi, COV[["gaussian"]]),
                 sigma2 * exp(-(d / phi)^2), tolerance = 1e-12)
    r <- d / phi
    sph <- if (d >= phi) 0 else sigma2 * (1 - 1.5 * r + 0.5 * r^3)
    expect_equal(cpp_test_compute_cov(d, sigma2, phi, COV[["spherical"]]),
                 sph, tolerance = 1e-12)
  }
})

test_that("the gaussian phi-derivative carries its factor of two", {
  # The GP copy returned cov_val*d^2/phi^3 instead of cov_val*2*d^2/phi^3, i.e.
  # exactly half. Pin the closed form so the halved version cannot come back.
  sigma2 <- 1.3; phi <- 0.9
  for (d in c(0.2, 0.6, 1.5)) {
    k <- sigma2 * exp(-(d / phi)^2)
    expect_equal(cpp_test_dcov_dphi(d, sigma2, phi, COV[["gaussian"]]),
                 k * 2 * d^2 / phi^3, tolerance = 1e-10)
  }
})

test_that("the spherical derivative is its own, not the exponential's", {
  # SPHERICAL fell through `default:` to the exponential derivative, so the
  # value and the gradient described different kernels.
  sigma2 <- 1.0; phi <- 1.0
  for (d in c(0.2, 0.5, 0.85)) {
    r <- d / phi
    expect_equal(cpp_test_dcov_dphi(d, sigma2, phi, COV[["spherical"]]),
                 sigma2 * 1.5 * r * (1 - r^2) / phi, tolerance = 1e-10)
    # and it is NOT the exponential derivative it used to return
    k_exp <- sigma2 * exp(-d / phi)
    expect_false(isTRUE(all.equal(
      cpp_test_dcov_dphi(d, sigma2, phi, COV[["spherical"]]),
      k_exp * d / (phi * phi))))
  }
  # beyond the range the kernel is identically zero, so it is flat in phi
  expect_equal(cpp_test_dcov_dphi(1.5, sigma2, phi, COV[["spherical"]]), 0)
})

test_that("the derivative vanishes at zero distance for every kernel", {
  for (nm in names(COV)) {
    expect_equal(cpp_test_dcov_dphi(0, 1.0, 1.0, COV[[nm]]), 0, info = nm)
  }
})
