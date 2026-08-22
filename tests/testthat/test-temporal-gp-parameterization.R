# The centered and non-centered temporal GP are reparameterizations of one
# posterior, so their densities differ by exactly the forward transform's
# log-determinant (gcol33/tulpa#499).
#
# What that pins is the conditional variance the two branches use. The
# non-centered branch reaches it through the transform's scale
# a_t = sigma sqrt(1 - rho_t^2) and the centered branch through
# cond_var_t = sigma^2 (1 - rho_t^2). Flooring DIFFERENT quantities --
# sigma^2 max(1 - rho^2, eps) against max(sigma^2 (1 - rho^2), eps) -- leaves
# the two agreeing wherever the floor is slack and disagreeing by orders of
# magnitude wherever it binds, which is any long lengthscale on a fine time
# grid. The identity below is what no single determinant can absorb.

# log|det A| for the AR1 state-space transform f = A z, plus the (2 pi)
# normalizers the centered density carries and the non-centered one does not.
# Written here rather than read off the engine: the arbiter has to be
# independent of the code under test.
tgp_log_det_shift <- function(times, n_groups, sigma2, phi) {
  T_times <- length(times)
  dt <- diff(times)
  rho <- exp(-dt / phi)
  omr2 <- pmax(1 - rho^2, 1e-10)          # kAr1StationaryFloor
  a <- sqrt(sigma2) * sqrt(omr2)
  n_groups * (0.5 * T_times * log(2 * pi) + 0.5 * log(sigma2) + sum(log(a)))
}

tgp_pair <- function(times, z, log_sigma2, logit_phi, n_groups = 1L,
                     phi_lower = 0.01, phi_upper = 10) {
  nc <- cpp_test_temporal_gp_density(times, n_groups, z, log_sigma2, logit_phi,
                                     parameterization = 1L,
                                     phi_lower = phi_lower,
                                     phi_upper = phi_upper)
  ce <- cpp_test_temporal_gp_density(times, n_groups, nc$field, log_sigma2,
                                     logit_phi, parameterization = 0L,
                                     phi_lower = phi_lower,
                                     phi_upper = phi_upper)
  list(nc = nc, centered = ce)
}

test_that("the two parameterizations differ by the transform's log-determinant", {
  set.seed(4L)
  times <- sort(runif(8L, 0, 5))
  z <- rnorm(8L)
  p <- tgp_pair(times, z, log_sigma2 = log(0.7), logit_phi = 0.4)

  shift <- tgp_log_det_shift(times, 1L, p$nc$sigma2, p$nc$phi)
  expect_equal(p$nc$log_post - p$centered$log_post, shift, tolerance = 1e-10)
})

test_that("the identity holds where the (1 - rho^2) floor binds", {
  # dt / phi = 1e-11, so 1 - rho^2 ~ 2e-11 is below the 1e-10 floor at every
  # step. sigma^2 = 1e-4 is small enough that flooring the product instead of
  # the correlation factor would place the conditional variance six orders of
  # magnitude away.
  times <- c(0, 1e-5, 2e-5, 3e-5)
  z <- c(0.3, -0.8, 0.15, 0.6)
  # bounded_from_logit at logit 0 sits at the midpoint of (lower, upper).
  p <- tgp_pair(times, z, log_sigma2 = log(1e-4), logit_phi = 0,
                phi_lower = 1, phi_upper = 2e6)

  expect_gt(p$nc$phi, 1e5)
  rho <- exp(-diff(times) / p$nc$phi)
  expect_lt(max(1 - rho^2), 1e-10)        # the floor is genuinely binding

  shift <- tgp_log_det_shift(times, 1L, p$nc$sigma2, p$nc$phi)
  expect_equal(p$nc$log_post - p$centered$log_post, shift, tolerance = 1e-9)
})

test_that("the identity holds over several independent groups", {
  set.seed(9L)
  times <- sort(runif(5L, 0, 3))
  z <- rnorm(15L)
  p <- tgp_pair(times, z, log_sigma2 = log(2.1), logit_phi = -0.6,
                n_groups = 3L)

  shift <- tgp_log_det_shift(times, 3L, p$nc$sigma2, p$nc$phi)
  expect_equal(p$nc$log_post - p$centered$log_post, shift, tolerance = 1e-10)
})
