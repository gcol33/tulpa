# The two NNGP conditional-moment cores, on one input (gcol33/tulpa#447).
#
# tulpa_nngp::cond_moments is the templated core the autodiff and value kernels
# run; tulpa_linalg::nngp_conditional_moments is the double core the Laplace and
# batched NNGP paths run. test-nngp-twin.R holds each templated kernel against
# its own double twin, and both of those route through the FIRST core, so the
# second one is outside its reach: a joint nested-Laplace NNGP fit and an HMC or
# SVC fit of the same field could factorize different matrices with nothing to
# say so.
#
# `jitter` is the thing they have to agree on. It is a diagonal NUGGET --
# C_jj + jitter before the factorization -- so it is part of the density being
# evaluated. A pivot FLOOR at the same number leaves a well-conditioned input
# untouched and replaces an ill-conditioned one, which puts the whole
# divergence on exactly the inputs where it matters and leaves the result
# looking ordinary.

.nngp_cores <- function(C, c_vec, w_nb, sigma2 = 1, jitter = 0,
                        var_floor = 1e-8) {
  n <- nrow(C)
  # row-major buffer: C is symmetric here, but the core is documented row-major.
  tulpa:::cpp_test_nngp_cond_cores(as.numeric(t(C)), n, c_vec, w_nb,
                                   sigma2, jitter, var_floor)
}

.nngp_fixture <- function(n = 5L, phi = 0.5, seed = 447L) {
  set.seed(seed)
  co <- matrix(stats::runif(2 * n), n, 2)
  new <- c(0.5, 0.5)
  d_nb <- as.matrix(stats::dist(co))
  d_new <- sqrt(colSums((t(co) - new)^2))
  list(C = exp(-d_nb / phi), c_vec = exp(-d_new / phi),
       w_nb = stats::rnorm(n))
}

test_that("the two cores return the same kriging moments at every jitter", {
  f <- .nngp_fixture()
  for (jit in c(0, 1e-12, 1e-10, 1e-6, 1e-3, 1e-1)) {
    r <- .nngp_cores(f$C, f$c_vec, f$w_nb, jitter = jit)
    expect_true(r$templated_ok)
    expect_identical(r$templated_ok, r$plain_ok)
    expect_equal(r$templated_mean, r$plain_mean, tolerance = 0)
    expect_equal(r$templated_var,  r$plain_var,  tolerance = 0)
  }
})

test_that("jitter is a nugget in both cores, not a pivot floor", {
  # A pivot floor is INERT on a well-conditioned input: C is far from singular
  # here, so its pivots all sit above 1e-3 and a floor at that value would
  # leave the answer bit-identical to jitter = 0. A nugget moves it.
  f <- .nngp_fixture()
  base <- .nngp_cores(f$C, f$c_vec, f$w_nb, jitter = 0)
  big  <- .nngp_cores(f$C, f$c_vec, f$w_nb, jitter = 1e-3)
  expect_gt(min(diag(chol(f$C))), 1e-3)     # a floor at 1e-3 would not bind
  expect_false(isTRUE(all.equal(base$plain_mean, big$plain_mean,
                                tolerance = 1e-12)))
  expect_false(isTRUE(all.equal(base$templated_mean, big$templated_mean,
                                tolerance = 1e-12)))

  # And what each core factorizes is C + jitter*I: check against R's own
  # solve of that matrix, which is the definition rather than either core.
  n <- nrow(f$C)
  ref <- solve(f$C + diag(1e-3, n), f$c_vec)
  expect_equal(big$plain_mean, sum(ref * f$w_nb), tolerance = 1e-10)
  expect_equal(big$templated_mean, sum(ref * f$w_nb), tolerance = 1e-10)
  expect_equal(big$plain_var, max(1e-8, 1 - sum(f$c_vec * ref)),
               tolerance = 1e-10)
})

test_that("both cores decline a neighbour covariance that is not PD", {
  # A core that cannot report a non-PD C hands its caller kriged moments off an
  # unusable factor. Rank 1, so any n > 1 is singular.
  n <- 4L
  C <- matrix(1, n, n)
  r <- .nngp_cores(C, rep(0.5, n), stats::rnorm(n), jitter = 0)
  expect_false(r$templated_ok)
  expect_false(r$plain_ok)
  # A nugget large enough restores definiteness in both.
  r2 <- .nngp_cores(C, rep(0.5, n), stats::rnorm(n), jitter = 1e-2)
  expect_true(r2$templated_ok)
  expect_true(r2$plain_ok)
  expect_equal(r2$templated_mean, r2$plain_mean, tolerance = 0)
})
