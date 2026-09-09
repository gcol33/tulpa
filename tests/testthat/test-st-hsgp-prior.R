# The HSGP-ST interaction prior: the cyclic flag its quadratic form reads, and
# the direction its sum pin does not reach.
#
# gcol33/tulpa#696 -- the branch scaled its log-precision by a CYCLIC-aware rank
# while evaluating an ACYCLIC operator beside it, the defect gcol33/tulpa#596
# fixed in st_kronecker_temporal_quad and left standing here.
# gcol33/tulpa#697 -- under a non-cyclic RW2 marginal each basis function's own
# kernel is {1_T, ramp}, and a penalty on the sum reaches the constant only, so
# the ramp direction carried no prior curvature at all.
#
# The discriminator for both is one field: a per-basis-function LINEAR RAMP. It
# is annihilated by the acyclic RW2 operator and sums to zero, so before the fix
# the prior did not move at all when it was added -- under either flag.

M <- 3L
T_st <- 7L
EIG <- c(0.4, 1.1, 2.7)          # spectral eigenvalues, one per basis function
HYP <- list(log_tau_st = 0.3, log_sigma2_hsgp = -0.2, log_lengthscale_hsgp = 0.1)

# Basis-major, matching st_delta[j * T_st + t].
ramp_field <- function(a, M, T_st) {
  v <- seq_len(T_st) - 1 - (T_st - 1) / 2      # centred, so it sums to zero
  as.numeric(rep(a, each = T_st) * rep(v, times = M))
}

lp <- function(delta, temporal, cyclic) {
  do.call(cpp_test_st_hsgp_log_prior,
          c(list(delta = delta, eigenvalues = EIG, T = T_st,
                 temporal = temporal, temporal_cyclic = cyclic), HYP))
}

test_that("a per-basis linear ramp is penalized under a non-cyclic RW2", {
  a <- c(0.7, -1.3, 0.4)
  zero <- numeric(M * T_st)
  d <- lp(ramp_field(a, M, T_st), "rw2", FALSE) - lp(zero, "rw2", FALSE)

  # The ramp has zero acyclic RW2 quadratic form and zero sum, so the whole
  # difference is the trend pin, at the precision the engine's own helper gives.
  v2 <- sum((seq_len(T_st) - 1 - (T_st - 1) / 2)^2)
  expect_equal(d,
               -0.5 * cpp_test_st_trend_precision(T_st) * sum((a * v2)^2),
               tolerance = 1e-10)
  expect_lt(d, 0)
})

test_that("the quadratic form reads the fit's own cyclic flag", {
  a <- c(0.7, -1.3, 0.4)
  zero <- numeric(M * T_st)
  d_acyclic <- lp(ramp_field(a, M, T_st), "rw2", FALSE) - lp(zero, "rw2", FALSE)
  d_cyclic  <- lp(ramp_field(a, M, T_st), "rw2", TRUE)  - lp(zero, "rw2", TRUE)

  # A ramp is not periodic, so the cyclic operator does NOT annihilate it: the
  # branch that hardcoded `false` returned exactly zero here.
  expect_lt(d_cyclic, 0)
  expect_false(isTRUE(all.equal(d_cyclic, d_acyclic)))

  # And what it picks up is a quadratic form: doubling the ramp quadruples it.
  d_cyclic2 <- lp(ramp_field(2 * a, M, T_st), "rw2", TRUE) - lp(zero, "rw2", TRUE)
  expect_equal(d_cyclic2 / d_cyclic, 4, tolerance = 1e-8)
})

test_that("RW1 and cyclic RW2 take no trend pin", {
  # An RW1 kernel is the constants alone, and a cyclic RW2's is too (a ramp is
  # not periodic, so rw2_rank reports T - 1 there). Adding the pin to either
  # would be a second, unstated prior on a direction that already carries one.
  a <- c(0.5, 0.5, 0.5)
  ramp <- ramp_field(a, M, T_st)
  zero <- numeric(M * T_st)

  # Under RW1 the ramp DOES have a quadratic form, so the difference must be
  # exactly that and carry no additional pin term: scaling is quadratic and the
  # ratio to the cyclic-RW2 arm is not the trend-pin constant.
  d_rw1 <- lp(ramp, "rw1", FALSE) - lp(zero, "rw1", FALSE)
  expect_lt(d_rw1, 0)
  expect_equal((lp(ramp_field(2 * a, M, T_st), "rw1", FALSE) -
                  lp(zero, "rw1", FALSE)) / d_rw1, 4, tolerance = 1e-8)
})

test_that("a field the operator and both pins annihilate is free", {
  # A constant per basis function has zero RW1/RW2 quadratic form and zero
  # trend, so only the sum pin reaches it -- which is the term that was already
  # there. This is the negative control for the two above: it must be unchanged
  # by the cyclic flag.
  const <- as.numeric(rep(c(0.3, -0.2, 0.9), each = T_st))
  zero <- numeric(M * T_st)
  d_acyclic <- lp(const, "rw2", FALSE) - lp(zero, "rw2", FALSE)
  d_cyclic  <- lp(const, "rw2", TRUE)  - lp(zero, "rw2", TRUE)
  expect_equal(d_acyclic, d_cyclic, tolerance = 1e-10)

  # and it is exactly the sum pin, at the engine's own precision.
  a <- c(0.3, -0.2, 0.9)
  kappa <- 0.001
  expect_equal(d_acyclic,
               -0.5 * tulpa:::cpp_test_s2z_precision(T_st, kappa) *
                 sum((a * T_st)^2),
               tolerance = 1e-10)
})
