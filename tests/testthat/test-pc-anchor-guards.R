# The PC calibration lambda = -log(alpha) / U exists only for U > 0 and alpha in
# (0, 1) (gcol33/tulpa#499). Outside that the density is -Inf or NaN at every
# value of the scale, reached inside a gradient with nothing naming the
# parameter. One predicate answers it, and each door reports it in its own
# terms: the R front doors name the argument, the sampler entry names the spec,
# and the templated density -- which runs inside OpenMP regions, where a throw
# is std::terminate -- falls back to a flat prior on sigma so no unvalidated
# path can emit a NaN.

test_that("the front doors reject anchors the calibration cannot represent", {
  expect_error(
    spatial_gp(~ x + y, approx = "hsgp", sigma_prior_alpha = 1),
    "sigma_prior_alpha"
  )
  expect_error(
    spatial_gp(~ x + y, approx = "hsgp", sigma_prior_U = 0),
    "sigma_prior_U"
  )
  expect_error(
    temporal_tvc("year", sigma_prior_alpha = -0.1),
    "sigma_prior_alpha"
  )
  expect_error(
    temporal_tvc("year", sigma_prior_U = -1),
    "sigma_prior_U"
  )
  # The defaults are the anchors the densities used to hardcode.
  s <- spatial_gp(~ x + y, approx = "hsgp")
  expect_equal(s$sigma2_prior_U, 1)
  expect_equal(s$sigma2_prior_alpha, 0.01)
  t <- temporal_tvc("year")
  expect_equal(t$sigma_prior_U, 1)
  expect_equal(t$sigma_prior_alpha, 0.01)
})

test_that("the density falls back to flat rather than -Inf or NaN", {
  # alpha = 1 gives rate 0, so log(rate) is -Inf at every sigma; alpha > 1 gives
  # a negative rate, so log(rate) is NaN and -rate*sigma grows without bound.
  # Both used to reach a gradient as a number.
  for (bad in list(c(1, 1), c(1, 1.5), c(1, 0), c(0, 0.01), c(-1, 0.01))) {
    v <- cpp_pc_prior_scales(sigma = 0.8, U = bad[1], alpha = bad[2])
    expect_true(all(is.finite(v)),
                info = paste("U =", bad[1], "alpha =", bad[2]))
  }
  # A valid pair is untouched: the base density is the calibrated exponential.
  v <- cpp_pc_prior_scales(sigma = 0.8, U = 1, alpha = 0.01)
  lambda <- -log(0.01) / 1
  expect_equal(unname(v[["sigma"]]), log(lambda) - lambda * 0.8)
})

test_that("a scale or precision grid axis must be positive", {
  # A zero cell makes log(tau) -Inf and a negative one makes it NaN, inside the
  # inner Newton solve, as a cell whose marginal is simply not finite.
  set.seed(1L)
  n_obs <- 30L
  X <- cbind(1, rnorm(n_obs))
  y <- rpois(n_obs, 3)
  ti <- rep(1:5, 6L)
  call_temporal <- function(tau_grid) {
    cpp_nested_laplace_temporal(
      y = as.numeric(y), n = rep(1L, n_obs), X = X, re_idx = numeric(0),
      n_re_groups = 0L, sigma_re = 1, temporal_idx = ti, n_times = 5L,
      temporal_type = "rw1", tau_grid = tau_grid, rho_grid = numeric(0),
      cyclic = FALSE, family = "poisson"
    )
  }
  expect_error(call_temporal(c(1, 0, 4)), "positive")
  expect_error(call_temporal(c(1, -2, 4)), "positive")
  expect_no_error(call_temporal(c(1, 2, 4)))
})

test_that("a random-effect scale reaching a logarithm must be positive", {
  set.seed(2L)
  n_obs <- 40L
  X <- cbind(1, rnorm(n_obs))
  y <- rpois(n_obs, 3)
  ti <- rep(1:5, 8L)
  expect_error(
    cpp_nested_laplace_temporal(
      y = as.numeric(y), n = rep(1L, n_obs), X = X,
      re_idx = as.numeric(rep(1:4, each = 10L)), n_re_groups = 4L,
      sigma_re = 0, temporal_idx = ti, n_times = 5L,
      temporal_type = "rw1", tau_grid = c(1, 2), rho_grid = numeric(0),
      cyclic = FALSE, family = "poisson"
    ),
    "sigma_re"
  )
})
