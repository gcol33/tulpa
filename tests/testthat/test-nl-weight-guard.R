# test-nl-weight-guard.R
# Regression tests for the finite-guarded outer-grid weight normaliser
# (gcol33/tulpa#65). A nested-Laplace outer grid can contain non-finite
# log_marginal cells (an inner Newton diverging in a grid corner returns
# +Inf / NaN). The normaliser must drop those cells and renormalise over the
# finite ones, so tulpa_posterior_draws() / predict() / WAIC always have the
# finite, precision-carrying cells to sample.

test_that(".nl_normalise_weights_safe drops +Inf / NaN cells and renormalises", {
  expected <- {
    w <- exp(c(-1, -2, -3) - max(c(-1, -2, -3)))
    w / sum(w)
  }

  w_inf <- tulpa:::.nl_normalise_weights_safe(c(-1, -2, -3, Inf), "x")
  expect_equal(w_inf, c(expected, 0))
  expect_equal(sum(w_inf), 1)

  w_nan <- tulpa:::.nl_normalise_weights_safe(c(-1, -2, -3, NaN), "x")
  expect_equal(w_nan, c(expected, 0))
  expect_equal(sum(w_nan), 1)

  # -Inf cells already get zero weight under a plain max-shift; the guard must
  # not change that behaviour.
  w_ninf <- tulpa:::.nl_normalise_weights_safe(c(-1, -2, -3, -Inf), "x")
  expect_equal(w_ninf, c(expected, 0))
})

test_that(".nl_normalise_weights_safe returns all-NA (not NaN) when no cell is finite", {
  expect_warning(
    w <- tulpa:::.nl_normalise_weights_safe(c(Inf, NaN, -Inf), "x"),
    "non-finite"
  )
  expect_true(all(is.na(w)))
  expect_false(any(is.nan(w[!is.na(w)])))
})

test_that("logLik on a grid with a non-finite cell drops it instead of returning NaN", {
  fit <- structure(
    list(log_marginal = c(-10, -11, -12, Inf), n_fixed = 2L, N = 100L),
    class = "tulpa_fit"
  )
  ll <- logLik(fit)
  expect_true(is.finite(as.numeric(ll)))

  ref <- structure(
    list(log_marginal = c(-10, -11, -12), n_fixed = 2L, N = 100L),
    class = "tulpa_fit"
  )
  expect_equal(as.numeric(ll), as.numeric(logLik(ref)))
})
