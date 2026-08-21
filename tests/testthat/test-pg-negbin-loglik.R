# The negative-binomial density the Gibbs kernels evaluate, held against R's
# own dnbinom over a range of eta the sampler is not confined to. mu = r exp(eta),
# so prob = r / (r + mu) = 1 / (1 + exp(eta)) and r cancels out of the
# parameterisation; what is left is exact at every eta.

nb_reference <- function(y, eta, r) {
  sum(stats::dnbinom(y, size = r, mu = r * exp(eta), log = TRUE))
}

test_that("negbin_loglik_eta matches dnbinom on the range the sampler visits", {
  y <- c(0L, 1L, 3L, 7L, 40L)
  for (r in c(0.5, 1, 12.5)) {
    for (e in c(-3, -1, 0, 1, 3)) {
      eta <- rep(e, length(y))
      expect_equal(
        cpp_test_negbin_loglik_eta(y, eta, r),
        nb_reference(y, eta, r),
        tolerance = 1e-12
      )
    }
  }
})

test_that("it stays exact where mu underflows and where it overflows", {
  # dnbinom is itself unusable once mu = r exp(eta) overflows, so the far tail
  # is scored against the closed form the kernel claims rather than against R.
  y <- c(0L, 2L, 11L)
  r <- 2.5
  closed_form <- function(y, eta, r) {
    sum(lgamma(y + r) - lgamma(y + 1) - lgamma(r) +
          y * eta - (y + r) * log1p(exp(-abs(eta))) -
          (y + r) * pmax(eta, 0))
  }
  for (e in c(-800, -200, -40, -20, 20, 40, 200, 800)) {
    eta <- rep(e, length(y))
    val <- cpp_test_negbin_loglik_eta(y, eta, r)
    expect_true(is.finite(val))
    expect_equal(val, closed_form(y, eta, r), tolerance = 1e-10)
  }
})

test_that("the two references agree wherever dnbinom is itself finite", {
  y <- c(0L, 5L)
  r <- 3
  eta <- c(-15, 15)
  ref <- nb_reference(y, eta, r)
  skip_if_not(is.finite(ref))
  expect_equal(cpp_test_negbin_loglik_eta(y, eta, r), ref, tolerance = 1e-10)
})

test_that("a length mismatch is refused", {
  expect_error(cpp_test_negbin_loglik_eta(c(1L, 2L), c(0.1), 1.0),
               "eta holds")
})
