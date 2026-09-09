# Every per-observation numeric the kernels index passes the finite guard, not
# only the design and the response (gcol33/tulpa#665).
#
# tulpa() checked X and y alone, and tulpa_laplace() -- a front door in its own
# right, which does not go through .validate_glm_design() -- checked neither the
# denominators nor the offset. An NA in either reached the C++ kernel and came
# back as an all-zero coefficient vector with no error and no warning, which is
# the shape gcol33/tulpa#613 closed on the nested door only.

set.seed(1)
n <- 120
x <- rnorm(n)
yb <- rbinom(n, 1, plogis(0.5 * x))
yp <- rpois(n, exp(0.3 + 0.4 * x))
d <- data.frame(x = x, yb = yb, yp = yp, off = log(runif(n, 1, 3)))
X <- cbind(`(Intercept)` = 1, x = x)

test_that("tulpa() rejects a non-finite n_trials and offset", {
  nt <- rep(1L, n); nt[5] <- NA_integer_
  expect_error(
    tulpa(yb ~ x, data = d, family = "binomial", n_trials = nt, mode = "laplace"),
    "n_trials"
  )

  d2 <- d; d2$off[3] <- NA_real_
  expect_error(
    tulpa(yp ~ x + offset(off), data = d2, family = "poisson", mode = "laplace"),
    "offset"
  )
})

test_that("tulpa_laplace() rejects them too", {
  nt <- rep(1L, n); nt[7] <- NA_integer_
  expect_error(
    tulpa_laplace(y = yb, n_trials = nt, X = X, family = "binomial"),
    "n_trials"
  )

  off <- d$off; off[11] <- Inf
  expect_error(
    tulpa_laplace(y = yp, n_trials = NULL, X = X, family = "poisson",
                  offset = off),
    "offset"
  )
})

test_that("the guard names the offending row and lets clean input through", {
  nt <- rep(1L, n); nt[42] <- NaN
  expect_error(
    tulpa_laplace(y = yb, n_trials = nt, X = X, family = "binomial"),
    "row 42"
  )
  fit <- tulpa_laplace(y = yb, n_trials = rep(1L, n), X = X,
                       family = "binomial", offset = d$off)
  expect_true(all(is.finite(fit$beta)))
})
