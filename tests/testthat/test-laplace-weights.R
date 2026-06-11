# Per-observation likelihood weights in tulpa_laplace() (gcol33/tulpa#108).
#
# The Newton line search backtracks on the SAME weighted objective its step
# direction is built from (the weighted score / Fisher Hessian), so a weighted
# fit converges to the weighted MLE -- matching lm()/glm() with case weights --
# and an unweighted fit is byte-unchanged (the weight is a no-op at w == 1).
# Before the fix the search judged the weighted step against the UNWEIGHTED
# log-likelihood and stalled, returning a mode shrunk toward the prior.

test_that("weighted gaussian tulpa_laplace matches lm(weights=)", {
  set.seed(1); n <- 400L
  x <- rnorm(n); grp <- rbinom(n, 1L, 0.4)
  # Two groups on different lines, so down-weighting must actually bite.
  y <- ifelse(grp == 1L, 1.2, -1.2) + 0.5 * x + rnorm(n, 0, 0.8)
  w <- as.numeric(grp)
  fit <- tulpa_laplace(y, rep(1L, n), cbind(1, x), family = "gaussian",
                       phi = 0.8, weights = w)
  expect_true(fit$converged)
  expect_equal(unname(fit$mode), unname(coef(lm(y ~ x, weights = w))),
               tolerance = 1e-5)
})

test_that("weighted binomial tulpa_laplace matches glm(weights=)", {
  set.seed(2); n <- 400L
  x <- rnorm(n); w <- as.numeric(rbinom(n, 1L, 0.4))
  y <- rbinom(n, 1L, plogis(0.3 + 0.6 * x))
  fit <- tulpa_laplace(y, rep(1L, n), cbind(1, x), family = "binomial",
                       weights = w)
  expect_true(fit$converged)
  expect_equal(unname(fit$mode),
               unname(coef(glm(y ~ x, family = binomial, weights = w))),
               tolerance = 1e-4)
})

test_that("an all-ones weight vector is a no-op", {
  set.seed(3); n <- 300L
  x <- rnorm(n); y <- 1.0 + 0.5 * x + rnorm(n, 0, 0.7)
  X <- cbind(1, x)
  a <- tulpa_laplace(y, rep(1L, n), X, family = "gaussian", phi = 0.7)
  b <- tulpa_laplace(y, rep(1L, n), X, family = "gaussian", phi = 0.7,
                     weights = rep(1, n))
  expect_equal(a$mode, b$mode, tolerance = 1e-10)
  expect_equal(a$log_marginal, b$log_marginal, tolerance = 1e-10)
})
