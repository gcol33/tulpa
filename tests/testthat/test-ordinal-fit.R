# Ordinal cumulative-logit / proportional-odds regression via Laplace
# (gcol33/tulpa C7, ordinal half). Validated by recovering the covariate effect
# and cutpoints from simulated proportional-odds data.

test_that("tulpa_ordinal recovers the effect and cutpoints", {
  skip_on_cran()
  set.seed(4)
  n <- 1000L; x <- rnorm(n)
  beta_true <- 0.8; cuts_true <- c(-1, 0.5, 2)     # K = 4 ordered classes
  eta <- beta_true * x
  Fm  <- plogis(outer(-eta, cuts_true, "+"))
  P   <- cbind(Fm, 1) - cbind(0, Fm)
  y   <- ordered(vapply(seq_len(n),
                        function(i) sample.int(4L, 1L, prob = P[i, ]), integer(1)))
  d <- data.frame(y = y, x = x)

  fit <- tulpa_ordinal(y ~ x, data = d)
  expect_s3_class(fit, "tulpa_ordinal")
  expect_true(is.finite(fit$log_marginal))
  expect_true(fit$converged)

  expect_lt(abs(fit$coefficients[["x"]] - beta_true), 0.2)
  expect_lt(max(abs(unname(fit$cutpoints) - cuts_true)), 0.3)
  # Cutpoints must be strictly increasing (ordering constraint).
  expect_true(all(diff(fit$cutpoints) > 0))
})

test_that("tulpa_ordinal: PD vcov, draws, and level guards", {
  skip_on_cran()
  set.seed(6)
  n <- 300L; x <- rnorm(n)
  y <- ordered(sample(1:3, n, replace = TRUE))
  fit <- tulpa_ordinal(y ~ x, data = data.frame(y = y, x = x))

  V <- vcov(fit)
  expect_true(all(eigen(V, symmetric = TRUE, only.values = TRUE)$values > 0))
  expect_equal(ncol(fit$draws), length(fit$coefficients) + length(fit$cutpoints))

  y2 <- ordered(sample(1:2, n, replace = TRUE))
  expect_error(tulpa_ordinal(y2 ~ x, data = data.frame(y2 = y2, x = x)),
               ">= 3 ordered levels")
})
