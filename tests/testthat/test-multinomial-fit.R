# Multinomial (nominal K-class) logistic regression via Laplace (gcol33/tulpa C7).
# The per-observation kernel is FD-validated in test-multinomial-logit.R; here the
# driver that assembles it into the joint Newton + Laplace fit is validated by
# recovering the baseline-category coefficients from simulated data.

test_that("tulpa_multinomial recovers baseline-category coefficients", {
  skip_on_cran()
  set.seed(2)
  n <- 900L; x <- rnorm(n)
  b1 <- c(0.5, 1.0); b2 <- c(-0.3, -0.8)          # classes 1, 2 vs baseline 3
  eta <- cbind(b1[1] + b1[2] * x, b2[1] + b2[2] * x)
  P <- cbind(exp(eta), 1); P <- P / rowSums(P)
  y <- factor(vapply(seq_len(n),
                     function(i) sample.int(3L, 1L, prob = P[i, ]), integer(1)))
  d <- data.frame(y = y, x = x)

  fit <- tulpa_multinomial(y ~ x, data = d)
  expect_s3_class(fit, "tulpa_multinomial")
  expect_true(is.finite(fit$log_marginal))
  expect_true(fit$converged)
  expect_equal(fit$baseline, "3")

  cf <- coef(fit)
  expect_lt(abs(cf[["1:(Intercept)"]] - b1[1]), 0.25)
  expect_lt(abs(cf[["1:x"]]           - b1[2]), 0.25)
  expect_lt(abs(cf[["2:(Intercept)"]] - b2[1]), 0.25)
  expect_lt(abs(cf[["2:x"]]           - b2[2]), 0.25)
})

test_that("tulpa_multinomial: PD vcov, draws, and 2-level rejection", {
  skip_on_cran()
  set.seed(5)
  n <- 250L; x <- rnorm(n)
  y <- factor(sample(letters[1:3], n, replace = TRUE))
  fit <- tulpa_multinomial(y ~ x, data = data.frame(y = y, x = x))

  V <- vcov(fit)
  expect_true(all(eigen(V, symmetric = TRUE, only.values = TRUE)$values > 0))
  expect_equal(ncol(fit$draws), length(coef(fit)))
  expect_equal(length(coef(fit)), 4L)              # (K-1) * (intercept + x)

  y2 <- factor(sample(letters[1:2], n, replace = TRUE))
  expect_error(
    tulpa_multinomial(y2 ~ x, data = data.frame(y2 = y2, x = x)),
    ">= 3 levels")
})
