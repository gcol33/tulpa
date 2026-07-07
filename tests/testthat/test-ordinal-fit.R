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

test_that("cumulative probit recovers probit-generated data (gcol33/tulpa C6)", {
  skip_on_cran()
  skip_if_not_installed("MASS")
  set.seed(8)
  n <- 1200L; x <- rnorm(n)
  beta_true <- 0.7; cuts_true <- c(-0.8, 0.4, 1.5)
  eta <- beta_true * x
  Fm  <- pnorm(outer(-eta, cuts_true, "+"))
  P   <- cbind(Fm, 1) - cbind(0, Fm)
  y   <- ordered(vapply(seq_len(n),
                        function(i) sample.int(4L, 1L, prob = P[i, ]), integer(1)))
  d <- data.frame(y = y, x = x)

  fit <- tulpa_ordinal(y ~ x, data = d, link = "probit")
  expect_equal(fit$link, "probit")
  expect_equal(fit$family, "ordinal_probit")
  expect_lt(abs(fit$coefficients[["x"]] - beta_true), 0.15)
  expect_lt(max(abs(unname(fit$cutpoints) - cuts_true)), 0.25)

  # Matches the MASS::polr probit MLE (weak ridge, large n).
  pol <- MASS::polr(y ~ x, data = d, method = "probit")
  expect_lt(abs(fit$coefficients[["x"]] - unname(coef(pol))), 0.02)
  expect_lt(max(abs(unname(fit$cutpoints) - unname(pol$zeta))), 0.05)

  # The logit fit of the same data lands elsewhere (different latent scale).
  fit_l <- tulpa_ordinal(y ~ x, data = d, link = "logit")
  expect_gt(abs(fit_l$coefficients[["x"]] - fit$coefficients[["x"]]), 0.2)
})

test_that("categorical families route through the tulpa() front door", {
  skip_on_cran()
  set.seed(9)
  n <- 400L; x <- rnorm(n)

  # Ordinal (logit + probit suffix).
  cuts <- c(-1, 0.5, 2); eta <- 0.8 * x
  Fm <- plogis(outer(-eta, cuts, "+")); P <- cbind(Fm, 1) - cbind(0, Fm)
  yo <- ordered(vapply(seq_len(n),
                       function(i) sample.int(4L, 1L, prob = P[i, ]), integer(1)))
  d <- data.frame(y = yo, x = x)
  f1 <- tulpa(y ~ x, data = d, family = "ordinal")
  f2 <- tulpa_ordinal(y ~ x, data = d)
  expect_s3_class(f1, "tulpa_ordinal")
  expect_equal(coef(f1), coef(f2))
  f3 <- tulpa(y ~ x, data = d, family = "ordinal_probit")
  expect_equal(f3$link, "probit")

  # Multinomial.
  eta2 <- cbind(0.5 + 1.0 * x, -0.3 - 0.8 * x)
  P2 <- cbind(exp(eta2), 1); P2 <- P2 / rowSums(P2)
  ym <- factor(vapply(seq_len(n),
                      function(i) sample.int(3L, 1L, prob = P2[i, ]), integer(1)))
  dm <- data.frame(y = ym, x = x)
  fm  <- tulpa(y ~ x, data = dm, family = "multinomial")
  fm2 <- tulpa_multinomial(y ~ x, data = dm)
  expect_s3_class(fm, "tulpa_multinomial")
  expect_equal(coef(fm), coef(fm2))

  # Guards: structure and non-Laplace modes refuse.
  d$g <- factor(rep(1:4, 100))
  expect_error(tulpa(y ~ x + (1 | g), data = d, family = "ordinal"),
               "fixed-effect models only")
  expect_error(tulpa(y ~ x, data = d, family = "ordinal", mode = "mala"),
               "not\\s+available")
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
