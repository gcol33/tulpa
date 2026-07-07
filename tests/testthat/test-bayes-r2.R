# bayes_R2() (gcol33/tulpa C13): per-draw Var(fit)/(Var(fit)+Var(res)) with
# the family's model-based residual variance. Validated against the known
# population R^2 of a simulated gaussian regression and structurally across
# families.

test_that("family_variance matches the family's known response variance", {
  eta <- c(-0.5, 0, 1)

  expect_equal(tulpa:::family_variance(eta, "poisson"), exp(eta))
  expect_equal(tulpa:::family_variance(eta, "gaussian", phi = 3), rep(3, 3))
  mu <- exp(eta)
  expect_equal(tulpa:::family_variance(eta, "neg_binomial_2", phi = 2),
               mu + mu^2 / 2)
  expect_equal(tulpa:::family_variance(eta, "gamma", phi = 4), mu^2 / 4)
  expect_equal(tulpa:::family_variance(eta, "inverse_gaussian", phi = 0.5),
               0.5 * mu^3)
  p <- plogis(eta)
  expect_equal(tulpa:::family_variance(eta, "binomial", n_trials = rep(7, 3)),
               7 * p * (1 - p), tolerance = 1e-6)
  expect_equal(tulpa:::family_variance(eta, "beta", phi = 9),
               p * (1 - p) / 10, tolerance = 1e-6)
  expect_equal(tulpa:::family_variance(eta, "t", phi = 1.5),
               rep(1.5^2 * 2, 3))

  # Response mean is trial-scaled for the binomial families only.
  expect_equal(tulpa:::family_response_mean(eta, "binomial",
                                            n_trials = rep(5, 3)),
               5 * p, tolerance = 1e-6)
  expect_equal(tulpa:::family_response_mean(eta, "poisson"), mu)
})

test_that("bayes_R2 recovers the population R^2 of a gaussian regression", {
  skip_on_cran()
  set.seed(31)
  n <- 500L
  d <- data.frame(x = rnorm(n))
  b <- 1.5; s2 <- 1
  d$y <- rnorm(n, b * d$x, sqrt(s2))
  # Population R^2 = b^2 Var(x) / (b^2 Var(x) + s2) = 2.25 / 3.25 ~ 0.692.
  r2_true <- b^2 / (b^2 + s2)

  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi = s2)
  tab <- bayes_R2(fit, ndraws = 500L, seed = 1)

  expect_s3_class(tab, "data.frame")
  expect_equal(nrow(tab), 1L)
  expect_equal(tab$estimate, r2_true, tolerance = 0.05)
  expect_true(tab$std.error > 0)
  expect_true(tab$`2.5%` < tab$estimate && tab$estimate < tab$`97.5%`)

  draws <- bayes_R2(fit, ndraws = 200L, summary = FALSE, seed = 2)
  expect_length(draws, 200L)
  expect_true(all(draws > 0 & draws < 1))
})

test_that("bayes_R2 is near zero for a null model and works for poisson", {
  skip_on_cran()
  set.seed(33)
  d <- data.frame(x = rnorm(300))
  d$y <- rpois(300, exp(0.5 + 0.8 * d$x))

  fit  <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  null <- tulpa(y ~ 1, data = d, family = "poisson", mode = "laplace")

  r2_fit  <- bayes_R2(fit, ndraws = 300L, seed = 3)$estimate
  r2_null <- bayes_R2(null, ndraws = 300L, seed = 3)$estimate
  expect_gt(r2_fit, 0.3)
  expect_lt(r2_null, 0.05)
})
