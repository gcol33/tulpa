# Count-family likelihoods against external MLE implementations.
#
# Two claims per family. The first is that the fixed-effect mode at the
# reference's own dispersion reproduces the reference's coefficients, which
# pins down the log-density's dependence on eta. The second is that the profile
# log-marginal peaks at the reference's dispersion estimate, which pins down its
# dependence on phi -- a term the first claim holds fixed and so cannot see.

test_that("poisson reproduces glm()", {
  skip_on_cran()
  set.seed(101)
  N <- 400L
  d <- data.frame(x1 = rnorm(N), x2 = rnorm(N))
  d$y <- rpois(N, exp(0.5 + 0.3 * d$x1 - 0.4 * d$x2))

  ref <- stats::glm(y ~ x1 + x2, data = d, family = stats::poisson())
  co  <- summary(ref)$coefficients

  fit <- suppressMessages(
    tulpa(y ~ x1 + x2, data = d, family = "poisson", mode = "laplace",
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), co[, "Estimate"], co[, "Std. Error"],
                         label = "poisson")
})


test_that("neg_binomial_2 reproduces MASS::glm.nb", {
  skip_on_cran()
  skip_if_not_installed("MASS")
  set.seed(99)
  N <- 400L
  d <- data.frame(x1 = rnorm(N), x2 = rnorm(N))
  d$y <- MASS::rnegbin(N, mu = exp(0.5 + 0.3 * d$x1 - 0.4 * d$x2), theta = 2)

  ref <- MASS::glm.nb(y ~ x1 + x2, data = d)
  co  <- summary(ref)$coefficients

  # tulpa's `phi` for neg_binomial_2 is the size parameter, which is MASS's
  # `theta` on the same scale.
  fit <- suppressMessages(
    tulpa(y ~ x1 + x2, data = d, family = "neg_binomial_2", mode = "laplace",
          phi = ref$theta, beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), co[, "Estimate"], co[, "Std. Error"],
                         label = "neg_binomial_2")

  grid <- seq(1, 4, by = 0.05)
  expect_lt(abs(ref_profile_phi(y ~ x1 + x2, d, "neg_binomial_2", grid,
                                beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)) -
                  ref$theta), 0.1)
})


test_that("neg_binomial_1 reproduces glmmTMB::nbinom1()", {
  skip_on_cran()
  skip_if_not_installed("glmmTMB")
  set.seed(7)
  N <- 400L
  x <- rnorm(N)
  # NB1: Var(Y) = mu * (1 + phi), so size = mu / phi and prob = 1 / (1 + phi).
  phi_true <- 1.0
  mu <- exp(1.0 + 0.6 * x)
  d <- data.frame(y = rnbinom(N, size = mu / phi_true, prob = 1 / (1 + phi_true)),
                  x = x)

  ref <- glmmTMB::glmmTMB(y ~ x, data = d, family = glmmTMB::nbinom1())
  co  <- summary(ref)$coefficients$cond
  # glmmTMB carries the dispersion on the log scale in its `disp` block.
  theta <- exp(glmmTMB::fixef(ref)$disp)

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "neg_binomial_1", mode = "laplace",
          phi = theta, beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), co[, "Estimate"], co[, "Std. Error"],
                         label = "neg_binomial_1")

  grid <- seq(0.3, 2, by = 0.025)
  expect_lt(abs(ref_profile_phi(y ~ x, d, "neg_binomial_1", grid,
                                beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)) -
                  theta), 0.05)
})


test_that("truncated_poisson reproduces glmmTMB::truncated_poisson()", {
  skip_on_cran()
  skip_if_not_installed("glmmTMB")
  set.seed(47)
  N <- 600L
  x <- rnorm(N)
  d <- data.frame(y = ref_sim_truncated_poisson(exp(0.7 + 0.6 * x)), x = x)
  expect_true(all(d$y >= 1L))

  ref <- glmmTMB::glmmTMB(y ~ x, data = d,
                          family = glmmTMB::truncated_poisson())
  co  <- summary(ref)$coefficients$cond

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "truncated_poisson", mode = "laplace",
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), co[, "Estimate"], co[, "Std. Error"],
                         label = "truncated_poisson")
})


test_that("truncated_neg_binomial_2 reproduces glmmTMB::truncated_nbinom2()", {
  skip_on_cran()
  skip_if_not_installed("glmmTMB")
  set.seed(43)
  N <- 500L
  x <- rnorm(N)
  d <- data.frame(y = ref_sim_truncated_nb2(exp(0.5 + 0.4 * x), 2), x = x)
  expect_true(all(d$y >= 1L))

  ref   <- glmmTMB::glmmTMB(y ~ x, data = d,
                            family = glmmTMB::truncated_nbinom2())
  co    <- summary(ref)$coefficients$cond
  theta <- exp(glmmTMB::fixef(ref)$disp)

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "truncated_neg_binomial_2",
          mode = "laplace", phi = theta,
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), co[, "Estimate"], co[, "Std. Error"],
                         label = "truncated_neg_binomial_2")

  grid <- seq(1, 4, by = 0.05)
  expect_lt(abs(ref_profile_phi(y ~ x, d, "truncated_neg_binomial_2", grid,
                                beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)) -
                  theta), 0.1)
})
