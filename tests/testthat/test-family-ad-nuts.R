# Analytic AD-gradient coverage for gamma / inverse_gaussian / beta under the
# ModelData NUTS kernel (gcol33/tulpa C8). Before this, these families fell back
# to the numerical gradient; builtin_family_ll_ad now supplies the templated
# closed form. A wrong analytic gradient samples the wrong posterior, so
# recovering the GLM MLE / simulation truth validates the gradient (the engine
# also verifies the active gradient against numerical before NUTS). lognormal's
# AD form ships too but its sampler-registry entry is deferred (its phi = SD
# convention collides with the load-bearing gaussian phi = variance entry).

hmc_ctrl <- list(n_iter = 1500L, warmup = 750L, n_chains = 2L, seed = 7L,
                 adapt_delta = 0.9)

test_that("NUTS recovers gamma(log) fixed effects (analytic AD gradient)", {
  skip_if_not_slow()
  set.seed(1)
  n <- 400L; x <- rnorm(n)
  eta <- 0.6 + 0.4 * x
  shape <- 3.0
  y <- rgamma(n, shape = shape, rate = shape / exp(eta))   # mean exp(eta)
  mle <- unname(coef(glm(y ~ x, family = Gamma(link = "log"))))

  fit <- tulpa_sample_glmm(y, NULL, cbind(1, x), family = "gamma",
                           backend = "hmc", phi = shape, control = hmc_ctrl)
  bh <- unname(fit$means[1:2])
  expect_lt(max(abs(bh - mle)), 0.08)
})

test_that("NUTS recovers inverse_gaussian(log) fixed effects (analytic AD gradient)", {
  skip_if_not_slow()
  # Michael-Schucany-Haas generator (no statmod dependency); lambda = 1/phi.
  rinvgauss <- function(n, mu, lambda) {
    nu <- rnorm(n)^2
    xj <- mu + (mu^2 * nu) / (2 * lambda) -
      (mu / (2 * lambda)) * sqrt(4 * mu * lambda * nu + mu^2 * nu^2)
    z <- runif(n)
    ifelse(z <= mu / (mu + xj), xj, mu^2 / xj)
  }
  set.seed(3)
  n <- 500L; x <- rnorm(n)
  eta <- 0.5 + 0.3 * x
  phi <- 0.5                       # engine variance = phi * mu^3 => lambda = 1/phi
  y <- rinvgauss(n, mu = exp(eta), lambda = 1 / phi)
  mle <- unname(coef(glm(y ~ x, family = inverse.gaussian(link = "log"))))

  fit <- tulpa_sample_glmm(y, NULL, cbind(1, x), family = "inverse_gaussian",
                           backend = "hmc", phi = phi, control = hmc_ctrl)
  bh <- unname(fit$means[1:2])
  expect_lt(max(abs(bh - mle)), 0.1)
})

test_that("NUTS recovers beta(logit) fixed effects (analytic AD gradient)", {
  skip_if_not_slow()
  set.seed(4)
  n <- 500L; x <- rnorm(n)
  b0 <- 0.2; b1 <- 0.7
  mu <- plogis(b0 + b1 * x)
  phi <- 8.0
  y <- rbeta(n, mu * phi, (1 - mu) * phi)

  fit <- tulpa_sample_glmm(y, NULL, cbind(1, x), family = "beta",
                           backend = "hmc", phi = phi, control = hmc_ctrl)
  bh <- unname(fit$means[1:2])
  # No base-R beta GLM MLE; recover the simulation truth.
  expect_lt(max(abs(bh - c(b0, b1))), 0.12)
})

test_that("NUTS recovers beta_binomial(logit) fixed effects (analytic AD gradient)", {
  skip_if_not_slow()
  set.seed(5)
  n <- 500L; x <- rnorm(n)
  b0 <- 0.1; b1 <- 0.6
  mu  <- plogis(b0 + b1 * x)
  phi <- 10.0                       # precision a + b
  size <- 20L                       # binomial trials per row
  p <- rbeta(n, mu * phi, (1 - mu) * phi)
  y <- rbinom(n, size, p)

  fit <- tulpa_sample_glmm(y, rep(size, n), cbind(1, x), family = "beta_binomial",
                           backend = "hmc", phi = phi, control = hmc_ctrl)
  bh <- unname(fit$means[1:2])
  # Overdispersed binomial: recover the simulation truth on the logit scale.
  expect_lt(max(abs(bh - c(b0, b1))), 0.1)
})

test_that("NUTS recovers Student-t location under heavy-tailed noise (analytic AD gradient)", {
  skip_if_not_slow()
  set.seed(6)
  n <- 500L; x <- rnorm(n)
  b0 <- 1.0; b1 <- -0.5; scale <- 0.8
  y <- b0 + b1 * x + scale * rt(n, df = 4)     # heavy-tailed (df = kStudentTDf)
  fit <- tulpa_sample_glmm(y, NULL, cbind(1, x), family = "t",
                           backend = "hmc", phi = scale, control = hmc_ctrl)
  bh <- unname(fit$means[1:2])
  expect_lt(max(abs(bh - c(b0, b1))), 0.12)
})
