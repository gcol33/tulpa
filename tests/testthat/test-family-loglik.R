# test-family-loglik.R
# Correctness of the family-ops single source of truth: log-likelihoods must
# agree with R's reference densities, scores must match finite differences of
# the log-likelihood, and the refactored glmm_weights() must be unchanged.

test_that("log-likelihoods match R reference densities", {
  set.seed(1)
  eta <- seq(-2, 2, length.out = 7)

  # binomial
  n <- rep(10L, length(eta)); y <- rbinom(length(eta), n, plogis(eta))
  mu <- plogis(eta)
  expect_equal(family_loglik(eta, y, "binomial", n_trials = n),
               dbinom(y, n, mu, log = TRUE), tolerance = 1e-10)

  # poisson
  yp <- rpois(length(eta), exp(eta))
  expect_equal(family_loglik(eta, yp, "poisson"),
               dpois(yp, exp(eta), log = TRUE), tolerance = 1e-10)

  # neg_binomial_2 (phi = size)
  phi <- 2.5; ynb <- rnbinom(length(eta), size = phi, mu = exp(eta))
  expect_equal(family_loglik(eta, ynb, "neg_binomial_2", phi = phi),
               dnbinom(ynb, size = phi, mu = exp(eta), log = TRUE),
               tolerance = 1e-10)

  # gaussian (phi = residual variance)
  s2 <- 0.7; yg <- rnorm(length(eta), eta, sqrt(s2))
  expect_equal(family_loglik(eta, yg, "gaussian", phi = s2),
               dnorm(yg, eta, sqrt(s2), log = TRUE), tolerance = 1e-10)

  # beta (phi = precision)
  prec <- 8; mub <- plogis(eta); yb <- rbeta(length(eta), mub * prec, (1 - mub) * prec)
  expect_equal(family_loglik(eta, yb, "beta", phi = prec),
               dbeta(yb, mub * prec, (1 - mub) * prec, log = TRUE),
               tolerance = 1e-10)
})

test_that("score equals the finite-difference gradient of the log-likelihood", {
  h <- 1e-6
  fd_score <- function(eta, y, family, n_trials = NULL, phi = 1) {
    vapply(seq_along(eta), function(i) {
      ep <- eta; em <- eta; ep[i] <- ep[i] + h; em[i] <- em[i] - h
      (family_loglik(ep, y, family, n_trials, phi)[i] -
         family_loglik(em, y, family, n_trials, phi)[i]) / (2 * h)
    }, numeric(1))
  }
  eta <- c(-1.3, -0.4, 0.2, 0.9, 1.6)

  n <- rep(12L, length(eta)); y <- c(2, 4, 6, 9, 11)
  expect_equal(family_score_eta(eta, y, "binomial", n_trials = n),
               fd_score(eta, y, "binomial", n), tolerance = 1e-5)

  yp <- c(0, 1, 2, 3, 6)
  expect_equal(family_score_eta(eta, yp, "poisson"),
               fd_score(eta, yp, "poisson"), tolerance = 1e-5)

  expect_equal(family_score_eta(eta, yp, "neg_binomial_2", phi = 3),
               fd_score(eta, yp, "neg_binomial_2", phi = 3), tolerance = 1e-5)

  yg <- c(-1.1, -0.5, 0.3, 1.0, 1.5)
  expect_equal(family_score_eta(eta, yg, "gaussian", phi = 0.5),
               fd_score(eta, yg, "gaussian", phi = 0.5), tolerance = 1e-5)

  yb <- c(0.15, 0.35, 0.55, 0.72, 0.85)
  expect_equal(family_score_eta(eta, yb, "beta", phi = 6),
               fd_score(eta, yb, "beta", phi = 6), tolerance = 1e-5)
})

test_that("glmm_weights is unchanged by the registry refactor", {
  eta <- seq(-2, 2, length.out = 9)
  n <- rep(7L, length(eta))
  # reference: the historical inline formulas
  ref_binom <- { mu <- pmin(pmax(plogis(eta), 1e-8), 1 - 1e-8); n * mu * (1 - mu) }
  ref_pois  <- pmax(exp(eta), 1e-8)
  ref_nb    <- { mu <- pmax(exp(eta), 1e-8); mu * 2 / (mu + 2) }
  ref_beta  <- {
    mu <- pmin(pmax(plogis(eta), 1e-7), 1 - 1e-7); dmu <- mu * (1 - mu)
    tg <- trigamma(mu * 5) + trigamma((1 - mu) * 5); 5 * 5 * tg * dmu * dmu
  }
  expect_equal(glmm_weights(eta, "binomial", n_trials = n), ref_binom, tolerance = 1e-12)
  expect_equal(glmm_weights(eta, "poisson"), ref_pois, tolerance = 1e-12)
  expect_equal(glmm_weights(eta, "neg_binomial_2", phi = 2), ref_nb, tolerance = 1e-12)
  expect_equal(glmm_weights(eta, "gaussian"), rep(1, length(eta)))
  expect_equal(glmm_weights(eta, "beta", phi = 5), ref_beta, tolerance = 1e-12)
})

test_that("family helpers validate names; glmm_weights stays permissive", {
  expect_error(family_loglik(0, 0, "weibull"), "Unknown family")
  expect_error(family_mean(0, "nope"), "Unknown family")
  expect_setequal(family_names(),
                  c("binomial", "poisson", "neg_binomial_2", "gaussian", "beta"))
  # historical fallback: unknown family -> unit weights, no error
  expect_equal(glmm_weights(c(0, 1, 2), "weibull"), rep(1, 3))
})

test_that("family_mean applies the documented clamps", {
  expect_true(all(family_mean(c(-50, 50), "binomial") > 0 &
                    family_mean(c(-50, 50), "binomial") < 1))
  expect_true(all(family_mean(c(-50, 50), "poisson") >= 1e-8))
})
