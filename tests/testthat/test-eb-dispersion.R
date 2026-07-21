# Empirical-Bayes estimation of the family dispersion.
#
# The gradient it walks is verified against a finite difference of the
# log-marginal in test-family-dispersion.R; what this file asks is the question
# that matters to a user, and that a gradient check cannot answer: does the
# maximizer land on the phi that generated the data? A gradient can be exact and
# the estimator still biased, if the objective is not the one the model implies.

eb_disp_data <- function(family, phi_true, sigma_true = 0.7,
                         G = 40L, per = 12L, seed = 1L) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- rnorm(G, 0, sigma_true)
  eta <- 0.4 + 0.5 * x + b[g]
  y <- tulpa:::.FAMILY_OPS[[family]]$sample(eta, NULL, phi_true)
  if (family == "gamma") y <- pmax(y, 1e-8)
  list(y = y, X = cbind(`(Intercept)` = 1, x = x),
       re_terms = list(list(idx = g, n_groups = G, n_coefs = 1L)))
}

eb_disp_fit <- function(d, family, phi_start) {
  tulpa_eb(y = d$y, n_trials = NULL, X = d$X, re_terms = d$re_terms,
           family = family, phi = phi_start, estimate_phi = TRUE)
}


test_that("the estimate does not depend on the starting value", {
  skip_on_cran()
  # A maximizer is a property of the objective; if it moves with where the
  # search began, what is being reported is a stopping point.
  d <- eb_disp_data("neg_binomial_2", phi_true = 3.0)
  lo <- eb_disp_fit(d, "neg_binomial_2", 0.5)
  hi <- eb_disp_fit(d, "neg_binomial_2", 20.0)

  expect_true(lo$phi_estimated)
  expect_equal(lo$phi, hi$phi, tolerance = 1e-3)
  expect_equal(lo$map$sigma, hi$map$sigma, tolerance = 1e-3)
  # And the reported log-marginal is the one at the reported phi.
  expect_equal(lo$log_marginal, hi$log_marginal, tolerance = 1e-4)
})


test_that("neg_binomial_2 recovers its dispersion across seeds", {
  skip_on_cran()
  phi_true <- 3.0
  est <- vapply(1:8, function(s) {
    d <- eb_disp_data("neg_binomial_2", phi_true, seed = 100L + s)
    fit <- eb_disp_fit(d, "neg_binomial_2", 1.0)
    c(phi = fit$phi, sigma = fit$map$sigma)
  }, numeric(2))

  # The dispersion of a negative binomial is weakly identified at this sample
  # size, so the bound is on the MEAN over seeds rather than on any single fit.
  expect_lt(abs(mean(est["phi", ]) - phi_true), 0.6)
  # Estimating phi must not corrupt the variance component it shares the
  # objective with.
  expect_lt(abs(mean(est["sigma", ]) - 0.7), 0.12)
})


test_that("gaussian recovers its residual variance across seeds", {
  skip_on_cran()
  # The sharpest of the three: the gaussian dispersion is strongly identified,
  # so a bias here would be the estimator's, not the data's.
  phi_true <- 1.5
  est <- vapply(1:8, function(s) {
    d <- eb_disp_data("gaussian", phi_true, seed = 200L + s)
    fit <- eb_disp_fit(d, "gaussian", 0.5)
    c(phi = fit$phi, sigma = fit$map$sigma)
  }, numeric(2))

  expect_lt(abs(mean(est["phi", ]) - phi_true), 0.1)
  expect_lt(abs(mean(est["sigma", ]) - 0.7), 0.1)
})


test_that("gamma recovers its shape across seeds", {
  skip_on_cran()
  phi_true <- 4.0
  est <- vapply(1:8, function(s) {
    d <- eb_disp_data("gamma", phi_true, seed = 300L + s)
    fit <- eb_disp_fit(d, "gamma", 1.0)
    c(phi = fit$phi, sigma = fit$map$sigma)
  }, numeric(2))

  expect_lt(abs(mean(est["phi", ]) - phi_true) / phi_true, 0.2)
  expect_lt(abs(mean(est["sigma", ]) - 0.7), 0.12)
})


test_that("a fixed-phi fit is unchanged by the new argument", {
  skip_on_cran()
  # estimate_phi = FALSE must be the previous behaviour exactly, not a
  # re-derivation of it: the dispersion coordinate simply is not there.
  d <- eb_disp_data("neg_binomial_2", phi_true = 3.0)
  fixed <- tulpa_eb(y = d$y, n_trials = NULL, X = d$X, re_terms = d$re_terms,
                    family = "neg_binomial_2", phi = 2.0)
  expect_false(isTRUE(fixed$phi_estimated))
  expect_equal(fixed$phi, 2.0)
  # theta_hat keeps the covariance-only length every downstream consumer
  # expects, whether or not a dispersion coordinate was optimized.
  free <- eb_disp_fit(d, "neg_binomial_2", 2.0)
  expect_length(free$theta_hat, length(fixed$theta_hat))
})


test_that("estimating the dispersion beats conditioning on a wrong one", {
  skip_on_cran()
  # The point of the feature: a user who does not know phi should not have to
  # guess. Conditioning on a badly wrong value distorts the variance component;
  # estimating it should not.
  d <- eb_disp_data("neg_binomial_2", phi_true = 3.0, seed = 7L)
  wrong <- tulpa_eb(y = d$y, n_trials = NULL, X = d$X, re_terms = d$re_terms,
                    family = "neg_binomial_2", phi = 0.3)
  free <- eb_disp_fit(d, "neg_binomial_2", 0.3)

  expect_lt(abs(free$phi - 3.0), abs(0.3 - 3.0))
  expect_lt(abs(free$map$sigma - 0.7), abs(wrong$map$sigma - 0.7))
})


test_that("estimate_phi is refused where the derivative is not registered", {
  d <- eb_disp_data("gaussian", 1.0, G = 12L, per = 6L)

  # No free dispersion at all.
  expect_error(
    tulpa_eb(y = pmax(round(d$y), 0), n_trials = NULL, X = d$X,
             re_terms = d$re_terms, family = "poisson", phi = 1.0,
             estimate_phi = TRUE),
    "not available for family")

  # Registered derivatives exist but the assembled gradient is not exact.
  expect_error(
    tulpa_eb(y = plogis(d$y), n_trials = NULL, X = d$X,
             re_terms = d$re_terms, family = "beta", phi = 5.0,
             estimate_phi = TRUE),
    "not available for family")

  # The AGHQ inner marginal is a different objective.
  expect_error(
    tulpa_eb(y = d$y, n_trials = NULL, X = d$X, re_terms = d$re_terms,
             family = "gaussian", phi = 1.0, estimate_phi = TRUE, n_quad = 5L),
    "n_quad = 1")
})
