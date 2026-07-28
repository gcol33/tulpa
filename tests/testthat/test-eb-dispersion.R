# Empirical-Bayes estimation of the family dispersion.
#
# The gradient it walks is verified against a finite difference of the
# log-marginal in test-family-dispersion.R; what this file asks is the question
# that matters to a user, and that a gradient check cannot answer: does the
# maximizer land on the phi that generated the data? A gradient can be exact and
# the estimator still biased, if the objective is not the one the model implies.

eb_disp_data <- function(family, phi_true, sigma_true = 0.7,
                         G = 40L, per = 12L, seed = 1L, n_trials = NULL,
                         loc = 0.4) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- rnorm(G, 0, sigma_true)
  eta <- loc + 0.5 * x + b[g]
  nt <- if (is.null(n_trials)) NULL else rep(n_trials, n)
  y <- tulpa:::.FAMILY_OPS[[family]]$sample(eta, nt, phi_true)
  if (family %in% c("gamma", "lognormal", "inverse_gaussian"))
    y <- pmax(y, 1e-8)
  if (family == "beta") y <- pmin(pmax(y, 1e-6), 1 - 1e-6)
  list(y = y, X = cbind(`(Intercept)` = 1, x = x), n_trials = nt,
       re_terms = list(list(idx = g, n_groups = G, n_coefs = 1L)))
}

# No `phi2`: tulpa_eb() has no such argument and the nested path hard-codes
# NA_real_ (gcol33/tulpa#257), so a `t` fit runs at the default df. The fixture
# above draws with the same default, which keeps the two sides describing one
# model -- and is why `t` is exercised on its scale only.
eb_disp_fit <- function(d, family, phi_start) {
  tulpa_eb(y = d$y, n_trials = d$n_trials, X = d$X, re_terms = d$re_terms,
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


test_that("every newly registered family recovers its dispersion", {
  skip_on_cran()
  # An exact gradient says the optimizer walks the right surface; it does not
  # say the maximizer of that surface is the generating phi. These seven
  # families reached the registry on gradient evidence alone, so each gets the
  # question the gradient cannot answer.
  #
  # Tolerances are the measured bias over the seeds below, roughly doubled: the
  # tightest is `t` (1%) and the loosest beta_binomial (14% spread across
  # seeds), which is the precision of the estimator on this design rather than
  # anything about the derivative.
  cases <- list(
    list(f = "lognormal",        phi = 1.5, start = 0.5, loc = 0.4, tol = 0.10),
    list(f = "neg_binomial_1",   phi = 2.0, start = 0.8, loc = 1.2, tol = 0.15),
    list(f = "beta",             phi = 8.0, start = 3.0, loc = 0.0, tol = 0.10),
    list(f = "inverse_gaussian", phi = 0.5, start = 1.5, loc = 0.4, tol = 0.10),
    list(f = "beta_binomial",    phi = 8.0, start = 3.0, loc = 0.0, tol = 0.15,
         nt = 10L),
    list(f = "t",               phi = 1.2, start = 0.5, loc = 0.4, tol = 0.08),
    list(f = "truncated_neg_binomial_2",
         phi = 3.0, start = 1.0, loc = 1.2, tol = 0.10)
  )
  for (cs in cases) {
    est <- vapply(1:5, function(s) {
      d <- eb_disp_data(cs$f, cs$phi, seed = 400L + s, loc = cs$loc,
                        n_trials = cs$nt)
      fit <- eb_disp_fit(d, cs$f, cs$start)
      c(phi = fit$phi, sigma = fit$map$sigma)
    }, numeric(2))
    expect_lt(abs(mean(est["phi", ]) - cs$phi) / cs$phi, cs$tol,
              label = paste0(cs$f, " dispersion"))
    expect_lt(abs(mean(est["sigma", ]) - 0.7), 0.12,
              label = paste0(cs$f, " sigma"))
  }
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

  expect_error(
    tulpa_eb(y = round(pmax(d$y, 0)) > 0, n_trials = NULL, X = d$X,
             re_terms = d$re_terms, family = "binomial", phi = 1.0,
             estimate_phi = TRUE),
    "not available for family")

  # beta was refused while its assembled gradient was inexact -- the mode motion
  # was solved on the working-weight inverse. With its observed curvature
  # registered the gradient is exact and the estimate runs.
  expect_no_error(
    tulpa_eb(y = plogis(d$y), n_trials = NULL, X = d$X,
             re_terms = d$re_terms, family = "beta", phi = 5.0,
             estimate_phi = TRUE))

  # The AGHQ inner marginal is a different objective.
  expect_error(
    tulpa_eb(y = d$y, n_trials = NULL, X = d$X, re_terms = d$re_terms,
             family = "gaussian", phi = 1.0, estimate_phi = TRUE, n_quad = 5L),
    "n_quad = 1")
})
