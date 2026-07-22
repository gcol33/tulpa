# Empirical-Bayes random-effect covariances (mode = "eb", tulpa_eb()).
#
# The estimator is the mode of the same outer objective tulpa_re_cov_nested()
# integrates, so the tests come in three groups: that the shared objective
# really is shared (identical theta_hat), that the estimator recovers known
# variance components and fixed effects, and that the registry wiring reaches it
# without auto ever selecting it.

# --- simulation helpers ------------------------------------------------------

sim_re_pois <- function(seed, G = 40L, per = 10L, sigma = 0.7,
                        beta = c(0.4, 0.6)) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- rnorm(G, 0, sigma)
  eta <- beta[1L] + beta[2L] * x + b[grp]
  list(y = rpois(n, exp(eta)), x = x, grp = grp, G = G,
       X = cbind(1, x), sigma = sigma, beta = beta,
       data = data.frame(y = rpois(n, exp(eta)), x = x, g = factor(grp)))
}

scalar_term <- function(d) list(idx = d$grp, n_groups = d$G, n_coefs = 1L)

eb_pois <- function(d, ...) {
  tulpa_eb(d$y, NULL, d$X, scalar_term(d), family = "poisson", ...)
}


# --- 1. EB and the nested integrator share one objective ---------------------

test_that("tulpa_eb() and tulpa_re_cov_nested() find the same theta_hat", {
  d <- sim_re_pois(11L, G = 25L, per = 8L)
  eb <- eb_pois(d)
  nl <- tulpa_re_cov_nested(d$y, NULL, d$X, scalar_term(d), family = "poisson",
                            control = list(diagnose_k = FALSE, n_draws = 50))
  # Not "close": the same closure, the same optimizer, the same start. Anything
  # other than an exact match means the extraction grew a second configuration.
  expect_identical(eb$theta_hat, nl$theta_hat)
})

test_that("EB reports the nested integrator's plug-in summary, not its marginal", {
  d <- sim_re_pois(12L, G = 25L, per = 8L)
  eb <- eb_pois(d)
  nl <- tulpa_re_cov_nested(d$y, NULL, d$X, scalar_term(d), family = "poisson",
                            control = list(diagnose_k = FALSE, n_draws = 50))
  expect_equal(eb$map$sigma, nl$map$sigma)

  # And the plug-in is NOT the marginalized summary -- that difference is the
  # whole reason both backends exist, so pin that it is actually present.
  marg_median <- nl$posterior$median[nl$posterior$parameter == "sigma_1" |
                                       rownames(nl$posterior) == "sigma_1"]
  if (length(marg_median) == 1L && is.finite(marg_median)) {
    expect_false(isTRUE(all.equal(eb$map$sigma, marg_median, tolerance = 1e-8)))
  }
})


# --- 2. Parameter recovery ---------------------------------------------------

test_that("EB recovers a known RE standard deviation across seeds", {
  skip_on_cran()
  seeds <- 1:8
  est <- vapply(seeds, function(s) {
    d <- sim_re_pois(s, G = 40L, per = 10L, sigma = 0.7)
    eb_pois(d)$map$sigma
  }, numeric(1))

  expect_true(all(is.finite(est)))
  # Mean over seeds within 0.1 of truth: the PC prior pulls the estimate down a
  # little, so this is a bias bound, not a per-seed tolerance.
  expect_lt(abs(mean(est) - 0.7), 0.1)
  # No seed collapses to the sigma = 0 boundary or runs away.
  expect_true(all(est > 0.3 & est < 1.3))
})

test_that("EB recovers known fixed effects across seeds", {
  skip_on_cran()
  seeds <- 1:8
  cf <- vapply(seeds, function(s) {
    d <- sim_re_pois(s, G = 40L, per = 10L)
    coef(eb_pois(d))
  }, numeric(2))

  expect_lt(abs(mean(cf[1L, ]) - 0.4), 0.15)   # intercept
  expect_lt(abs(mean(cf[2L, ]) - 0.6), 0.05)   # slope
})

test_that("EB fixed-effect intervals cover the truth", {
  skip_on_cran()
  seeds <- 1:20
  covered <- vapply(seeds, function(s) {
    d <- sim_re_pois(s, G = 40L, per = 10L)
    ci <- confint(eb_pois(d))
    ci["x", 1L] <= 0.6 && 0.6 <= ci["x", 2L]
  }, logical(1))
  # Conditional-on-theta_hat intervals, so nominal 95% coverage is not the
  # claim; this pins that they are not grossly miscalibrated.
  expect_gte(mean(covered), 0.85)
})

test_that("EB recovers a scalar RE sd for a binomial response", {
  skip_on_cran()
  set.seed(101)
  G <- 60L; per <- 20L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- rnorm(G, 0, 0.9)
  y <- rbinom(n, 1L, plogis(-0.2 + 0.5 * x + b[grp]))
  fit <- tulpa_eb(y, rep(1L, n), cbind(1, x),
                  list(idx = grp, n_groups = G, n_coefs = 1L),
                  family = "binomial")
  expect_lt(abs(fit$map$sigma - 0.9), 0.25)
  expect_lt(abs(coef(fit)[2L] - 0.5), 0.2)
})

test_that("EB recovers a correlated random-slope covariance", {
  skip_on_cran()
  set.seed(202)
  G <- 60L; per <- 15L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  S <- matrix(c(0.64, 0.18, 0.18, 0.25), 2L, 2L)   # sd 0.8 / 0.5, rho 0.45
  L <- t(chol(S))
  b <- t(L %*% matrix(rnorm(2L * G), 2L, G))
  eta <- 0.3 + 0.5 * x + b[grp, 1L] + b[grp, 2L] * x
  y <- rpois(n, exp(eta))
  fit <- tulpa_eb(y, NULL, cbind(1, x),
                  list(idx = grp, n_groups = G, n_coefs = 2L,
                       Z = cbind(1, x), correlated = TRUE),
                  family = "poisson")
  expect_length(fit$map$sigma, 2L)
  expect_lt(abs(fit$map$sigma[1L] - 0.8), 0.25)
  expect_lt(abs(fit$map$sigma[2L] - 0.5), 0.25)
  expect_lt(abs(fit$map$rho - 0.45), 0.35)
  # Sigma is returned as the covariance itself, PD by construction.
  expect_equal(dim(fit$Sigma), c(2L, 2L))
  expect_gt(min(eigen(fit$Sigma, symmetric = TRUE, only.values = TRUE)$values), 0)
})

test_that("EB estimates each block of a two-term model", {
  skip_on_cran()
  set.seed(303)
  G <- 40L; H <- 30L; n <- 600L
  g <- sample.int(G, n, replace = TRUE)
  h <- sample.int(H, n, replace = TRUE)
  x <- rnorm(n)
  bg <- rnorm(G, 0, 0.9); bh <- rnorm(H, 0, 0.4)
  y <- rpois(n, exp(0.2 + 0.4 * x + bg[g] + bh[h]))
  fit <- tulpa_eb(y, NULL, cbind(1, x),
                  list(list(idx = g, n_groups = G, n_coefs = 1L, label = "g"),
                       list(idx = h, n_groups = H, n_coefs = 1L, label = "h")),
                  family = "poisson")
  expect_equal(fit$n_blocks, 2L)
  expect_named(fit$map, c("g", "h"))
  # The larger component is identified as the larger one, and both are in range.
  expect_gt(fit$map$g$sigma, fit$map$h$sigma)
  expect_lt(abs(fit$map$g$sigma - 0.9), 0.3)
  expect_lt(abs(fit$map$h$sigma - 0.4), 0.3)
})

test_that("a flat hyperprior gives the unpenalized ML-II estimate", {
  d <- sim_re_pois(7L, G = 40L, per = 10L, sigma = 0.7)
  pen  <- eb_pois(d)
  flat <- eb_pois(d, log_prior_theta = function(theta) 0)

  # With a flat prior the objective IS the log marginal, so its maximizer must
  # attain at least the log marginal the penalized maximizer does. This is the
  # property that pins the prior as an addend on the objective rather than
  # something applied elsewhere; the direction the estimate moves is not fixed
  # (the PC prior shrinks sigma, its log-scale Jacobian pushes the other way).
  expect_gte(flat$log_marginal, pen$log_marginal - 1e-6)
  expect_false(isTRUE(all.equal(flat$theta_hat, pen$theta_hat, tolerance = 1e-8)))
})


# --- 3. Offsets --------------------------------------------------------------

sim_offset <- function(seed = 9L, G = 30L, per = 10L) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  E <- runif(n, 1, 50)                       # exposure spanning 50x
  x <- rnorm(n)
  b <- rnorm(G, 0, 0.6)
  y <- rpois(n, E * exp(-1.5 + 0.6 * x + b[grp]))
  list(y = y, X = cbind(1, x), grp = grp, G = G, off = log(E),
       data = data.frame(y = y, x = x, g = factor(grp), E = E))
}

test_that("EB honours an offset instead of silently dropping it", {
  d <- sim_offset()
  term <- list(idx = d$grp, n_groups = d$G, n_coefs = 1L)
  with_off <- tulpa_eb(d$y, NULL, d$X, term, family = "poisson", offset = d$off)
  no_off   <- tulpa_eb(d$y, NULL, d$X, term, family = "poisson")

  # A dropped offset fits a different model entirely: the rate intercept is
  # -1.5, the count intercept is far above zero.
  expect_lt(abs(coef(with_off)[1L] - (-1.5)), 0.3)
  expect_gt(abs(coef(with_off)[1L] - coef(no_off)[1L]), 1)
  expect_lt(abs(coef(with_off)[2L] - 0.6), 0.15)
})

test_that("the nested integrator honours an offset too", {
  d <- sim_offset(10L)
  term <- list(idx = d$grp, n_groups = d$G, n_coefs = 1L)
  fit <- tulpa_re_cov_nested(d$y, NULL, d$X, term, family = "poisson",
                             offset = d$off,
                             control = list(diagnose_k = FALSE, n_draws = 50))
  expect_lt(abs(coef(fit)[1L] - (-1.5)), 0.3)
})

test_that("offset() reaches the EB backend through tulpa()", {
  d <- sim_offset(11L)
  fit <- tulpa(y ~ x + offset(log(E)) + (1 | g), data = d$data,
               family = "poisson", mode = "eb")
  expect_lt(abs(coef(fit)[1L] - (-1.5)), 0.3)
})

test_that("an offset the inner solve cannot carry errors rather than dropping", {
  d <- sim_offset(12L)
  term <- list(idx = d$grp, n_groups = d$G, n_coefs = 1L)
  # The AGHQ inner marginal runs on the compiled per-group oracle, which carries
  # no offset term.
  expect_error(
    tulpa_eb(d$y, NULL, d$X, term, family = "poisson", offset = d$off,
             n_quad = 5L),
    "does not support an offset")
  # And the Metropolis-within-Gibbs sampler has no offset term at all. The
  # counts are thresholded to 0/1 so the binomial response support holds and
  # the fit reaches the backend's offset check rather than stopping short of it.
  d$data$y01 <- as.integer(d$data$y > 0)
  expect_error(
    suppressWarnings(tulpa(y01 ~ x + offset(log(E)) + (1 | g), data = d$data,
                           family = "binomial", mode = "re_cov_gibbs")),
    "not supported by the re_cov_gibbs backend")
})


# --- 4. Fit object and accessors ---------------------------------------------

test_that("an EB fit carries the shape the generic accessors read", {
  d <- sim_re_pois(21L, G = 25L, per = 8L)
  fit <- eb_pois(d)
  expect_s3_class(fit, "tulpa_fit")
  expect_identical(fit$backend, "eb")
  expect_true(is.numeric(fit$mode))
  expect_true(is.matrix(fit$H_beta))
  expect_length(coef(fit), 2L)
  expect_equal(dim(vcov(fit)), c(2L, 2L))
  expect_equal(dim(confint(fit)), c(2L, 2L))
  expect_true(is.finite(fit$log_marginal))
  expect_s3_class(summary(fit), "data.frame")
  expect_output(print(fit), "eb")
})

test_that("the EB fit has no duplicated field names", {
  d <- sim_re_pois(23L, G = 25L, per = 8L)
  fit <- eb_pois(d)
  # The fit is built by c()-ing the inner Laplace result onto EB's own fields.
  # c() concatenates, so any shared name lands twice and `fit$name` silently
  # answers with whichever came first.
  expect_false(any(duplicated(names(fit))))
  expect_length(fit$log_marginal, 1L)
  expect_true(is.finite(fit$log_marginal))
})

test_that("EB reports the outer optimizer's convergence code", {
  d <- sim_re_pois(24L, G = 25L, per = 8L)
  fit <- eb_pois(d)
  expect_identical(fit$outer_convergence, 0L)
  # `converged` stays the INNER Newton solve's flag; the two are different
  # questions and a fit that conflated them could report success for either.
  expect_true(isTRUE(fit$converged))
})

test_that("a non-converging outer optimization warns rather than passing silently", {
  d <- sim_re_pois(25L, G = 25L, per = 8L)
  two_blocks <- list(
    list(idx = d$grp, n_groups = d$G, n_coefs = 1L, label = "a"),
    list(idx = d$grp, n_groups = d$G, n_coefs = 1L, label = "b"))

  # Two blocks puts the outer optimization on Nelder-Mead, which honours maxit;
  # starved of iterations it stops short, and the theta_hat it returns is where
  # it stopped rather than a maximizer. That has to be said out loud.
  expect_warning(
    fit <- tulpa_eb(d$y, NULL, d$X, two_blocks, family = "poisson",
                    control = list(outer_maxit = 2L)),
    "did not converge")
  expect_false(identical(fit$outer_convergence, 0L))

  # And the default budget does converge on the same data, so the warning is
  # reporting the budget rather than firing on every two-block model.
  expect_silent(fit2 <- tulpa_eb(d$y, NULL, d$X, two_blocks, family = "poisson"))
  expect_identical(fit2$outer_convergence, 0L)
})

test_that("a variance component pinned at the bracket is flagged, not reported", {
  # Brent returns convergence code 0 at a bracket endpoint, so the code alone
  # cannot tell a pinned value from a fitted one. The objective is
  # log_marginal + log_prior, so a hyperprior linear in log(sigma) drives the
  # maximizer to an endpoint deterministically -- which is what the low end
  # looks like in practice (the empirical-Bayes collapse to sigma = 0).
  d <- sim_re_pois(28L, G = 25L, per = 8L)
  term <- scalar_term(d)

  expect_warning(
    lo <- tulpa_eb(d$y, NULL, d$X, term, family = "poisson",
                   log_prior_theta = function(theta) -1e4 * theta),
    "lower end of the search bracket")
  expect_lt(lo$map$sigma, 1e-3)
  # The fixed effects are still returned; the warning is about sigma alone.
  expect_length(coef(lo), 2L)
  expect_true(all(is.finite(coef(lo))))

  expect_warning(
    hi <- tulpa_eb(d$y, NULL, d$X, term, family = "poisson",
                   log_prior_theta = function(theta) 1e4 * theta),
    "upper end of the search bracket")
  expect_gt(hi$map$sigma, 1e2)
})

test_that("a well-identified variance component draws no boundary warning", {
  d <- sim_re_pois(27L, G = 40L, per = 10L, sigma = 0.7)
  expect_silent(eb_pois(d))
})

test_that("outer_maxit reaches the nested integrator too", {
  d <- sim_re_pois(26L, G = 25L, per = 8L)
  expect_warning(
    tulpa_re_cov_nested(
      d$y, NULL, d$X,
      list(list(idx = d$grp, n_groups = d$G, n_coefs = 1L, label = "a"),
           list(idx = d$grp, n_groups = d$G, n_coefs = 1L, label = "b")),
      family = "poisson",
      control = list(outer_maxit = 2L, diagnose_k = FALSE, n_draws = 50)),
    "did not converge")
})

test_that("EB reports no posterior draws rather than a bogus array", {
  d <- sim_re_pois(22L, G = 25L, per = 8L)
  fit <- eb_pois(d)
  expect_null(posterior_sample(fit))
  expect_null(tulpa_draws_array(fit))
  expect_null(mcmc_draws(fit))
})


# --- 5. Front-door wiring ----------------------------------------------------

test_that("mode = 'eb' reaches the EB backend through tulpa()", {
  d <- sim_re_pois(31L, G = 25L, per = 8L)
  fit <- tulpa(y ~ x + (1 | g), data = d$data, family = "poisson", mode = "eb")
  expect_identical(fit$backend, "eb")
  expect_identical(fit$inference_tier, 2L)
  expect_true(is.finite(fit$map$sigma))
})

test_that("eb is registered but never selected by auto or by tier", {
  expect_true("eb" %in% ALL_BACKENDS)
  expect_identical(BACKEND_REGISTRY$eb$tier, "structured")
  expect_identical(BACKEND_REGISTRY$eb$fitter, "tulpa_eb")

  # Opt-in by name only: the plug-in understates hyperparameter uncertainty, so
  # nothing may route to it implicitly.
  fam <- list(name = "poisson", distribution = "poisson")
  for (n_obs in c(50, 5000, 100000)) {
    expect_false(identical(
      auto_select_mode(fam, n_obs, FALSE, FALSE, FALSE)$backend, "eb"))
  }
  expect_false(identical(
    select_backend_for_mode("structured", fam, 500, FALSE, FALSE), "eb"))
  expect_false(identical(
    select_backend_for_mode("exact", fam, 500, FALSE, FALSE), "eb"))
})

test_that("EB warns that a supplied sigma_re is estimated, not conditioned", {
  d <- sim_re_pois(32L, G = 25L, per = 8L)
  expect_warning(
    tulpa(y ~ x + (1 | g), data = d$data, family = "poisson", mode = "eb",
          sigma_re = 0.5),
    "estimated, not conditioned")
})

test_that("EB rejects a model with no random effects", {
  d <- sim_re_pois(33L, G = 25L, per = 8L)
  expect_error(
    tulpa_eb(d$y, NULL, d$X, list(), family = "poisson"),
    "no random-effect terms")
})

test_that("EB rejects control keys belonging to the nested integrator", {
  d <- sim_re_pois(34L, G = 25L, per = 8L)
  expect_error(
    tulpa_eb(d$y, NULL, d$X, scalar_term(d), family = "poisson",
             control = list(integration = "ccd")),
    "integration")
})


# --- estimate_phi on the front door ------------------------------------------

sim_re_gauss <- function(seed, G = 15L, per = 12L, sigma_re = 0.7,
                         sigma_y = 0.5) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  u <- rnorm(G, 0, sigma_re)
  data.frame(y = 1 + 0.5 * x + u[g] + rnorm(n, 0, sigma_y),
             x = x, g = factor(g))
}

test_that("estimate_phi frees the dispersion and reports it as estimated", {
  skip_on_cran()
  d <- sim_re_gauss(51L)

  held <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb"))
  free <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb",
          estimate_phi = TRUE))

  expect_false(held$phi_estimated)
  expect_identical(held$phi, 1.0)
  expect_true(free$phi_estimated)

  # The residual SD is 0.5 by construction, so a freed dispersion has to move
  # well away from the default it started at.
  expect_lt(abs(sqrt(free$phi) - 0.5), 0.1)
})

test_that("estimate_phi is refused where the dispersion is conditioned on", {
  d <- sim_re_gauss(52L)

  expect_error(
    suppressMessages(
      tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "laplace",
            sigma_re = 0.7, estimate_phi = TRUE)),
    "available under mode = 'eb'")
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb",
          estimate_phi = NA),
    "must be TRUE or FALSE")
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb",
          estimate_phi = "yes"),
    "must be TRUE or FALSE")
})

test_that("estimate_phi is refused for a family with no free dispersion", {
  skip_on_cran()
  d <- sim_re_pois(53L, G = 20L, per = 8L)
  expect_error(
    suppressMessages(
      tulpa(y ~ x + (1 | g), data = d$data, family = "poisson", mode = "eb",
            estimate_phi = TRUE)),
    "not available for family")
})
