# The fixed-effect prior is a modelling statement, and `mode = "auto"` routes on
# model structure: adding a random-effect term moves the fit to a different
# backend. If each backend carried its own default prior, that structural change
# would move the posterior for a reason the fit does not report.

test_that("there is one default fixed-effect prior, at the documented scale", {
  expect_equal(tulpa:::.TULPA_PRIOR$beta_sd, 2.5)
  expect_equal(tulpa:::.tulpa_prior_sd(), 2.5)
  expect_equal(tulpa:::.tulpa_default_beta_prior(), list(mean = 0, sd = 2.5))
  # prior_normal()'s own default is what the setting records.
  expect_equal(tulpa:::.TULPA_PRIOR$beta_sd, prior_normal()$sd)
})

test_that("every consumer reads that one setting", {
  for (nm in names(tulpa:::.PRIOR_CONSUMERS)) {
    expect_equal(tulpa:::.tulpa_prior_sd(nm), tulpa:::.TULPA_PRIOR$beta_sd,
                 info = nm)
  }
  expect_error(tulpa:::.tulpa_prior_sd("not_a_backend"),
               "Unknown fixed-effect prior consumer")
})

test_that("every fitter's own default resolves to the same prior SD", {
  fitters <- list(
    ep           = tulpa:::tulpa_ep,
    ep_fit       = tulpa:::ep_fit,
    gaussian     = tulpa:::tulpa_gaussian,
    gibbs        = tulpa:::tulpa_gibbs,
    multinomial  = tulpa:::tulpa_multinomial,
    ordinal      = tulpa:::tulpa_ordinal,
    re_cov_gibbs = tulpa:::tulpa_re_cov_gibbs,
    glmm_logpost = tulpa:::build_glmm_logpost
  )
  for (nm in names(fitters)) {
    bp <- eval(formals(fitters[[nm]])$beta_prior,
               envir = asNamespace("tulpa"))
    expect_equal(bp$sd, tulpa:::.TULPA_PRIOR$beta_sd, info = nm)
    expect_equal(bp$mean, 0, info = nm)
  }
})

test_that("the scalar-ridge resolver takes the same default", {
  expect_equal(tulpa:::.beta_prior_ridge_sd(NULL), tulpa:::.TULPA_PRIOR$beta_sd)
  expect_equal(tulpa:::.beta_prior_ridge_sd(list(mean = 0, sd = 7)), 7)
})

# --- a supplied prior the fitter cannot express is an error, not a swap -------

test_that("a beta_prior with no sd errors instead of taking the default", {
  # Substituting the default for an input the fitter cannot express replaces the
  # user's modelling statement and leaves no trace on the posterior.
  expect_error(tulpa:::.beta_prior_ridge_sd(prior_half_cauchy(2.5)),
               "half_cauchy|Gaussian on every fitter")
  expect_error(tulpa:::.beta_prior_ridge_sd(prior_pc(1, 0.01)),
               "Gaussian on every fitter")
  expect_error(tulpa:::.beta_prior_ridge_sd(list(sigma = 1)),
               "must supply `sd`")
  expect_error(tulpa:::.beta_prior_ridge_sd(list(scale = 1)),
               "must supply `sd`")
  expect_error(tulpa:::.beta_prior_ridge_sd(2.5), "must be NULL or a list")
})

test_that("the two prior resolvers accept and reject the same inputs", {
  ok   <- list(list(mean = 0, sd = 1), prior_normal(0, 1), list(sd = 3))
  bad  <- list(prior_half_cauchy(2.5), prior_pc(1, 0.01), list(sigma = 1))
  for (bp in ok) {
    expect_silent(tulpa:::.normalize_beta_prior(bp, 2))
    expect_silent(tulpa:::.beta_prior_ridge_sd(bp))
  }
  for (bp in bad) {
    expect_error(tulpa:::.normalize_beta_prior(bp, 2))
    expect_error(tulpa:::.beta_prior_ridge_sd(bp))
  }
})

test_that("prior_normal() with no sd is still accepted through its own default", {
  expect_equal(tulpa:::.beta_prior_fields(prior_normal())$sd, 2.5)
  expect_equal(tulpa:::.beta_prior_fields(list(sd = 4)), list(mean = 0, sd = 4))
})

# --- what the fit reports ----------------------------------------------------

test_that("a fit records the fixed-effect prior it ran under", {
  skip_on_cran()
  set.seed(4)
  n <- 60
  d <- data.frame(x = rnorm(n))
  d$y <- rbinom(n, 1L, plogis(0.3 + 0.5 * d$x))

  fit <- tulpa(y ~ x, data = d, family = "binomial", mode = "ep")
  expect_equal(fit$beta_prior$sd, tulpa:::.TULPA_PRIOR$beta_sd)
  expect_equal(attr(summary(fit), "beta_prior")$sd,
               tulpa:::.TULPA_PRIOR$beta_sd)

  supplied <- tulpa(y ~ x, data = d, family = "binomial", mode = "ep",
                    beta_prior = list(mean = 0, sd = 0.4))
  expect_equal(supplied$beta_prior$sd, 0.4)
  # A tighter prior on the same data has to shrink the slope.
  expect_lt(abs(unname(coef(supplied)[2])), abs(unname(coef(fit)[2])))
})

test_that("the backend mode = auto picks does not change the prior", {
  skip_on_cran()
  # The issue's own pair: a fixed-effect model routes to mala and the same model
  # plus a random-effect term to re_cov_gibbs, which used to mean N(0, 2.5)
  # against N(0, 100) on the fixed effects.
  set.seed(9)
  n <- 90
  d <- data.frame(x = rnorm(n), g = factor(rep(1:9, each = 10)))
  d$y <- rbinom(n, 1L, plogis(0.2 + 0.4 * d$x))

  fixed <- tulpa(y ~ x, data = d, family = "binomial", mode = "auto")
  re    <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "auto")
  expect_false(identical(fixed$backend, re$backend))
  expect_equal(fixed$beta_prior$sd, tulpa:::.TULPA_PRIOR$beta_sd)
  expect_equal(re$beta_prior$sd, tulpa:::.TULPA_PRIOR$beta_sd)
})
