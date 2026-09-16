# Power-scaling prior/likelihood sensitivity (gcol33/tulpa C1). The CJS +
# gradient pipeline is validated end-to-end: a tight prior that conflicts with
# the data must be far more prior-sensitive than a weak prior on the same data
# and must flag a prior-data conflict; the likelihood component must be finite
# and informative; and the input guards must fire.

test_that("power-scaling flags a tight conflicting prior as more sensitive", {
  skip_on_cran()
  set.seed(7)
  n <- 250L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.4 + 0.8 * d$x))

  fit_weak  <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
                     beta_prior = list(mean = 0, sd = 5))
  fit_tight <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
                     beta_prior = list(mean = 0, sd = 0.15))

  s_weak  <- tulpa_powerscale_sensitivity(fit_weak,  prior = list(mean = 0, sd = 5))
  s_tight <- tulpa_powerscale_sensitivity(fit_tight, prior = list(mean = 0, sd = 0.15))

  # Likelihood component is finite and informative.
  expect_true(all(is.finite(s_weak$likelihood)))
  expect_true(all(s_weak$likelihood > 0))

  # The tight, conflicting prior is much more prior-sensitive on the slope.
  islope <- which(s_tight$variable == "x")
  expect_gt(s_tight$prior[islope], s_weak$prior[islope])

  # ... and a conflict is flagged for it.
  expect_true(any(grepl("conflict", s_tight$diagnosis)))
})

test_that("hyperparameter-prior sensitivity reads the stored per-draw log-prior", {
  skip_on_cran()
  set.seed(9)
  n <- 300L
  g <- factor(rep(1:12, length.out = n))
  b0 <- rnorm(12, 0, 0.8); b1 <- rnorm(12, 0, 0.5)
  d <- data.frame(x = rnorm(n), g = g)
  d$y <- rbinom(n, 1, plogis(0.2 + 0.6 * d$x +
                             b0[as.integer(g)] + b1[as.integer(g)] * d$x))

  # Random slopes route mode = "laplace" to the RE-covariance integrator,
  # whose mixture draws now carry the per-draw hyperparameter log-prior.
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "laplace")
  expect_false(is.null(fit$hyper_log_prior_draws))
  expect_length(fit$hyper_log_prior_draws, nrow(fit$draws))
  expect_true(all(is.finite(fit$hyper_log_prior_draws)))

  s <- tulpa_powerscale_sensitivity(fit)
  expect_true("hyperparameter" %in% names(s))
  expect_true(all(is.finite(s$hyperparameter)))
  expect_true(all(s$hyperparameter >= 0))

  # A plain fixed-effect Laplace fit has no stored hyper log-prior -> NA.
  d2 <- data.frame(x = rnorm(100)); d2$y <- rpois(100, exp(0.3 + 0.4 * d2$x))
  fit2 <- tulpa(y ~ x, data = d2, family = "poisson", mode = "laplace")
  s2 <- tulpa_powerscale_sensitivity(fit2)
  expect_true(all(is.na(s2$hyperparameter)))
})

test_that("power-scaling: likelihood-only mode and input guards", {
  skip_on_cran()
  set.seed(8)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.3 + 0.5 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  s <- tulpa_powerscale_sensitivity(fit)             # no prior -> likelihood only
  expect_true(all(is.na(s$prior)))
  expect_true(all(is.finite(s$likelihood)))
  expect_equal(nrow(s), 2L)

  # Prior component without a prior spec errors.
  expect_error(tulpa_powerscale_sensitivity(fit, prior = list(mean = 0)), "sd")

  # Spatial fit rejected.
  sp <- structure(list(spatial = list(type = "icar"), family = "poisson"),
                  class = "tulpa_fit")
  expect_error(tulpa_powerscale_sensitivity(sp), "not supported")

  # A fit exposing no engine formula (e.g. a tulpaObs fit) is refused by
  # name rather than crashing inside the formula parser.
  tobs_like <- structure(
    list(family = "poisson", formula = NULL, model_matrix = matrix(1, 2, 1)),
    class = "tulpa_fit")
  expect_error(tulpa_powerscale_sensitivity(tobs_like), "engine formula")
})

# gcol33/tulpa#784: the likelihood component used to be built from the
# fixed-effect linear predictor alone (n_trials = 1, no RE, no offset), which
# gave -Inf on an n_trials > 1 binomial RE fit and a component wildly off the
# fit's own pointwise log-likelihood on a poisson RE / offset fit. It now
# reuses .tulpa_eta_draws() / .tulpa_eta_loglik() -- the same assembly
# waic()/loo() read -- so every component the fit carries is included.
test_that("the likelihood component includes n_trials, random effects and the offset", {
  skip_on_cran()
  set.seed(1)
  n <- 160L; G <- 12L
  x <- rnorm(n); g <- factor(sample(seq_len(G), n, replace = TRUE))
  b0 <- rnorm(G, 0, 0.5); b1 <- rnorm(G, 0, 0.3)
  d <- data.frame(x = x, g = g,
                  y = rbinom(n, 10, plogis(-0.3 + x + b0[g] + b1[g] * x)))
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", n_trials = 10,
               mode = "laplace", sigma_re = 0.5)

  s <- tulpa_powerscale_sensitivity(fit)
  expect_true(all(is.finite(s$likelihood)))

  # A poisson random-intercept HMC fit: the component is the rowSums of the
  # fit's own pointwise log-likelihood (RE and all), so their means agree to
  # Monte-Carlo error rather than differing by hundreds of nats.
  g2 <- factor(sample(1:12, 200, TRUE)); x2 <- rnorm(200); b <- rnorm(12, 0, 1.5)
  d2 <- data.frame(x = x2, g = g2, y = rpois(200, exp(0.3 + 0.5 * x2 + b[g2])))
  fit2 <- tulpa(y ~ x + (1 | g), data = d2, family = "poisson", mode = "hmc",
                control = list(n_iter = 300, warmup = 150, n_chains = 2, seed = 1))
  full_ll <- mean(rowSums(pointwise_loglik(fit2)))
  comp_ll <- mean(tulpa:::.ps_component_loglik(fit2))
  # The two read the SAME .tulpa_eta_draws()/.tulpa_eta_loglik() assembly, so
  # they must agree closely -- unlike the old fixed-effects-only component,
  # which was off from the fit's own pointwise log-likelihood by hundreds of
  # nats on this fixture.
  expect_equal(comp_ll, full_ll, tolerance = 5)
  s2 <- tulpa_powerscale_sensitivity(fit2)
  expect_true(all(is.finite(s2$likelihood)))

  # An offset: dropping it previously moved the total by tens of nats.
  d2$expo <- runif(200, 0.5, 2)
  d2$y2 <- rpois(200, d2$expo * exp(0.3 + 0.5 * d2$x))
  fit3 <- tulpa(y2 ~ x + offset(log(expo)), data = d2, family = "poisson",
                mode = "laplace")
  s3 <- tulpa_powerscale_sensitivity(fit3)
  expect_true(all(is.finite(s3$likelihood)))
})

# gcol33/tulpa#784: the guard only checked fit$spatial / fit$temporal /
# fit$temporal_field, so an inline spatial() field (stored as
# fit$spatial_fields) or a latent() block (a nested-Laplace outer-grid mixture
# fit) were silently accepted and scored on a fixed-effects-only component.
test_that("inline field and latent-block fits are rejected, not silently scored", {
  skip_on_cran()
  spatial_fields_fit <- structure(
    list(family = "poisson", spatial_fields = list(1), formula = y ~ x),
    class = "tulpa_fit")
  expect_error(tulpa_powerscale_sensitivity(spatial_fields_fit), "not supported")

  grid_mixture_fit <- structure(
    list(family = "poisson", formula = y ~ x,
         fitted_eta = matrix(0, 3, 5), weights = c(0.2, 0.3, 0.5)),
    class = "tulpa_fit")
  expect_error(tulpa_powerscale_sensitivity(grid_mixture_fit), "not supported")
})
