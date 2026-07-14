# Generic ModelData sampler front door (tulpa_sample_glmm): NUTS (gcol33/tulpa#55)
# and ESS / SGHMC / SGLD / MCLMC / SMC / VI (gcol33/tulpa#54). One C++ entry
# (cpp_tulpa_sample_glmm) builds the ModelData via the built-in-family scaffold
# and dispatches to each kernel. These tests check (a) wiring + the draws-kind
# contract and (b) parameter recovery against the glm MLE reference.

# Posterior-mean fixed effects from every backend land near the glm MLE on a
# fixed-effect binomial GLM (the MLE is the right reference at a weak prior).
test_that("every sampler backend recovers the binomial-GLM fixed effects", {
  skip_if_not_slow()
  set.seed(101)
  n <- 800L
  x1 <- rnorm(n); x2 <- rnorm(n)
  X  <- cbind(1, x1, x2)
  beta_true <- c(-0.4, 0.9, -0.5)
  y <- rbinom(n, 1L, plogis(as.numeric(X %*% beta_true)))
  mle <- unname(coef(glm(y ~ x1 + x2, family = binomial)))

  # Looser tolerance for the stochastic-gradient and variational approximations.
  cfg <- list(
    hmc   = list(tol = 0.12, ctrl = list(n_iter = 1500L, warmup = 750L, n_chains = 2L)),
    ess   = list(tol = 0.12, ctrl = list(n_iter = 1500L, warmup = 750L)),
    sghmc = list(tol = 0.25, ctrl = list(n_iter = 2000L, warmup = 1000L)),
    sgld  = list(tol = 0.30, ctrl = list(n_iter = 2000L, warmup = 1000L)),
    mclmc = list(tol = 0.15, ctrl = list(n_iter = 1500L, warmup = 750L)),
    smc   = list(tol = 0.20, ctrl = list(n_particles = 2000L, n_mcmc_steps = 8L)),
    vi    = list(tol = 0.25, ctrl = list(vi_max_iter = 8000L, n_draws = 3000L))
  )
  for (backend in names(cfg)) {
    fit <- tulpa_sample_glmm(
      y = as.numeric(y), n_trials = rep(1L, n), X = X,
      family = "binomial", backend = backend,
      fixed_names = c("(Intercept)", "x1", "x2"),
      control = c(cfg[[backend]]$ctrl, list(seed = 11L)))
    bh <- fit$means
    expect_equal(length(bh), 3L, info = backend)
    expect_lt(max(abs(unname(bh) - mle)), cfg[[backend]]$tol)
  }
})

# Poisson + Gaussian through NUTS, to exercise the family scaffold beyond binomial.
test_that("NUTS recovers poisson and gaussian fixed effects", {
  skip_if_not_slow()
  set.seed(202)
  n <- 800L; x <- rnorm(n); X <- cbind(1, x)

  yp <- rpois(n, exp(0.5 + 0.6 * x))
  fp <- tulpa_sample_glmm(yp, rep(1L, n), X, "poisson", "hmc",
                          control = list(n_iter = 1500L, warmup = 750L,
                                         n_chains = 2L, seed = 5L))
  mlp <- unname(coef(glm(yp ~ x, family = poisson)))
  expect_lt(max(abs(unname(fp$means) - mlp)), 0.10)

  yg <- 1.0 - 0.7 * x + rnorm(n, 0, 0.8)
  fg <- tulpa_sample_glmm(yg, rep(1L, n), X, "gaussian", "hmc", phi = 0.64,
                          control = list(n_iter = 1500L, warmup = 750L,
                                         n_chains = 2L, seed = 6L))
  mlg <- unname(coef(lm(yg ~ x)))
  expect_lt(max(abs(unname(fg$means) - mlg)), 0.10)
})

# The draws-kind contract: chain backends emit MCMC chains (Rhat/ESS apply),
# iid backends emit particles/variational draws (the diagnostics gate withholds).
test_that("draws_kind contract holds through tulpa(mode = ...)", {
  skip_if_not_slow()
  set.seed(303)
  n <- 400L; x <- rnorm(n)
  d <- data.frame(y = rbinom(n, 1L, plogis(0.2 + 0.7 * x)), x = x)

  fit_chain <- tulpa(y ~ x, d, family = "binomial", mode = "hmc",
                     control = list(n_iter = 1000L, warmup = 500L, n_chains = 3L))
  expect_equal(fit_chain$draws_kind, "chain")
  expect_equal(fit_chain$n_chains, 3L)
  dd <- mcmc_diagnostics(fit_chain)
  expect_true(is.data.frame(dd))
  expect_true(all(is.finite(dd$rhat)))

  fit_iid <- tulpa(y ~ x, d, family = "binomial", mode = "smc",
                   control = list(n_particles = 1500L))
  expect_equal(fit_iid$draws_kind, "iid")
  # Non-chain fits withhold the chain-only view; mcmc_diagnostics dispatches
  # to the approximation-reliability table instead of vacuous Rhat/ESS.
  expect_null(mcmc_draws(fit_iid))
  expect_false(is.null(mcmc_diagnostics(fit_iid)))
})

# Random-effect models now thread through the ModelData samplers (gcol33/tulpa#75)
# instead of being routed away; offset() is threaded through too.
test_that("RE models thread through the sampler; offset() is supported", {
  skip_if_not_slow()
  set.seed(404)
  G <- 18L; n_per <- 16L; N <- G * n_per
  g <- rep(seq_len(G), each = n_per); x <- rnorm(N)
  b <- rnorm(G, 0, 0.7)
  y <- rpois(N, exp(0.4 + 0.6 * x + b[g]))
  d <- data.frame(y = y, x = x, g = factor(g))
  fit <- suppressMessages(tulpa(y ~ x + (1 | g), d, family = "poisson",
                                mode = "hmc",
                                control = list(n_iter = 1200L, warmup = 600L,
                                               n_chains = 2L, seed = 11L)))
  expect_true("log_sigma_re" %in% fit$param_names)
  expect_equal(fit$n_params, 2L + 1L + G)         # beta(2) + log_sigma_re + b[G]
  expect_lt(abs(fit$means[["x"]] - 0.6), 0.18)
  expect_lt(abs(exp(fit$means[["log_sigma_re"]]) - 0.7), 0.35)

  # offset() threads into the sampler's linear predictor (gcol33/tulpa#72).
  d$o <- runif(N)
  fit_off <- tulpa(y ~ x + offset(o), d, family = "poisson", mode = "ess",
                   control = list(n_iter = 400L, warmup = 200L))
  expect_false(is.null(fit_off$draws))
  expect_equal(ncol(fit_off$draws), 2L)
})

# offset() reaches the kernel's linear predictor: a log-exposure offset on a
# Poisson rate model recovers the same fixed effects as glm(offset = ...).
test_that("the sampler recovers fixed effects under a log-exposure offset (gcol33/tulpa#72)", {
  skip_if_not_slow()
  set.seed(505)
  n <- 800L; x <- rnorm(n); X <- cbind(1, x)
  expo <- runif(n, 0.5, 3); off <- log(expo)
  y <- rpois(n, expo * exp(0.4 + 0.7 * x))
  fit <- tulpa_sample_glmm(
    as.numeric(y), rep(1L, n), X, "poisson", "hmc", offset = off,
    control = list(n_iter = 1500L, warmup = 750L, n_chains = 2L, seed = 7L))
  mle <- unname(coef(glm(y ~ x + offset(off), family = poisson)))
  expect_lt(max(abs(unname(fit$means) - mle)), 0.10)
})

# offset = 0 must reproduce the no-offset draws bit-for-bit at a fixed seed: the
# linear predictors are identical, so the deterministic chain is identical.
test_that("offset = 0 reproduces the no-offset sampler draws exactly (gcol33/tulpa#72)", {
  skip_if_not_slow()
  set.seed(606)
  n <- 300L; x <- rnorm(n); X <- cbind(1, x)
  y <- rbinom(n, 1L, plogis(0.2 + 0.6 * x))
  ctrl <- list(n_iter = 600L, warmup = 300L, n_chains = 1L, seed = 9L)
  a <- tulpa_sample_glmm(as.numeric(y), rep(1L, n), X, "binomial", "hmc",
                         offset = NULL, control = ctrl)
  b <- tulpa_sample_glmm(as.numeric(y), rep(1L, n), X, "binomial", "hmc",
                         offset = rep(0, n), control = ctrl)
  expect_equal(b$draws, a$draws, tolerance = 1e-10)
})

# VI is validated at mean level above; this quantifies its SPREAD. The band
# makes the variational marginal SDs a measured property instead of an
# unchecked one: within [0.4, 1.8] of the asymptotic (glm) SE, so a broken
# ELBO (collapsed or exploded variances) fails. Measured at this seed the
# mean-field fit over-disperses the intercept (~1.5x) and slightly
# under-disperses the slope (~0.9x).
test_that("VI posterior SDs sit in a quantified band around the asymptotic SE", {
  skip_if_not_slow()
  set.seed(303)
  n <- 600L
  x <- rnorm(n)
  X <- cbind(1, x)
  y <- rpois(n, exp(0.4 + 0.6 * x))
  fit <- tulpa_sample_glmm(
    y = as.numeric(y), n_trials = rep(1L, n), X = X,
    family = "poisson", backend = "vi",
    control = list(vi_max_iter = 8000L, n_draws = 4000L, seed = 17L))
  ref <- glm(y ~ x, family = poisson)
  se_ref <- unname(sqrt(diag(vcov(ref))))
  expect_lt(max(abs(unname(fit$means) - unname(coef(ref)))), 0.15)
  sd_fit <- apply(tulpa:::.fixed_draws_mat(fit), 2, stats::sd)[seq_along(se_ref)]
  ratio  <- unname(sd_fit) / se_ref
  expect_true(all(ratio > 0.4 & ratio < 1.8),
              label = paste("VI sd / asymptotic se:",
                            paste(round(ratio, 3), collapse = ", ")))
})
