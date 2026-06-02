# Generic ModelData sampler front door (tulpa_sample_glmm): NUTS (gcol33/tulpa#55)
# and ESS / SGHMC / SGLD / MCLMC / SMC / VI (gcol33/tulpa#54). One C++ entry
# (cpp_tulpa_sample_glmm) builds the ModelData via the built-in-family scaffold
# and dispatches to each kernel. These tests check (a) wiring + the draws-kind
# contract and (b) parameter recovery against the glm MLE reference.

# Posterior-mean fixed effects from every backend land near the glm MLE on a
# fixed-effect binomial GLM (the MLE is the right reference at a weak prior).
test_that("every sampler backend recovers the binomial-GLM fixed effects", {
  skip_on_cran()
  skip_if_fast()
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
  skip_on_cran()
  skip_if_fast()
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
  skip_on_cran()
  skip_if_fast()
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
  expect_true(is.null(mcmc_diagnostics(fit_iid)))
})

# Random-effect / spatial / offset models are routed away from the fixed-effect
# samplers rather than silently dropping the structure.
test_that("RE and offset models are rejected with guidance", {
  set.seed(404)
  n <- 200L
  d <- data.frame(y = rbinom(n, 1L, 0.4), x = rnorm(n),
                  g = factor(sample(6, n, TRUE)), o = runif(n))
  err_re <- expect_error(suppressMessages(
    tulpa(y ~ x + (1 | g), d, family = "binomial", mode = "hmc")))
  expect_match(conditionMessage(err_re), "random-effect")
  expect_match(conditionMessage(err_re), "mala")

  err_off <- expect_error(
    tulpa(y ~ x + offset(o), d, family = "binomial", mode = "ess"))
  expect_match(conditionMessage(err_off), "offset")
})
