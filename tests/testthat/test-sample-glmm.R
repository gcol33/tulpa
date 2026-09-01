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
  fg <- tulpa_sample_glmm(yg, rep(1L, n), X, "gaussian", "hmc", phi = 0.4096,
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

# ESS random-effect recovery: the RE block previously used a unit ellipse and
# the full posterior as its slice target, double-counting the RE prior and
# shrinking sigma_re low (gcol33/tulpa#150). Multi-seed gate on the corrected
# path (truth sigma_re = 0.7); the fix binds the ellipse to sigma_re and removes
# the block prior from the slice target.
test_that("ESS recovers the RE variance without double-prior shrinkage", {
  skip_if_not_slow()
  sig <- numeric(6)
  for (s in seq_len(6)) {
    set.seed(200 + s)
    G <- 40L; npg <- 12L; N <- G * npg
    grp <- rep(seq_len(G), each = npg); x <- rnorm(N)
    b <- rnorm(G, 0, 0.7)
    y <- rpois(N, exp(0.2 + 0.5 * x + b[grp]))
    fit <- suppressMessages(tulpa(
      y ~ x + (1 | g), data.frame(y = y, x = x, g = factor(grp)),
      family = "poisson", mode = "ess",
      control = list(n_iter = 1500L, warmup = 750L, seed = 7L)))
    sig[s] <- exp(fit$means[["log_sigma_re"]])
  }
  # Mean recovered sigma_re within 20% of truth (was biased low pre-fix).
  expect_lt(abs(mean(sig) - 0.7) / 0.7, 0.20)
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
# ELBO (collapsed or exploded variances) fails.
#
# The variant is named explicitly rather than left to `vi_variant = 3`
# (VIVariant::AUTO), which selects full-rank below D = 200 and low-rank in
# [200, 2000) -- at D = 2 the default is full-rank, so leaving it implicit gives
# the mean-field path no coverage at all. The gradients themselves are
# finite-difference checked per variant in test-vi-gradient.R; these two fits
# are the recovery arm.
for (.vi_variant in list(list(code = 2L, name = "full-rank"),
                         list(code = 1L, name = "low-rank"))) {
  local({
    code <- .vi_variant$code
    name <- .vi_variant$name
    test_that(sprintf(
      "%s VI posterior SDs sit in a quantified band around the asymptotic SE",
      name), {
      skip_if_not_slow()
      set.seed(303)
      n <- 600L
      x <- rnorm(n)
      X <- cbind(1, x)
      y <- rpois(n, exp(0.4 + 0.6 * x))
      fit <- tulpa_sample_glmm(
        y = as.numeric(y), n_trials = rep(1L, n), X = X,
        family = "poisson", backend = "vi",
        control = list(vi_variant = code, vi_max_iter = 8000L,
                       n_draws = 4000L, seed = 17L))
      ref <- glm(y ~ x, family = poisson)
      se_ref <- unname(sqrt(diag(vcov(ref))))
      expect_lt(max(abs(unname(fit$means) - unname(coef(ref)))), 0.15)
      sd_fit <- apply(tulpa:::.fixed_draws_mat(fit), 2, stats::sd)[seq_along(se_ref)]
      ratio  <- unname(sd_fit) / se_ref
      expect_true(all(ratio > 0.4 & ratio < 1.8),
                  label = paste(name, "VI sd / asymptotic se:",
                                paste(round(ratio, 3), collapse = ", ")))
    })
  })
}

# --- stochastic / auxiliary sampler kernels -----------------------------------

# The elliptical-slice kernel draws from R's own RNG, so set.seed() is what
# fixes a run and control$seed does not reach it.
test_that("the ESS backend reproduces a run from set.seed (gcol33/tulpa#557)", {
  skip_on_cran()
  set.seed(91)
  n <- 120L
  x <- rnorm(n); X <- cbind(1, x)
  y <- as.numeric(rbinom(n, 1L, plogis(0.3 + 0.6 * x)))
  ess_draws <- function(seed, ctrl = list()) {
    set.seed(seed)
    tulpa_sample_glmm(y, rep(1L, n), X, "binomial", "ess",
                      control = c(list(n_iter = 120L, warmup = 60L), ctrl))$draws
  }
  a <- ess_draws(4L)
  b <- ess_draws(4L)
  d <- ess_draws(5L)
  expect_equal(a, b)
  expect_false(isTRUE(all.equal(a, d)))

  # control$seed is carried into the other kernels' own generators and is inert
  # here, so it cannot move an ESS run.
  s1 <- ess_draws(4L, list(seed = 1L))
  s2 <- ess_draws(4L, list(seed = 2L))
  expect_equal(s1, s2)
})

# ess_adapt_interval is read by the kernel: two runs from one R seed that differ
# only in it must diverge (they scale the same random-walk draws differently).
# Needs a random-effect term, since that is what puts a parameter on the
# adapted random-walk block in the first place.
test_that("ess_adapt_interval reaches the kernel (gcol33/tulpa#571)", {
  skip_on_cran()
  set.seed(77)
  G <- 12L; npg <- 8L; N <- G * npg
  grp <- rep(seq_len(G), each = npg); x <- rnorm(N)
  b <- rnorm(G, 0, 0.6)
  d <- data.frame(y = rpois(N, exp(0.2 + 0.4 * x + b[grp])),
                  x = x, g = factor(grp))
  run <- function(interval) {
    set.seed(12)
    suppressMessages(suppressWarnings(tulpa(
      y ~ x + (1 | g), d, family = "poisson", mode = "ess",
      control = list(n_iter = 160L, warmup = 80L,
                     ess_adapt_during_warmup = TRUE,
                     ess_adapt_interval = interval))))$draws
  }
  expect_false(isTRUE(all.equal(run(5L), run(200L))))
})

# A control knob that no longer reaches any kernel is rejected, not accepted and
# dropped. Structural: the control check runs before any model is built.
test_that("the withdrawn ESS cholesky knob is rejected", {
  expect_error(
    tulpa_sample_glmm(y = c(0, 1), n_trials = c(1L, 1L),
                      X = cbind(1, c(-1, 1)), family = "binomial",
                      backend = "ess",
                      control = list(ess_use_cholesky = TRUE)),
    "ess_use_cholesky")
})

# The draw matrix is sized by the number of times the store condition fires, so
# a warmup at least as long as the run stores nothing instead of asking Eigen
# for a negative extent.
test_that("sampler backends size the draw matrix by ceil (gcol33/tulpa#473)", {
  skip_on_cran()
  set.seed(31)
  n <- 60L
  x <- rnorm(n); X <- cbind(1, x)
  y <- as.numeric(rbinom(n, 1L, plogis(0.1 + 0.4 * x)))
  for (backend in c("sghmc", "sgld", "ess")) {
    fit <- suppressWarnings(tulpa_sample_glmm(
      y, rep(1L, n), X, "binomial", backend,
      control = list(n_iter = 40L, warmup = 60L, seed = 3L)))
    expect_equal(nrow(fit$draws), 0L, info = backend)
  }
})

# SGHMC's injected momentum noise sets the sampled temperature. At
# sqrt(2 * alpha * epsilon) the chain samples p(theta)^(1/epsilon), whose
# marginal sds are short by sqrt(epsilon) -- 0.1 to 0.3 over the step sizes the
# adapter is bounded to. The band separates that from a chain at temperature 1;
# it is a band, not a point, because the Euler discretization carries an O(eps)
# error of its own. The reference is the asymptotic (glm) SE, which is the
# posterior sd at this n under the weak prior.
test_that("SGHMC samples the posterior spread, not a tempered one (gcol33/tulpa#422)", {
  skip_if_not_slow()
  set.seed(717)
  n <- 800L
  x1 <- rnorm(n); x2 <- rnorm(n); X <- cbind(1, x1, x2)
  y <- as.numeric(rbinom(n, 1L, plogis(-0.3 + 0.8 * x1 - 0.4 * x2)))
  ref <- glm(y ~ x1 + x2, family = binomial)
  se_ref <- unname(sqrt(diag(vcov(ref))))
  fit <- tulpa_sample_glmm(
    y, rep(1L, n), X, "binomial", "sghmc",
    fixed_names = c("(Intercept)", "x1", "x2"),
    control = list(n_iter = 6000L, warmup = 3000L, seed = 23L))
  ratio <- unname(apply(fit$draws, 2, stats::sd)[seq_along(se_ref)]) / se_ref
  expect_true(all(ratio > 0.4 & ratio < 2.5),
              label = paste("SGHMC sd / asymptotic se:",
                            paste(round(ratio, 3), collapse = ", ")))
})

# The SMC driver initializes its particles from a Gaussian perturbation of
# `init`, not from p(theta), and carries no p / q correction in the weights, so
# the log-Z accumulator estimates a different integral than the marginal
# likelihood. It reports no evidence rather than that number; the draws recover
# because the mutations target p(theta) L(theta)^beta.
test_that("the SMC backend reports no evidence under its stand-in prior draw (gcol33/tulpa#445)", {
  skip_on_cran()
  set.seed(52)
  n <- 200L
  x <- rnorm(n); X <- cbind(1, x)
  y <- as.numeric(rbinom(n, 1L, plogis(0.2 + 0.5 * x)))
  fit <- tulpa_sample_glmm(
    y, rep(1L, n), X, "binomial", "smc",
    control = list(n_particles = 300L, n_mcmc_steps = 2L, seed = 8L))
  expect_true(is.na(fit$log_evidence))
  expect_gt(nrow(fit$draws), 0L)
  expect_equal(ncol(fit$draws), 2L)
})
