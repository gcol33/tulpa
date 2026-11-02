
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
                         list(code = 1L, name = "low-rank"),
                         list(code = 0L, name = "mean-field"))) {
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

# The draw matrix is sized by the number of times the store condition fires,
# and a warmup at least as long as the run stored nothing (gcol33/tulpa#473).
# Such a run is now refused before it reaches the store, by the one
# run-length check every sampler resolves its defaults through
# (gcol33/tulpa#872).
test_that("sampler backends refuse warmup >= n_iter (gcol33/tulpa#473, #872)", {
  set.seed(31)
  n <- 60L
  x <- rnorm(n); X <- cbind(1, x)
  y <- as.numeric(rbinom(n, 1L, plogis(0.1 + 0.4 * x)))
  for (backend in c("sghmc", "sgld", "ess")) {
    expect_error(tulpa_sample_glmm(
      y, rep(1L, n), X, "binomial", backend,
      control = list(n_iter = 40L, warmup = 60L, seed = 3L)),
      "warmup", info = backend)
  }
})

skip_on_cran_budget()

# A control knob that no longer reaches any kernel is rejected, not accepted and
# dropped. Structural: the control check runs before any model is built.
test_that("the withdrawn ESS cholesky knob is rejected", {
  skip_on_cran_budget()
  expect_error(
    tulpa_sample_glmm(y = c(0, 1), n_trials = c(1L, 1L),
                      X = cbind(1, c(-1, 1)), family = "binomial",
                      backend = "ess",
                      control = list(ess_use_cholesky = TRUE)),
    "ess_use_cholesky")
})

skip_on_cran()
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
    smc   = list(tol = 0.20, ctrl = list(n_particles = 2000L)),
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
  dd <- diagnostics(fit_chain)
  expect_true(is.data.frame(dd))
  expect_true(all(is.finite(dd$rhat)))

  fit_iid <- tulpa(y ~ x, d, family = "binomial", mode = "smc",
                   control = list(n_particles = 1500L))
  expect_equal(fit_iid$draws_kind, "iid")
  # Non-chain fits withhold the chain-only view; diagnostics() dispatches
  # to the approximation-reliability table instead of vacuous Rhat/ESS.
  expect_null(mcmc_draws(fit_iid))
  expect_false(is.null(diagnostics(fit_iid)))
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

# The SMC path runs from a normalized Gaussian reference to the posterior, so
# its running log Z plus the normalizing constants of the prior and likelihood
# it tempered is log p(y). On a Gaussian linear model with a known residual
# variance and a N(0, sd^2) prior on every coefficient the evidence is closed
# form: y ~ N(0, phi I + sd^2 X X').
test_that("the SMC backend reports the log evidence with its Monte Carlo SE (gcol33/tulpa#1061)", {
  skip_on_cran()
  set.seed(1060)
  n <- 60L
  x <- rnorm(n); X <- cbind(1, x)
  y <- 0.5 + 0.8 * x + rnorm(n, 0, 0.8)
  phi <- 0.64; sdb <- 2
  S <- phi * diag(n) + sdb^2 * tcrossprod(X)
  R <- chol(S)
  z <- backsolve(R, y, transpose = TRUE)
  exact <- -0.5 * n * log(2 * pi) - sum(log(diag(R))) - 0.5 * sum(z^2)

  fit <- tulpa_sample_glmm(
    y, NULL, X, "gaussian", "smc", phi = phi, sigma_beta = sdb,
    control = list(n_particles = 1000L, seed = 8L))
  expect_true(is.finite(fit$log_evidence))
  expect_gt(fit$log_evidence_se, 0)
  expect_lt(fit$log_evidence_se, 0.2)
  expect_null(fit$log_evidence_declined)
  expect_lt(abs(fit$log_evidence - exact), 0.25)

  ll <- logLik(fit)
  expect_identical(attr(ll, "quantity"), "log_evidence")
  expect_equal(as.numeric(ll), fit$log_evidence)
  expect_equal(attr(ll, "mc_se"), fit$log_evidence_se)
})

# A random intercept's prior is normalized too (half-Cauchy on its SD, N(0, 1)
# on the non-centered effects): the evidence integrates sigma_re out, here by
# quadrature over log sigma_re of the Gaussian marginal.
test_that("the SMC evidence integrates a random-effect SD (gcol33/tulpa#1061)", {
  skip_if_not_slow()
  set.seed(11)
  J <- 15; n <- 120; gi <- sample(J, n, TRUE); u <- rnorm(J, 0, 0.7)
  x <- rnorm(n); X <- cbind(1, x); Z <- outer(gi, seq_len(J), "==") * 1
  y <- 1 + .5 * x + u[gi] + rnorm(n, 0, .5)
  phi <- 0.25; sdb <- 2; s_hc <- 2.5
  lmarg <- function(sig) {
    S <- phi * diag(n) + sdb^2 * tcrossprod(X) + sig^2 * tcrossprod(Z)
    R <- chol(S); zz <- backsolve(R, y, transpose = TRUE)
    -0.5 * n * log(2 * pi) - sum(log(diag(R))) - 0.5 * sum(zz^2)
  }
  lg <- seq(-6, 4, length.out = 2001)
  lv <- vapply(exp(lg), lmarg, 0) +
    log(2 / (pi * s_hc)) - log1p((exp(lg) / s_hc)^2) + lg
  m <- max(lv)
  exact <- m + log(sum(exp(lv - m)) * diff(lg)[1])

  re_spec <- list(idx = list(as.integer(gi)), ngroups = J, ncoefs = 1L,
                  correlated = FALSE, Z = list(NULL))
  fits <- lapply(1:6, function(s) suppressWarnings(tulpa_sample_glmm(
    y, NULL, X, "gaussian", "smc", phi = phi, sigma_beta = sdb,
    re_spec = re_spec, sigma_re_scale = s_hc,
    control = list(n_particles = 1000L, seed = s))))
  err <- vapply(fits, function(f) f$log_evidence - exact, 0)
  se <- vapply(fits, function(f) f$log_evidence_se, 0)
  expect_true(all(is.finite(err)))
  # The move count adapts until the resampled copies have decorrelated, so the
  # estimate is centred on the quadrature value and scattered by its own SE:
  # over 20 seeds the mean error is -0.06 (MC SE 0.02) with SD 0.11 against a
  # reported SE of 0.12, where a fixed 5 moves per temperature sat at -0.16
  # with SD 0.25.
  expect_lt(abs(mean(err)), 3 * mean(se) / sqrt(length(err)) + 0.06)
  expect_lt(sd(err), 1.8 * mean(se))
  # Each temperature ran at least one move and no more than the cap.
  expect_true(all(vapply(fits, function(f)
    all(f$n_moves >= 1L & f$n_moves <= 50L), TRUE)))
})

# The move count is a range now, not a number: the old knob is refused rather
# than read as one bound of it.
test_that("the SMC mutation count is bounded by smc_min_moves / smc_max_moves", {
  skip_on_cran()
  set.seed(5)
  n <- 40L; x <- rnorm(n); X <- cbind(1, x); y <- rnorm(n)
  expect_error(tulpa_sample_glmm(y, NULL, X, "gaussian", "smc", phi = 1,
                                 control = list(n_mcmc_steps = 5L)),
               "n_mcmc_steps")
  fit <- tulpa_sample_glmm(y, NULL, X, "gaussian", "smc", phi = 1,
                           control = list(n_particles = 200L, seed = 1L,
                                          smc_min_moves = 3L,
                                          smc_max_moves = 3L))
  expect_true(all(fit$n_moves == 3L))
  expect_error(tulpa_sample_glmm(y, NULL, X, "gaussian", "smc", phi = 1,
                                 control = list(n_particles = 200L,
                                                smc_min_moves = 4L,
                                                smc_max_moves = 2L)),
               "1 <= min <= max")
})

# A block whose prior the sampler evaluates only up to a constant leaves the
# evidence unknown; the fit says which block, and logLik() falls back to the
# mean log posterior.
test_that("the SMC backend declines the evidence where a prior constant is unknown", {
  skip_on_cran()
  set.seed(4)
  Tn <- 40L
  xt <- rnorm(Tn); X <- cbind(1, xt)
  y <- 1 - 0.5 * xt + cumsum(rnorm(Tn, 0, 0.2)) + rnorm(Tn, 0, 0.3)
  tp <- list(type = "rw1", time_idx = seq_len(Tn), n_times = Tn,
             n_groups = 1L, group_idx = NULL, cyclic = FALSE)
  fit <- tulpa_sample_glmm(y, NULL, X, "gaussian", "smc", phi = 0.09,
                           temporal_spec = tp,
                           control = list(n_particles = 200L,
                                          smc_max_moves = 2L, seed = 8L))
  expect_true(is.na(fit$log_evidence))
  expect_identical(fit$log_evidence_declined,
                   "prior_normalizer_not_carried:temporal")
  expect_identical(attr(logLik(fit), "quantity"), "log_posterior_mean")
})
