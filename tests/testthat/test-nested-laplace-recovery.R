# test-nested-laplace-recovery.R
# Parameter-recovery + CI-coverage tests for the built-in families fit through
# the unified spec-driven nested-Laplace path (dev_notes/plans/clean_migration.md L6).
#
# After L1-L5 every built-in family flows through ONE inner solve: the
# multi-block nested driver -> spec_inner_solve -> builtin_family_spec. The
# equivalence tests (test-laplace-spec-builtin-family.R) prove that solve
# reproduces the family-enum mode bit-for-bit; these tests prove the *method*
# recovers known parameters with calibrated intervals, per the global
# "statistical code needs recovery tests" rule -- shape / dispatch / mode-match
# tests are plumbing, not statistical validation.
#
# Design. A region-grouped IID latent block (the same identified shape as the
# occupancy recovery test) keeps the random effect identified -- several sites
# per region -- while the outer grid integrates the RE standard deviation. The
# fixed-effect posterior is the Gaussian mixture OVER the hyperparameter grid:
# component k is N(grid_modes[[k]], solve(grid_hessians[[k]])) with weight
# w_k = weights[k]. beta is therefore marginalized over the grid (law of total
# variance for the SD), never read off a single plug-in-MAP cell -- the global
# "marginalize derived quantities" discipline applied to the fixed effects.
#
# Coverage gate. The authoritative six-family / N = 20-seed / >= 85%-coverage
# gate runs only with TULPA_SLOW_TESTS = true. It judges coverage in AGGREGATE
# (mean over all family x coefficient cells, plus a loose per-cell floor). A
# hard per-coefficient "17/20" gate would be a mis-designed test: a correctly
# calibrated ~90%-coverage Laplace method fails a 17/20 floor on any one
# coefficient ~13% of the time at N = 20, so across 12 coefficients at least
# one spurious failure is the *expected* outcome (~80%). Pooling 240 trials
# drives the mean-coverage standard error to ~0.02, making the >= 0.85 mean a
# stable gate. This mirrors the occupancy test's mean-coverage criterion.

# ----- simulate: known beta + region IID RE -> built-in family response -----
sim_re <- function(seed, family, nr, spr, ntr, beta, sigma_u, phi) {
  set.seed(seed)
  N      <- nr * spr
  region <- rep(seq_len(nr), each = spr)
  x      <- rnorm(N)
  X      <- cbind(1, x)
  u      <- rnorm(nr, 0, sigma_u)
  eta    <- as.numeric(X %*% beta) + u[region]
  y <- switch(family,
    poisson        = rpois(N, exp(eta)),
    binomial       = rbinom(N, ntr, plogis(eta)),
    gaussian       = eta + rnorm(N, 0, sqrt(phi)),
    neg_binomial_2 = rnbinom(N, mu = exp(eta), size = phi),
    gamma          = rgamma(N, shape = phi, rate = phi / exp(eta)),
    beta           = {
      mu <- plogis(eta)
      pmin(pmax(rbeta(N, mu * phi, (1 - mu) * phi), 1e-4), 1 - 1e-4)
    },
    stop("unhandled family ", family))
  list(y = y, X = X, region = as.integer(region), nr = nr, N = N,
       beta = beta, sigma_u = sigma_u, ntr = ntr)
}

# Marginalized fixed-effect posterior: Gaussian mixture over the grid.
# Returns the mixture mean and the 95% normal interval per coefficient.
beta_post <- function(fit) {
  w  <- fit$weights
  gm <- fit$grid_modes
  gh <- fit$grid_hessians
  p  <- length(gm[[1]])
  mu_k  <- do.call(rbind, gm)                                    # K x p modes
  var_k <- t(vapply(gh, function(H) diag(solve(H)), numeric(p))) # K x p variances
  mean  <- as.numeric(crossprod(w, mu_k))
  sd    <- sqrt(pmax(0, as.numeric(crossprod(w, var_k + mu_k^2)) - mean^2))
  list(mean = mean, lo = mean - 1.96 * sd, hi = mean + 1.96 * sd)
}

# Fit n_seed data sets for one family; return per-coefficient mean estimate,
# bias and 95%-CI coverage counts, plus the RE-SD recovery summary.
#
# It also records the SAME coverage read off the fit's own reporting path
# (confint()), twice: once with the inner-Laplace skew correction on and once
# with it off (gcol33/tulpa#302). Both come from ONE fit per seed -- the
# correction is post-processing on the reported quantiles and changes nothing
# the solve produced -- so that comparison is paired and carries no fit-to-fit
# noise. `beta_post` above is untouched and remains what the coverage gates
# below judge.
recov_sweep <- function(family, cfg, n_seed, seed_off) {
  beta <- cfg$beta
  p    <- length(beta)
  sg   <- exp(seq(log(0.2), log(1.5), length.out = 7))
  est  <- matrix(NA_real_, n_seed, p)
  covb <- integer(p)
  cov_gauss <- integer(p)
  cov_skew  <- integer(p)
  gamma3    <- matrix(NA_real_, n_seed, p)
  s_med <- numeric(n_seed)
  s_cov <- 0L
  for (s in seq_len(n_seed)) {
    d <- sim_re(seed_off + s, family, cfg$nr, cfg$spr, cfg$ntr,
                beta, cfg$su, cfg$phi)
    prior <- list(list(type = "iid", obs_idx = d$region,
                       n_units = d$nr, sigma_grid = sg))
    f <- suppressWarnings(tulpa_nested_laplace(
      y = d$y, n_trials = rep(cfg$ntr, d$N), X = d$X,
      prior = prior, family = family, phi = cfg$phi,
      control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                     keep_grid_hessians = TRUE, skew_correct = TRUE)))
    bp <- beta_post(f)
    est[s, ] <- bp$mean
    for (j in seq_len(p)) {
      if (beta[j] >= bp$lo[j] && beta[j] <= bp$hi[j]) covb[j] <- covb[j] + 1L
    }
    gamma3[s, ] <- f$skew_correction$gamma3
    ci_skew <- confint(f, level = 0.95)
    f$skew_correction$enabled <- FALSE
    ci_gauss <- confint(f, level = 0.95)
    for (j in seq_len(p)) {
      if (beta[j] >= ci_gauss[j, 1] && beta[j] <= ci_gauss[j, 2]) {
        cov_gauss[j] <- cov_gauss[j] + 1L
      }
      if (beta[j] >= ci_skew[j, 1] && beta[j] <= ci_skew[j, 2]) {
        cov_skew[j] <- cov_skew[j] + 1L
      }
    }
    s_med[s] <- f$theta_median[[1]]
    if (cfg$su >= f$theta_ci_lo[[1]] && cfg$su <= f$theta_ci_hi[[1]]) {
      s_cov <- s_cov + 1L
    }
  }
  list(mean = colMeans(est), bias = colMeans(est) - beta, cov = covb,
       cov_gauss = cov_gauss, cov_skew = cov_skew,
       gamma3 = colMeans(gamma3), sigma_bias = mean(s_med) - cfg$su,
       sigma_cov = s_cov, n_seed = n_seed)
}

# Per-family identified regimes (RE SD recoverable, link well-determined).
# phi (dispersion / residual SD^2 / shape) is supplied at its true value -- the
# single-block nested driver takes phi as fixed; this is a beta + RE-SD recovery.
CFG <- list(
  poisson        = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c( 0.3, 0.6), su = 0.5, phi = 1.0),
  binomial       = list(nr = 60L, spr = 10L, ntr = 10L, beta = c(-0.2, 0.7), su = 0.5, phi = 1.0),
  gaussian       = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c(-0.2, 0.7), su = 0.7, phi = 0.5),
  neg_binomial_2 = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c( 0.3, 0.6), su = 0.5, phi = 3.0),
  gamma          = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c( 0.3, 0.6), su = 0.5, phi = 3.0),
  beta           = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c( 0.0, 0.6), su = 0.5, phi = 6.0)
)

test_that("poisson + binomial: beta recovers with calibrated intervals through the unified path", {
  skip_if_not_slow()
  for (fam in c("poisson", "binomial")) {
    R <- recov_sweep(fam, CFG[[fam]], n_seed = 12L, seed_off = 2000L)
    # Point recovery: the marginalized posterior mean tracks every true beta.
    expect_lt(max(abs(R$bias)), 0.12,
              label = sprintf("%s max |beta bias|", fam))
    # Intervals are not grossly miscalibrated (>= 9/12 per coefficient). The
    # strict >= 85% @ N = 20 standard is the TULPA_SLOW_TESTS gate below.
    for (j in seq_along(R$cov)) {
      expect_gte(R$cov[j], 9L,
                 label = sprintf("%s beta[%d] coverage (of %d)", fam, j, R$n_seed))
    }
    # The integrated RE standard deviation recovers and its interval covers.
    expect_lt(abs(R$sigma_bias), 0.10, label = sprintf("%s sigma_u bias", fam))
    expect_gte(R$sigma_cov, 10L, label = sprintf("%s sigma_u coverage", fam))
  }
})

test_that("all built-in families: beta recovery and >= 85% aggregate coverage (slow gate)", {
  skip_if_not_slow()
  if (!isTRUE(as.logical(Sys.getenv("TULPA_SLOW_TESTS", "false")))) {
    skip("Slow 20-seed x 6-family recovery gate. Set TULPA_SLOW_TESTS=true to run.")
  }

  cov_rate  <- numeric(0)   # one entry per family x coefficient
  bias_max  <- 0
  s_cov_rate <- numeric(0)  # one entry per family (RE-SD coverage)
  for (fam in names(CFG)) {
    R <- recov_sweep(fam, CFG[[fam]], n_seed = 20L, seed_off = 3000L)
    cov_rate   <- c(cov_rate, R$cov / R$n_seed)
    s_cov_rate <- c(s_cov_rate, R$sigma_cov / R$n_seed)
    bias_max   <- max(bias_max, max(abs(R$bias)))
  }

  # Aggregate beta coverage at the nominal 95% target (pooled over all
  # coefficients -- the stable headline), with no single family grossly
  # miscalibrated.
  expect_gte(mean(cov_rate), 0.85)
  expect_gte(min(cov_rate), 0.70)
  # Low point bias across every family and coefficient.
  expect_lt(bias_max, 0.12)
  # RE standard deviation covers across families.
  expect_gte(mean(s_cov_rate), 0.85)
})

# --- the inner-Laplace skew correction, judged by coverage (gcol33/tulpa#302) --
#
# A small-group Bernoulli random-effect design with rare events: N = 48 over 12
# groups of 4, intercept -2.5. That is the known-skewed case -- the fitted
# gamma_3 averages -0.695 on the intercept and +0.306 on the slope over 200
# seeds, both inside the band the correction is gated to. The six
# configurations above are not: the second test below measures their gamma_3
# and holds the correction to leaving their coverage where it was.
#
# MEASURED, 200 seeds x 2 coefficients = 400 trials (standard error ~0.011),
# both intervals read off the same fits:
#
#   level 0.95   Gaussian 0.9650   corrected 0.9600   (nominal 0.95)
#   level 0.80   Gaussian 0.8050   corrected 0.8075   (nominal 0.80)
#   level 0.50   Gaussian 0.4950   corrected 0.5000   (nominal 0.50)
#
# The correction is directionally right and immaterial at this size: every
# difference is within one standard error. That is the reason
# `.NL_DIAG$skew_correct` is FALSE by default. Where it IS unambiguous is
# against the exact posterior quantiles, with the Laplace point bias and the
# grid-mixture inflation taken out of the comparison --
# test-inner-skew-correction.R measures a 44% reduction in absolute endpoint
# error there. Two reasons the coverage gain is smaller than that: gamma_3 is a
# lower bound on the true skewness, and the correction is skewness-only, so a
# biased Laplace mode stays biased.
#
# The gate below is therefore the acceptance criterion and not more: no worse
# than the Gaussian intervals, within the sampling error of the seed count run.
SKEW_CFG <- list(nr = 12L, spr = 4L, ntr = 1L, beta = c(-2.5, 0.8),
                 su = 0.7, phi = 1.0)

test_that("skew-corrected intervals cover no worse than Gaussian ones on a skewed fixture", {
  skip_if_not_slow()
  n_seed <- 60L
  R <- recov_sweep("binomial", SKEW_CFG, n_seed = n_seed, seed_off = 7000L)

  # The fixture is genuinely skewed and inside the band, so the correction has
  # something to do here rather than being trivially inert.
  expect_gt(max(abs(R$gamma3)), 0.25)
  expect_lt(max(abs(R$gamma3)), .nl_diag("gamma3_unreliable"))

  n_trial <- n_seed * length(SKEW_CFG$beta)
  cov_g <- sum(R$cov_gauss) / n_trial
  cov_c <- sum(R$cov_skew)  / n_trial
  se <- sqrt(0.95 * 0.05 / n_trial)
  expect_lte(abs(cov_c - 0.95), abs(cov_g - 0.95) + 3 * se)
})

test_that("the correction leaves a near-symmetric family's coverage where it was", {
  skip_if_not_slow()
  # The six-family gate above sits far inside the `good` band, so turning the
  # correction on there may not move coverage. A gaussian response is the
  # sharpest case: its log-likelihood is exactly quadratic in eta, gamma_3 is
  # identically 0, and the two intervals must be the same interval.
  Rg <- recov_sweep("gaussian", CFG[["gaussian"]], n_seed = 12L, seed_off = 2000L)
  expect_identical(max(abs(Rg$gamma3)), 0)
  expect_identical(Rg$cov_skew, Rg$cov_gauss)

  # A poisson response is barely skewed at these counts; the shift is a small
  # fraction of a standard error, so coverage moves by at most one seed.
  Rp <- recov_sweep("poisson", CFG[["poisson"]], n_seed = 12L, seed_off = 2000L)
  expect_lt(max(abs(Rp$gamma3)), 0.1)
  expect_lte(max(abs(Rp$cov_skew - Rp$cov_gauss)), 1L)
})
