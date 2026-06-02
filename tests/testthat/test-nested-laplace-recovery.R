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
recov_sweep <- function(family, cfg, n_seed, seed_off) {
  beta <- cfg$beta
  p    <- length(beta)
  sg   <- exp(seq(log(0.2), log(1.5), length.out = 7))
  est  <- matrix(NA_real_, n_seed, p)
  covb <- integer(p)
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
                     keep_grid_hessians = TRUE)))
    bp <- beta_post(f)
    est[s, ] <- bp$mean
    for (j in seq_len(p)) {
      if (beta[j] >= bp$lo[j] && beta[j] <= bp$hi[j]) covb[j] <- covb[j] + 1L
    }
    s_med[s] <- f$theta_median[[1]]
    if (cfg$su >= f$theta_ci_lo[[1]] && cfg$su <= f$theta_ci_hi[[1]]) {
      s_cov <- s_cov + 1L
    }
  }
  list(mean = colMeans(est), bias = colMeans(est) - beta, cov = covb,
       sigma_bias = mean(s_med) - cfg$su, sigma_cov = s_cov, n_seed = n_seed)
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
  skip_on_cran()
  skip_if_fast()
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
  skip_on_cran()
  skip_if_fast()
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
