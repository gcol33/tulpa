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

# Draw a response of one built-in family at a given linear predictor. Shared by
# every simulator below so the response law has one definition.
recov_draw_y <- function(family, eta, ntr, phi) {
  N <- length(eta)
  switch(family,
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
}

# ----- simulate: known beta + region IID RE -> built-in family response -----
sim_re <- function(seed, family, nr, spr, ntr, beta, sigma_u, phi) {
  set.seed(seed)
  N      <- nr * spr
  region <- rep(seq_len(nr), each = spr)
  x      <- rnorm(N)
  X      <- cbind(1, x)
  u      <- rnorm(nr, 0, sigma_u)
  eta    <- as.numeric(X %*% beta) + u[region]
  y <- recov_draw_y(family, eta, ntr, phi)
  list(y = y, X = X, region = as.integer(region), regions = list(as.integer(region)),
       sigma_u = sigma_u, nr = nr, N = N, beta = beta, ntr = ntr)
}

# The same design with `length(su_extra) + 1` CROSSED IID groupings: block 1 is
# the one `recov_sweep` judges (its SD is `sigma_u`), the rest are further
# independent random effects at the supplied SDs. Each block contributes one
# outer hyperparameter axis, so the block count IS the outer dimension `d` --
# the knob the CCD's node placement (and, before gcol33/tulpa#309, the reported
# interval's width) depends on. Returns a simulator with `sim_re`'s signature.
sim_re_crossed <- function(su_extra) {
  function(seed, family, nr, spr, ntr, beta, sigma_u, phi) {
    set.seed(seed)
    N   <- nr * spr
    sus <- c(sigma_u, su_extra)
    grp <- lapply(sus, function(s) sample.int(nr, N, replace = TRUE))
    x   <- rnorm(N)
    X   <- cbind(1, x)
    eta <- as.numeric(X %*% beta)
    for (k in seq_along(sus)) eta <- eta + rnorm(nr, 0, sus[k])[grp[[k]]]
    y <- recov_draw_y(family, eta, ntr, phi)
    list(y = y, X = X, region = grp[[1L]], regions = grp,
         sigma_u = sigma_u, nr = nr, N = N, beta = beta, ntr = ntr)
  }
}

# Marginalized fixed-effect posterior: Gaussian mixture over the grid.
# Returns the mixture mean and a normal interval per coefficient at multiplier
# `z` (1.96, the 95% interval, unless a caller asks for another level).
beta_post <- function(fit, z = 1.96) {
  w  <- fit$weights
  gm <- fit$grid_modes
  gh <- fit$grid_hessians
  # An empty cell would be dropped by rbind() below and silently shift every
  # weight onto the wrong component, so require the retention to be complete.
  stopifnot(length(gm) == length(w), length(gh) == length(w),
            !any(vapply(gm, is.null, logical(1))),
            !any(vapply(gh, is.null, logical(1))))
  p  <- length(gm[[1]])
  mu_k  <- do.call(rbind, gm)                                    # K x p modes
  var_k <- t(vapply(gh, function(H) diag(solve(H)), numeric(p))) # K x p variances
  mean  <- as.numeric(crossprod(w, mu_k))
  sd    <- sqrt(pmax(0, as.numeric(crossprod(w, var_k + mu_k^2)) - mean^2))
  list(mean = mean, lo = mean - z * sd, hi = mean + z * sd)
}

# The two fitters the sweep runs. Both describe the SAME model -- one region
# IID block over the outer RE-SD grid, the same design and response -- so their
# fixed-effect estimates and intervals are each other's reference.
recov_fit_single <- function(d, sg, family, cfg) {
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(cfg$ntr, d$N), X = d$X,
    prior = list(list(type = "iid", obs_idx = d$region,
                      n_units = d$nr, sigma_grid = sg)),
    family = family, phi = cfg$phi,
    control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                   keep_grid_hessians = TRUE, skew_correct = TRUE)))
}

recov_fit_joint <- function(d, sg, family, cfg) {
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = as.numeric(d$y),
                              n_trials = rep(cfg$ntr, d$N), X = d$X,
                              family = family, phi = cfg$phi)),
    prior = list(list(type = "iid", obs_idx = list(d$region),
                      n_units = d$nr, sigma_grid = sg)),
    control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                   diagnose_k = FALSE, skew_correct = TRUE)))
}

# The same joint fitter over every grouping the simulator produced, with the CCD
# outer integrator asked for explicitly (it engages from three transformable
# axes). One iid block per grouping, so `length(d$regions)` is the outer
# dimension. Axis 1 is block 1's SD, which is what `recov_sweep` covers.
recov_fit_joint_ccd <- function(d, sg, family, cfg) {
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = as.numeric(d$y),
                              n_trials = rep(cfg$ntr, d$N), X = d$X,
                              family = family, phi = cfg$phi)),
    prior = lapply(d$regions, function(g)
      list(type = "iid", obs_idx = list(g), n_units = d$nr, sigma_grid = sg)),
    control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                   diagnose_k = FALSE, integration = "ccd")))
}

# The same joint fitter on a deliberately COARSE tensor base grid: every fourth
# level of `sg`, so four levels per axis and 4^4 = 256 cells at four groupings.
# That is the regime local-CCD refinement exists for -- a base grid too coarse to
# resolve the peak, at the latent dimension where making it finer is k^d --
# and every condition `.joint_local_ccd_engage()` and its caller impose is met
# here: four transformable latent axes, a tensor grid (`integration = "grid"`),
# no phi grid, `store_Q` off. `local_ccd` is the ONLY setting that differs
# between the two fits, so a difference in what they report is a difference the
# refinement made (gcol33/tulpa#320).
recov_fit_joint_coarse <- function(d, sg, family, cfg, local_ccd = NULL) {
  sgc <- sg[c(1L, 3L, 5L, 7L)]
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = as.numeric(d$y),
                              n_trials = rep(cfg$ntr, d$N), X = d$X,
                              family = family, phi = cfg$phi)),
    prior = lapply(d$regions, function(g)
      list(type = "iid", obs_idx = list(g), n_units = d$nr, sigma_grid = sgc)),
    control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                   diagnose_k = FALSE, integration = "grid",
                   local_ccd = local_ccd)))
}

recov_fit_joint_local_ccd <- function(d, sg, family, cfg) {
  recov_fit_joint_coarse(d, sg, family, cfg, local_ccd = TRUE)
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
#
# `fit_fn(d, sg, family, cfg)` is the fitter under test. It defaults to the
# single-block nested driver; `recov_fit_joint` below fits the SAME simulated
# data as a one-arm joint model, so the joint tier's fixed-effect intervals are
# judged by this one harness rather than a second copy of it (gcol33/tulpa#305).
# `sim_fn(seed, family, nr, spr, ntr, beta, sigma_u, phi)` is the simulator. It
# defaults to the single-grouping design; `sim_re_crossed()` above supplies the
# same design with several crossed groupings, so the outer dimension can be
# varied while the regime is held fixed.
#
# `z` and `level` are the nominal coverage the two reads are judged at: `z`
# multiplies the mixture SD in `beta_post()`, `level` is what `confint()` is
# asked for. The defaults are the 95% every gate below runs at, `z` carrying the
# 1.96 multiplier those gates were measured with.
recov_sweep <- function(family, cfg, n_seed, seed_off, fit_fn = recov_fit_single,
                        sim_fn = sim_re, z = 1.96, level = 0.95) {
  beta <- cfg$beta
  p    <- length(beta)
  sg   <- exp(seq(log(0.2), log(1.5), length.out = 7))
  est  <- matrix(NA_real_, n_seed, p)
  wid  <- matrix(NA_real_, n_seed, p)
  covb <- integer(p)
  cov_gauss <- integer(p)
  cov_skew  <- integer(p)
  gamma3    <- matrix(NA_real_, n_seed, p)
  s_med <- numeric(n_seed)
  s_wid <- numeric(n_seed)
  s_cov <- 0L
  for (s in seq_len(n_seed)) {
    d <- sim_fn(seed_off + s, family, cfg$nr, cfg$spr, cfg$ntr,
                beta, cfg$su, cfg$phi)
    f <- fit_fn(d, sg, family, cfg)
    bp <- beta_post(f, z = z)
    est[s, ] <- bp$mean
    wid[s, ] <- bp$hi - bp$lo
    for (j in seq_len(p)) {
      if (beta[j] >= bp$lo[j] && beta[j] <= bp$hi[j]) covb[j] <- covb[j] + 1L
    }
    gamma3[s, ] <- f$skew_correction$gamma3
    ci_skew <- confint(f, level = level)
    f$skew_correction$enabled <- FALSE
    ci_gauss <- confint(f, level = level)
    for (j in seq_len(p)) {
      if (beta[j] >= ci_gauss[j, 1] && beta[j] <= ci_gauss[j, 2]) {
        cov_gauss[j] <- cov_gauss[j] + 1L
      }
      if (beta[j] >= ci_skew[j, 1] && beta[j] <= ci_skew[j, 2]) {
        cov_skew[j] <- cov_skew[j] + 1L
      }
    }
    s_med[s] <- f$theta_median[[1]]
    s_wid[s] <- f$theta_ci_hi[[1]] - f$theta_ci_lo[[1]]
    if (cfg$su >= f$theta_ci_lo[[1]] && cfg$su <= f$theta_ci_hi[[1]]) {
      s_cov <- s_cov + 1L
    }
  }
  list(mean = colMeans(est), bias = colMeans(est) - beta, cov = covb,
       width = colMeans(wid),
       cov_gauss = cov_gauss, cov_skew = cov_skew,
       gamma3 = colMeans(gamma3), sigma_bias = mean(s_med) - cfg$su,
       sigma_cov = s_cov, sigma_width = mean(s_wid), n_seed = n_seed)
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

# --- the joint tier's fixed-effect intervals (gcol33/tulpa#305) --------------
#
# Until #305 a joint fit reported NA for every standard error and both bounds,
# so nothing below could be asked of it. The two tests here are the recovery
# standard applied to what it now reports.
#
# The reference is exact rather than approximate: a one-arm joint fit and the
# single-block fit describe the same model, so their marginalized fixed-effect
# posteriors are the same posterior and any disagreement beyond the inner-solve
# tolerance is a defect in one of them.

test_that("a one-arm joint fit reproduces the single-block fixed-effect posterior", {
  skip_on_cran()
  for (fam in c("poisson", "binomial", "gaussian")) {
    cfg <- CFG[[fam]]
    d  <- sim_re(4100L, fam, cfg$nr, cfg$spr, cfg$ntr, cfg$beta, cfg$su, cfg$phi)
    sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
    fs <- recov_fit_single(d, sg, fam, cfg)
    fj <- recov_fit_joint(d, sg, fam, cfg)

    expect_true(is.na(fj$grid_fixed_declined))
    # The joint tier reports real uncertainty at all -- the #305 defect.
    expect_true(all(is.finite(summary(fj)$std.error)))
    expect_true(all(is.finite(confint(fj))))

    # Unnamed: the joint tier prefixes each coefficient with its arm name, the
    # single-block tier uses the design column names. Same numbers, and the
    # numbers are what the two tiers owe each other.
    expect_equal(unname(coef(fj)), unname(coef(fs)), tolerance = 1e-5,
                 label = sprintf("%s joint vs single coef", fam))
    expect_equal(unname(summary(fj)$std.error), unname(summary(fs)$std.error),
                 tolerance = 1e-5,
                 label = sprintf("%s joint vs single SE", fam))
    expect_equal(unname(vcov(fj)), unname(vcov(fs)), tolerance = 1e-5,
                 label = sprintf("%s joint vs single vcov", fam))
  }
})

test_that("a CCD-integrated joint fit's hyperparameter interval covers at the nominal rate", {
  skip_if_not_slow()
  # gcol33/tulpa#309. The block count IS the outer dimension, and a CCD places
  # its axial nodes at 1.1 sqrt(d) whitened SDs, so a summary that reports the
  # design's extent covers 2 Phi(1.1 sqrt(d)) - 1 -- 0.943 at d = 3, 0.972 at
  # d = 4 -- whatever the data. Reading the interval from the design's MOMENTS
  # puts it at nominal, and takes the width off the design.
  cfg <- list(nr = 40L, spr = 30L, ntr = 1L, beta = c(-0.2, 0.7), su = 0.7,
              phi = 0.5)
  n_seed <- 40L
  for (extra in list(c(0.5, 0.35), c(0.5, 0.35, 0.25))) {
    R <- recov_sweep("gaussian", cfg, n_seed = n_seed, seed_off = 6100L,
                     fit_fn = recov_fit_joint_ccd,
                     sim_fn = sim_re_crossed(extra))
    d_axes <- length(extra) + 1L
    p  <- R$sigma_cov / n_seed
    se <- sqrt(0.95 * 0.05 / n_seed)
    expect_lt(abs(p - 0.95), 4 * se,
              label = sprintf("d = %d sigma_1 coverage %.3f", d_axes, p))
    expect_lt(abs(R$sigma_bias), 0.12,
              label = sprintf("d = %d sigma_1 bias", d_axes))
  }
})

test_that("joint fixed-effect intervals cover at the nominal rate", {
  skip_if_not_slow()
  # The same sweep, the same simulated data and the same coverage rule the
  # single-block gate above runs under, with the joint fitter substituted.
  n_seed <- 20L
  for (fam in c("poisson", "binomial")) {
    Rj <- recov_sweep(fam, CFG[[fam]], n_seed = n_seed, seed_off = 4200L,
                      fit_fn = recov_fit_joint)
    expect_lt(max(abs(Rj$bias)), 0.12,
              label = sprintf("%s joint max |beta bias|", fam))
    # Per-coefficient floor at the same 70% the pooled single-block gate uses
    # as its loose per-cell criterion; the mean below is the stable statistic.
    expect_gte(min(Rj$cov / n_seed), 0.70,
               label = sprintf("%s joint per-coefficient coverage", fam))
    expect_gte(mean(Rj$cov / n_seed), 0.85,
               label = sprintf("%s joint mean coverage", fam))
  }
})

# --- locally CCD-refined outer grid, judged by coverage (gcol33/tulpa#320) ---
#
# Four rounds of work on this path (#315-#318) were scored on grid-internal
# metrics -- box widths, mass fractions, design mass. None of them says whether
# the fit's intervals cover. The joint sweep above cannot reach the path at all:
# its one `iid` block gives one latent axis and `.joint_local_ccd_engage()`
# requires four.
#
# The fixture below is that sweep at four crossed groupings, on a coarse
# four-level base grid, run twice on the SAME seeds with the refinement on and
# off. Everything else -- design, response, grid, inner solve -- is shared, so
# the difference between the two tallies is the refinement's.
LCCD_CFG <- list(nr = 40L, spr = 30L, ntr = 1L, beta = c(-0.2, 0.7), su = 0.7,
                 phi = 0.5)
LCCD_SIM <- sim_re_crossed(c(0.5, 0.35, 0.25))

test_that("local-CCD refinement engages on the four-axis joint recovery fixture", {
  skip_on_cran()
  # The engage gate has several conditions and every one of them declines
  # silently, so the settings are not evidence that the refinement ran. What it
  # left on the fit is.
  d  <- LCCD_SIM(8100L, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr, LCCD_CFG$ntr,
                 LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
  sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
  f  <- recov_fit_joint_local_ccd(d, sg, "gaussian", LCCD_CFG)

  expect_false(is.null(f$local_ccd_info))
  expect_gt(f$local_ccd_info$n_design_nodes, 0L)
  expect_gt(f$local_ccd_info$n_cells_refined, 0L)
  # The refined cells are design-weighted and the rest keep their cell mass, so
  # the grid carries both kinds and says so per cell (gcol33/tulpa#311).
  expect_true(all(c("mass", "design") %in% f$weight_kind))
  expect_identical(f$integration, "grid")
  # The per-cell fixed-effect retention survives refinement (gcol33/tulpa#307),
  # which is what `beta_post()` reads the coverage off.
  expect_true(is.na(f$grid_fixed_declined))
  expect_true(all(is.finite(confint(f))))

  # The unrefined twin differs in `local_ccd` alone and leaves nothing.
  f0 <- recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG)
  expect_null(f0$local_ccd_info)
  expect_true(all(f0$weight_kind == "mass"))
})

# MEASURED, 150 seeds x 2 coefficients = 300 trials per level (standard error
# 0.0126), the same simulated data fit twice with the refinement on and off:
#
#                        local_ccd = TRUE   local_ccd = NULL
#   level 0.95  intercept    144/150            144/150
#               slope        120/150            120/150
#   level 0.80  intercept    126/150            127/150
#               slope         93/150             91/150
#   sigma_1 (the fit's own 95% interval)
#                            149/150            150/150
#
# Pooled over both coefficients: 0.8800 against 0.8800 at level 0.95, 0.7300
# against 0.7267 at 0.80. Refinement is worth +1 trial of 300 at 0.80 and 0 at
# 0.95, both far inside the 0.0126 standard error.
#
# What the refinement does move is the WIDTH, in the direction gcol33/tulpa#319
# predicts. The intercept's mean 95% interval goes 0.6523 -> 0.6337, 2.9%
# narrower; the slope's is unchanged to four figures (0.06055 -> 0.06056), being
# a within-group contrast that barely reads the RE-SD grid. A 2.9% narrowing at
# 0.96 coverage predicts about one seed of 150 changing hands at level 0.95 and
# about two at 0.80, which is the size of what was observed. So the measurement
# bounds the asymmetry's calibration cost well under a percentage point rather
# than showing it is zero -- 150 seeds has no power below roughly one seed.
#
# The sigma_1 axis is where the refinement pays: mean interval width 0.2330
# against 1.0590, a coarse four-level grid's weighted quantiles spanning most of
# the 0.2-1.5 range, and posterior-median bias 0.0245 against 0.0592. It sharpens
# the hyperparameter interval more than fourfold and still covers 149/150 against
# a nominal 0.95, so the narrowing is conservative-side slack being removed.
#
# The slope's 120/150 at nominal 0.95 belongs to the FIXTURE, not the refinement:
# four crossed random effects on a four-level grid undercover it identically with
# the refinement off. That is why the gate below is the paired comparison and a
# loose floor rather than a nominal-rate assertion -- the two arms are each
# other's reference, which is the comparison the refinement is responsible for.
test_that("a locally CCD-refined outer grid covers as the grid it refined does", {
  skip_if_not_slow()
  n_seed  <- 40L
  R_on  <- recov_sweep("gaussian", LCCD_CFG, n_seed = n_seed, seed_off = 8100L,
                       fit_fn = recov_fit_joint_local_ccd, sim_fn = LCCD_SIM)
  R_off <- recov_sweep("gaussian", LCCD_CFG, n_seed = n_seed, seed_off = 8100L,
                       fit_fn = recov_fit_joint_coarse, sim_fn = LCCD_SIM)

  n_trial <- n_seed * length(LCCD_CFG$beta)
  p_on    <- sum(R_on$cov)  / n_trial
  p_off   <- sum(R_off$cov) / n_trial
  se      <- sqrt(0.95 * 0.05 / n_trial)

  # The gcol33/tulpa#319 criterion: a refined cell's mass rising against its
  # unrefined siblings' would move weight onto the peak and narrow the reported
  # intervals, so refinement-on covers no worse than the base grid it replaced,
  # on the same seeds.
  expect_lte(abs(p_on - 0.95), abs(p_off - 0.95) + 3 * se)
  expect_lte(max(abs(R_on$cov - R_off$cov)), 2L)
  # Neither arm is grossly miscalibrated, at the same per-coefficient floor the
  # joint gate above uses.
  expect_gte(min(R_on$cov / n_seed), 0.70)
  expect_lt(max(abs(R_on$bias)), 0.12)
  # The refinement sharpens the RE-SD interval fourfold on this grid; that has to
  # come out of slack, so the interval still covers at or above nominal.
  expect_gte(R_on$sigma_cov / n_seed, 0.90)
  expect_lt(R_on$sigma_width, 0.5 * R_off$sigma_width)
  # The refinement's effect is on the width, and it is small: the intervals it
  # reports stay within a tenth of the unrefined grid's.
  expect_lt(max(abs(R_on$width / R_off$width - 1)), 0.10)
})
