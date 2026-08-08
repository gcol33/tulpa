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
#
# `phi` is ALWAYS the parameterization the fitters' own door takes, which for
# every fitter in this file is a DIRECT door (`tulpa_nested_laplace()`,
# `tulpa_nested_laplace_joint()`) and therefore the kernel convention: for
# gaussian and lognormal that is the residual SD, variance `phi^2`
# (`R/nested_laplace_joint.R`, `@param phi_grid`). `tulpa()` is the door that
# takes the residual VARIANCE and converts at its boundary with
# `.phi_to_kernel()`; nothing in this file goes through it.
#
# So the draw applies NO conversion of its own: `cfg$phi` reaches the simulator
# and the fitter as the same number meaning the same thing, and a crossing
# between the two conventions has nowhere to enter (gcol33/tulpa#332, which was
# a `sqrt(phi)` here fitting against a `phi` at the door -- half the residual
# variance, and a slope interval narrow by sqrt(2)). The convention itself is
# pinned against the engine by the test at the end of this file, so a change on
# the door's side fails loudly here instead of silently rescaling every gaussian
# fixture.
recov_draw_y <- function(family, eta, ntr, phi) {
  N <- length(eta)
  switch(family,
    poisson        = rpois(N, exp(eta)),
    binomial       = rbinom(N, ntr, plogis(eta)),
    gaussian       = eta + rnorm(N, 0, phi),
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

# A per-fit diagnostic read to a fixed length. A backend that declined the
# diagnostic attaches nothing, and a shorter vector than the coefficient block
# leaves the tail NA -- "not scored", never 0.
.recov_pad <- function(x, n) {
  out <- rep(NA_real_, n)
  if (length(x)) out[seq_len(min(n, length(x)))] <- as.numeric(x)[seq_len(min(n, length(x)))]
  out
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
#
# `levels` selects the base-grid resolution off `sg`: four levels by default,
# and `sg` itself at seven. `phi` is the arm's residual scale, which the joint
# door reads as the gaussian residual SD (variance `phi^2`) -- passing something
# other than `cfg$phi` is what lets a fit be scored against a twin whose
# assumed residual scale is the simulator's.
recov_fit_joint_coarse <- function(d, sg, family, cfg, local_ccd = NULL,
                                   phi = cfg$phi, levels = 4L,
                                   diagnose_k = FALSE) {
  sgc <- if (levels >= length(sg)) sg else
    exp(seq(log(min(sg)), log(max(sg)), length.out = levels))
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = as.numeric(d$y),
                              n_trials = rep(cfg$ntr, d$N), X = d$X,
                              family = family, phi = phi)),
    prior = lapply(d$regions, function(g)
      list(type = "iid", obs_idx = list(g), n_units = d$nr, sigma_grid = sgc)),
    control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                   diagnose_k = diagnose_k, integration = "grid",
                   local_ccd = local_ccd, skew_correct = TRUE)))
}

recov_fit_joint_local_ccd <- function(d, sg, family, cfg) {
  recov_fit_joint_coarse(d, sg, family, cfg, local_ccd = TRUE)
}

# The same coarse fit with the two residual-scale conventions crossed: the arm
# is handed `cfg$phi^2`, the residual VARIANCE, where the door reads an SD. That
# is exactly the gcol33/tulpa#332 defect, on data the corrected fixture also
# simulates, so it is the negative control the coverage gate is scored against
# -- a fit at the wrong residual scale must fail the gate the corrected one
# passes, or the gate is not sensitive to the thing it grades.
recov_fit_joint_phi_crossed <- function(d, sg, family, cfg) {
  recov_fit_joint_coarse(d, sg, family, cfg, phi = cfg$phi^2)
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
# Alongside the aggregates it keeps the PER-SEED reads: `cov_seed` / `width_seed`
# (which seeds missed and how wide their intervals were), `gamma3_seed`,
# `inner_k_seed` / `inner_k_ess_seed` and `outer_k_seed` (the inner and outer
# reliability scores at the same probed indices), and `ci_skew_gap` (how far the
# skew-corrected endpoints moved). An aggregate cannot say WHICH seeds a
# diagnostic was bad on, and relating the two is what attributing a coverage
# deficit to a layer takes (gcol33/tulpa#325).
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
  cvs  <- matrix(NA_integer_, n_seed, p)
  covb <- integer(p)
  cov_gauss <- integer(p)
  cov_skew  <- integer(p)
  cov_mix   <- integer(p)
  wid_mix   <- matrix(NA_real_, n_seed, p)
  wid_gauss <- matrix(NA_real_, n_seed, p)
  gamma3    <- matrix(NA_real_, n_seed, p)
  inner_k     <- matrix(NA_real_, n_seed, p)
  inner_k_ess <- matrix(NA_real_, n_seed, p)
  ci_skew_gap <- numeric(n_seed)
  outer_k     <- numeric(n_seed)
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
      cvs[s, j] <- as.integer(beta[j] >= bp$lo[j] && beta[j] <= bp$hi[j])
      covb[j] <- covb[j] + cvs[s, j]
    }
    gamma3[s, ] <- f$skew_correction$gamma3
    # The two inner-layer scores at the SAME probed indices gamma_3 is read at
    # (the fixed-effects coefficients), plus the outer one, per seed rather than
    # averaged -- which seed missed is only relatable to a diagnostic that was
    # kept per seed (gcol33/tulpa#325). A backend that declined leaves NA.
    inner_k[s, ]     <- .recov_pad(f$inner_pareto_k, p)
    inner_k_ess[s, ] <- .recov_pad(f$inner_pareto_k_rel_ess, p)
    outer_k[s]       <- .recov_pad(f$pareto_k, 1L)
    # Three reads of the SAME fit, so the arms are paired by construction and
    # differ only in how the marginal posterior is turned into an interval:
    #   ci_gauss  the collapsed Gaussian, `mu +/- z sigma` on the marginalized
    #             moments -- the pre-gcol33/tulpa#336 report, and the baseline
    #             the #302 correction is judged against
    #   ci_mix    the quantiles of the Gaussian mixture the grid defines, which
    #             carries the same two moments and the shape as well (#336)
    #   ci_skew   the #302 Cornish-Fisher read at the MAP-cell gamma_3
    # The moments behind all three are identical, so any difference in coverage
    # or width is attributable to the marginal read and to nothing else.
    ci_skew <- confint(f, level = level)
    f$skew_correction$enabled <- FALSE
    ci_mix <- confint(f, level = level)
    # The collapsed arm goes through the engine's own quantile helper at a
    # disabled correction, which is bit-for-bit the interval `confint()` itself
    # returned before #336. Recomputing `mu +/- z sigma` in R instead would be
    # the same arithmetic in a different summation order, and the #302 gate
    # below asserts the corrected and collapsed reads agree EXACTLY at
    # gamma_3 = 0 -- an exact zero that a re-derivation turns into 1e-14.
    mom_f <- .nested_fixed_moments(f)
    a_lvl <- (1 - level) / 2
    ci_gauss <- .nl_skew_marginal(mom_f$mean[seq_len(p)],
                                  sqrt(pmax(diag(mom_f$cov)[seq_len(p)], 0)),
                                  rep(NA_real_, p), c(a_lvl, 1 - a_lvl),
                                  enabled = FALSE)$q
    ci_skew_gap[s] <- max(abs(ci_skew - ci_gauss))
    wid_mix[s, ]   <- ci_mix[, 2] - ci_mix[, 1]
    wid_gauss[s, ] <- ci_gauss[, 2] - ci_gauss[, 1]
    for (j in seq_len(p)) {
      if (beta[j] >= ci_gauss[j, 1] && beta[j] <= ci_gauss[j, 2]) {
        cov_gauss[j] <- cov_gauss[j] + 1L
      }
      if (beta[j] >= ci_skew[j, 1] && beta[j] <= ci_skew[j, 2]) {
        cov_skew[j] <- cov_skew[j] + 1L
      }
      if (beta[j] >= ci_mix[j, 1] && beta[j] <= ci_mix[j, 2]) {
        cov_mix[j] <- cov_mix[j] + 1L
      }
    }
    s_med[s] <- f$theta_median[[1]]
    s_wid[s] <- f$theta_ci_hi[[1]] - f$theta_ci_lo[[1]]
    if (cfg$su >= f$theta_ci_lo[[1]] && cfg$su <= f$theta_ci_hi[[1]]) {
      s_cov <- s_cov + 1L
    }
  }
  list(mean = colMeans(est), bias = colMeans(est) - beta, cov = covb,
       width = colMeans(wid), width_seed = wid, cov_seed = cvs,
       cov_gauss = cov_gauss, cov_skew = cov_skew, cov_mix = cov_mix,
       width_mix = colMeans(wid_mix), width_gauss = colMeans(wid_gauss),
       width_mix_seed = wid_mix, width_gauss_seed = wid_gauss,
       ci_skew_gap = ci_skew_gap,
       gamma3 = colMeans(gamma3), gamma3_seed = gamma3,
       inner_k_seed = inner_k, inner_k_ess_seed = inner_k_ess,
       outer_k_seed = outer_k,
       sigma_bias = mean(s_med) - cfg$su,
       sigma_cov = s_cov, sigma_width = mean(s_wid), n_seed = n_seed)
}

# Per-family identified regimes (RE SD recoverable, link well-determined).
# phi is supplied at its true value in the DIRECT door's own parameterization --
# gaussian residual SD, negbin dispersion, gamma / beta shape -- the one
# `recov_draw_y()` draws from unchanged. The nested driver takes phi as fixed;
# this is a beta + RE-SD recovery.
#
# `RESID_SD` is the gaussian residual SD every gaussian fixture in this file
# runs at, written as the root of the variance it corresponds to so the number
# says which quantity it is. 0.5 is the residual VARIANCE these fixtures have
# always simulated at; before gcol33/tulpa#332 the same 0.5 was handed to the
# door, which reads an SD, and every gaussian fit assumed a quarter.
RESID_SD <- sqrt(0.5)

CFG <- list(
  poisson        = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c( 0.3, 0.6), su = 0.5, phi = 1.0),
  binomial       = list(nr = 60L, spr = 10L, ntr = 10L, beta = c(-0.2, 0.7), su = 0.5, phi = 1.0),
  gaussian       = list(nr = 60L, spr = 10L, ntr = 1L,  beta = c(-0.2, 0.7), su = 0.7, phi = RESID_SD),
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

# --- the mixture read of the fixed effects (gcol33/tulpa#336) ----------------
#
# The grid defines a Gaussian mixture per coefficient. Collapsing it to one
# Gaussian keeps the mean and the variance exactly and throws the shape away,
# which only the quantiles ever see. So the question this asks is narrow: does
# retaining that shape improve or preserve interval calibration relative to
# discarding it? Both arms come off ONE solve per seed and carry identical
# moments, so any difference is the marginal read and nothing else.
#
# The acceptance criterion is "no worse", the same standard #302 was held to.
# The mixture read is not a competing estimator -- the collapsed Gaussian is a
# lossy compression of a posterior the engine has already computed -- so the
# gate is against a coverage regression, not for a coverage gain.

test_that("the mixture read covers no worse than the collapsed Gaussian", {
  skip_if_not_slow()
  n_seed <- 60L
  for (fam in c("binomial", "poisson")) {
    R <- recov_sweep(fam, CFG[[fam]], n_seed = n_seed, seed_off = 5100L)
    n_trial <- n_seed * length(CFG[[fam]]$beta)
    cov_m <- sum(R$cov_mix)   / n_trial
    cov_g <- sum(R$cov_gauss) / n_trial
    se <- sqrt(0.95 * 0.05 / n_trial)
    expect_lte(abs(cov_m - 0.95), abs(cov_g - 0.95) + 3 * se,
               label = sprintf("%s mixture vs collapsed coverage", fam))
  }
})

test_that("a skewed-hyperparameter fixture is where the two reads part", {
  skip_if_not_slow()
  # SKEW_CFG has 12 groups of 4 at a rare-event intercept, so the RE-SD
  # posterior is genuinely skewed and the grid spreads over it -- the regime
  # where the mixture is least like the Gaussian matching its moments.
  R <- recov_sweep("binomial", SKEW_CFG, n_seed = 60L, seed_off = 7300L)
  # The two reads differ per seed, so this fixture is discriminating rather
  # than one where the mixture collapses back onto its own moments.
  expect_gt(max(abs(R$width_mix_seed - R$width_gauss_seed)), 1e-6)

  n_trial <- 60L * length(SKEW_CFG$beta)
  se <- sqrt(0.95 * 0.05 / n_trial)
  expect_lte(abs(sum(R$cov_mix) / n_trial - 0.95),
             abs(sum(R$cov_gauss) / n_trial - 0.95) + 3 * se)
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
              phi = RESID_SD)
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
                 phi = RESID_SD)
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
#               slope        146/150            146/150
#   level 0.80  intercept    125/150            126/150
#               slope        115/150            116/150
#   sigma_1 (the fit's own 95% interval)
#                            150/150            150/150
#
# Pooled over both coefficients: 0.9667 against 0.9667 at level 0.95, 0.8000
# against 0.8067 at 0.80. Refinement is worth 0 trials of 300 at 0.95 and -1 at
# 0.80, both far inside the 0.0126 standard error, and both levels sit at or
# above nominal on both arms.
#
# What the refinement does move is the WIDTH, in the direction gcol33/tulpa#319
# predicts. The intercept's mean 95% interval goes 0.63767 -> 0.63177, 0.9%
# narrower; the slope's is unchanged to four figures (0.08525 -> 0.08527), being
# a within-group contrast that barely reads the RE-SD grid. A 0.9% narrowing at
# 0.96 coverage predicts well under one seed of 150 changing hands, which is what
# was observed. So the measurement bounds the asymmetry's calibration cost well
# under a percentage point rather than showing it is zero -- 150 seeds has no
# power below roughly one seed.
#
# The sigma_1 axis is where the refinement pays: mean interval width 0.2399
# against 1.0591, a coarse four-level grid's weighted quantiles spanning most of
# the 0.2-1.5 range, and posterior-median bias 0.0350 against 0.0546. It sharpens
# the hyperparameter interval more than fourfold and still covers 150/150 against
# a nominal 0.95, so the narrowing is conservative-side slack being removed.
#
# These are the rates after gcol33/tulpa#332. Before it the same data were fitted
# at a quarter of their residual variance and the slope covered 120/150 and
# 91/150 at the two levels, which is the table this block used to carry; the
# `recov_fit_joint_phi_crossed()` arm reproduces it exactly and the block after
# the gate is what attributed it.
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
  # The per-coefficient floor, three binomial standard errors below the weaker of
  # the two rates measured at 150 seeds in the block above. The intercept covers
  # 0.9600 and the slope 0.9733 there; at this gate's own 40 seeds those carry
  # standard errors 0.0310 and 0.0255, putting the two bounds at 0.8670 and
  # 0.8969, so 0.85 clears both. It replaces a 0.70 that had been set beneath the
  # gcol33/tulpa#332 residual-scale defect, which is the shape of gate that cannot
  # fail for the reason it was built to catch (gcol33/tulpa#325).
  #
  # What 0.85 now protects against is a coefficient losing about eleven points of
  # coverage. That is calibrated on the defect it replaced: the crossing cost the
  # slope 0.9733 -> 0.8000, and a 40-seed draw at 0.8000 falls below 0.85 with
  # probability 0.714, against 0.0010 and 0.0001 of a spurious failure at the two
  # rates actually measured.
  expect_gte(min(R_on$cov / n_seed), 0.85)
  expect_lt(max(abs(R_on$bias)), 0.12)
  # The refinement sharpens the RE-SD interval fourfold on this grid; that has to
  # come out of slack, so the interval still covers at or above nominal.
  expect_gte(R_on$sigma_cov / n_seed, 0.90)
  expect_lt(R_on$sigma_width, 0.5 * R_off$sigma_width)
  # The refinement's effect is on the width, and it is small: the intervals it
  # reports stay within a tenth of the unrefined grid's.
  expect_lt(max(abs(R_on$width / R_off$width - 1)), 0.10)
})

# --- how the slope's 120/150 was attributed (gcol33/tulpa#325) ---------------
#
# The table above is the corrected one. Before gcol33/tulpa#332 the slope covered
# 120/150 at nominal 0.95, identically with the refinement on and off, while the
# intercept covered 144/150 and sigma_1 149/150 -- so the paired design placed
# the whole pooled deficit on the fixture rather than on the refinement, and
# said nothing about which part of the fixture. What follows is how that was
# narrowed to the residual scale, because the ladder is reusable and the answer
# is not: the same three perturbations distinguish a coefficient that reads the
# outer grid from one that does not, on any fixture.
#
# Every number below is 150 seeds of the CORRECTED fixture unless it says
# otherwise, so the ladder can be re-run against what ships.
#
# THE OUTER GRID, three perturbations, each large:
#
#                                    sigma_1 mean width   slope width   slope cov
#   four-level base grid (shipped)         1.0591          0.08525       146/150
#   the same grid, locally refined         0.2399          0.08527       146/150
#   six-level base grid, 1296 cells        0.6391          0.08527       146/150
#
# The six-level grid is the finest tensor base reachable at four axes under the
# joint driver's 2048-cell cap, and it is a materially better integration -- the
# sigma_1 posterior-median bias goes +0.0546 -> -0.0097 and its interval narrows
# on every one of the 150 seeds (width ratio at most 0.9334, median 0.5098). The
# INTERCEPT reads that: its width ratio runs 0.8573 to 1.2282, median 0.9460. The
# slope's runs 0.9934 to 1.0041, median 1.0005, and not one of its 150 trials
# changes hands on any of the three.
#
# A fourth perturbation runs offline at zero fits, through the gcol33/tulpa#322
# dump and the gcol33/tulpa#329 fixed-effect rebuild: the same 150 unrefined fits
# reweighted under gcol33/tulpa#326's local-quadratic box-integral cell mass,
# entering as a per-cell design multiplier. It redistributes 0.3890 of the total
# outer weight on average (median 0.2806, max 0.9489 in total variation) and
# moves the hyperparameter median above what the grid resolves on 133 of 150
# seeds (0.0467 against a 0.0136 coarsening floor). It moves the INTERCEPT's
# standard error 0.162671 -> 0.154063, and per seed by 2.38% at the median and up
# to 25.4%; its coverage goes 144 -> 143. It moves the SLOPE's 0.021747 ->
# 0.021767, per seed 0.088% at the median and never more than 0.637%; its
# coverage goes 146 -> 146. On the pre-fix fits the same rule moved 0.6746 of the
# weight, the intercept's standard error 13.4%, and the slope's coverage from
# 120/150 to 120/150 -- not one trial, under a rule rewriting two thirds of the
# outer mass. That is what exonerated the outer layer by measurement rather than
# by the width argument alone.
#
# THE INNER LAPLACE never had a case to answer, and structurally so: a gaussian
# log-likelihood is exactly quadratic in eta, so the inner Gaussian IS the
# conditional posterior. gamma_3 is 0 at both coefficients on every one of the
# 300 fits, and every probed index's realized importance efficiency is 1.000000
# -- the materiality gate `inner_pareto_k`'s shape is banded behind, so its
# values (0.141 median at the intercept, -0.122 at the slope) describe a proposal
# with nothing to correct. gcol33/tulpa#302's correction consumes gamma_3, so it
# consumes nothing: corrected and Gaussian endpoints agree to 0.000e+00 on every
# fit, and both cover 290/300.
#
# The one diagnostic that is NOT clean points away from the slope. The outer
# k-hat on the unrefined four-level grid is 0.752 median and >= 0.7 on 92 of 150
# seeds; refinement takes it to 0.440 median and 24 of 150. That is the
# diagnostic reading the coarse grid's Gaussian proposal correctly, on the axis
# whose interval refinement sharpens fourfold -- an axis the slope does not read.
#
# WHAT THE SLOPE READS is the residual scale. `recov_draw_y()` used to draw
# `rnorm(N, 0, sqrt(phi))`, the residual-VARIANCE convention `tulpa()` takes,
# while every fitter here goes to a direct door reading `phi` as the residual SD
# -- so a gaussian arm was fitted at a quarter of the variance it simulated, and
# every interval was narrow by sqrt(2) in whatever part of its variance the
# residual carries. For the intercept that part is under one percent, its
# posterior variance being the integrated RE-SD grid's: crossing the conventions
# moves its width 0.63767 -> 0.65226, 2.2%, and its coverage 144/150 -> 144/150.
# For the slope, a within-group contrast, it is all of it: 0.08525 -> 0.06055, a
# ratio of 1.4079 against sqrt(2) = 1.4142, and 146/150 -> 120/150. The narrowing
# alone predicts 2 Phi(1.96 / sqrt(2)) - 1 = 0.8342 against the 0.8000 measured;
# the remainder is the crossed random effects' own projection onto x, which a fit
# at the wrong residual scale also understates.
#
# `recov_fit_joint_phi_crossed()` is that crossing, kept as the negative control:
# it reproduces the pre-fix table exactly on the same data, so the gate's floor
# is scored against a defect that can still be run rather than against one that
# only exists in a comment.
test_that("the coverage gate reads the residual scale and not a layer of the fit", {
  skip_if_not_slow()
  n_seed <- 40L
  R_ok <- recov_sweep("gaussian", LCCD_CFG, n_seed = n_seed, seed_off = 8100L,
                      fit_fn = recov_fit_joint_coarse, sim_fn = LCCD_SIM)
  R_x  <- recov_sweep("gaussian", LCCD_CFG, n_seed = n_seed, seed_off = 8100L,
                      fit_fn = recov_fit_joint_phi_crossed, sim_fn = LCCD_SIM)

  # The inner layer is exact here, so it cannot own a calibration deficit: the
  # cubic term is identically zero and every probed index's importance
  # efficiency clears the materiality gate its k-hat is banded behind.
  expect_true(all(R_ok$gamma3_seed == 0))
  expect_gte(min(R_ok$inner_k_ess_seed), .nl_diag("inner_k_material_ess"))
  # So the correction that consumes gamma_3 has nothing to consume, and the two
  # reads are the same interval rather than merely a close one.
  expect_identical(max(R_ok$ci_skew_gap), 0)
  expect_identical(R_ok$cov_skew, R_ok$cov_gauss)

  # Crossing the residual-scale conventions narrows the slope's interval by
  # sqrt(2) and costs it coverage, and leaves the intercept's -- carried by the
  # RE-SD grid -- where it was. That pair is the attribution, and it is also the
  # sensitivity check on the gate above: the floor it now carries is one a fit at
  # the wrong residual scale fails.
  expect_equal(R_ok$width[2] / R_x$width[2], sqrt(2), tolerance = 0.02)
  expect_equal(R_ok$width[1] / R_x$width[1], 1, tolerance = 0.05)
  se <- sqrt(0.95 * 0.05 / n_seed)
  expect_lt(abs(R_ok$cov[2] / n_seed - 0.95), 3 * se)
  expect_lt(R_x$cov[2], R_ok$cov[2])
  expect_identical(R_x$cov[1], R_ok$cov[1])

  # Nor does the slope read the base grid's resolution. At six levels per axis
  # the sigma_1 interval sharpens on every one of the 150 seeds above and the
  # intercept's width follows it, while the slope's stays inside five parts in a
  # thousand of the four-level grid's -- across those seeds it never left
  # [0.9934, 1.0041].
  d  <- LCCD_SIM(8130L, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr, LCCD_CFG$ntr,
                 LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
  sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
  f4 <- recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG)
  f6 <- recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG, levels = 6L)
  expect_identical(nrow(f6$theta_grid), 1296L)
  expect_lt(f6$theta_ci_hi[[1]] - f6$theta_ci_lo[[1]],
            0.6 * (f4$theta_ci_hi[[1]] - f4$theta_ci_lo[[1]]))
  w4 <- beta_post(f4)
  w6 <- beta_post(f6)
  expect_lt(abs((w6$hi[2] - w6$lo[2]) / (w4$hi[2] - w4$lo[2]) - 1), 5e-3)

  # Nor the outer weight rule. The same fit re-read under gcol33/tulpa#326's
  # box-integral cell mass, through the gcol33/tulpa#322 dump and the
  # gcol33/tulpa#329 fixed-effect rebuild at no further fitting: on this seed the
  # rule rewrites 0.9184 of the outer mass and moves the intercept's standard
  # error 25.4%, the slope's 0.085%.
  dm <- outer_grid_dump(f4)
  bm <- .joint_local_ccd_box_mass(dm$joint_grid, dm$log_marginal,
                                  dm$axis_names, dm$axis_tags)
  w_box <- outer_grid_weights(dm, dnode = exp(bm$log_box_ratio))
  expect_gt(0.5 * sum(abs(w_box / sum(w_box) - dm$weights / sum(dm$weights))), 0.5)
  r0 <- outer_grid_rebuild_fixed(dm)
  r1 <- outer_grid_rebuild_fixed(dm, w_box)
  expect_gt(abs(r1$se[1] / r0$se[1] - 1), 0.05)
  expect_lt(abs(r1$se[2] / r0$se[2] - 1), 0.01)
})

# --- the two free cell rules, judged by coverage (gcol33/tulpa#331) ----------
#
# gcol33/tulpa#326 (a cell's mass integrated over its own box) and
# gcol33/tulpa#327 (a cell's atom placed at its own mass barycentre) are two
# corrections to what the midpoint atom throws away, both read off the same
# three-point stencil at zero inner solves, and neither has a production caller.
# Scored as distance to a finer grid's read, no rule dominates and the ranking
# flips with resolution, which is what gcol33/tulpa#331 asks coverage to settle.
#
# The four arms are the two rules crossed -- shipped, mass, location, pair --
# and all four are post-processing of ONE fit per seed through the
# gcol33/tulpa#322 dump, so they are paired on the same solve and not merely on
# the same data. The rules read a TENSOR base grid: on a locally refined grid
# the spliced design nodes leave only the base grid's own interior cells with a
# centred stencil (16 of 448 on this fixture), so the base grid is where they
# are defined and where they are scored.
#
# Two of the twelve cells of the table are not measurements. A weight rule
# cannot move a grid coordinate and a placement rule cannot move a cell's own
# inner solve, so `location` reads a fixed effect exactly as `shipped` does and
# `pair` exactly as `mass` does. The test below asserts that at 0.000e+00 rather
# than reporting four fixed-effect arms as if they were four.
lccd_arms <- function(dm) {
  bm <- .joint_local_ccd_box_mass(dm$joint_grid, dm$log_marginal,
                                  dm$axis_names, dm$axis_tags)
  base <- if (is.null(dm$dnode)) rep(1, nrow(dm$joint_grid)) else dm$dnode
  w <- outer_grid_weights(dm, dnode = base * exp(bm$log_box_ratio))
  g <- .joint_local_ccd_barycentre(dm$joint_grid, dm$log_marginal,
                                   dm$axis_names, dm$axis_tags)$joint_grid
  list(shipped  = list(w = NULL, g = NULL),
       mass     = list(w = w,    g = NULL),
       location = list(w = NULL, g = g),
       pair     = list(w = w,    g = g))
}

# One arm's two reads off one dump, each through the engine's own summary path:
# the hyperparameter axes through `.nl_axis_quantiles()`, the coefficients
# through `.nested_fixed_moments()`.
lccd_arm_read <- function(dm, arm, z = 1.96) {
  rb <- outer_grid_rebuild(dm, arm$w, arm$g)
  fx <- outer_grid_rebuild_fixed(dm, arm$w)
  list(s_lo = rb$ci_lo[[1L]], s_hi = rb$ci_hi[[1L]], s_med = rb$median[[1L]],
       lo = fx$mean - z * fx$se, hi = fx$mean + z * fx$se)
}

# All four arms over `n_seed` seeds at one base-grid resolution: per-arm sigma_1
# coverage and mean width, and the same pair for each coefficient. One fit per
# seed serves every arm.
lccd_arm_sweep <- function(levels, n_seed, seed_off = 8100L) {
  sg   <- exp(seq(log(0.2), log(1.5), length.out = 7))
  arms <- c("shipped", "mass", "location", "pair")
  p    <- length(LCCD_CFG$beta)
  s_cov <- stats::setNames(integer(length(arms)), arms)
  s_wid <- stats::setNames(numeric(length(arms)), arms)
  b_cov <- matrix(0L, length(arms), p, dimnames = list(arms, NULL))
  b_wid <- matrix(0,  length(arms), p, dimnames = list(arms, NULL))
  for (s in seq_len(n_seed)) {
    d  <- LCCD_SIM(seed_off + s, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr,
                   LCCD_CFG$ntr, LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
    dm <- outer_grid_dump(recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG,
                                                 levels = levels))
    A <- lccd_arms(dm)
    for (a in arms) {
      r <- lccd_arm_read(dm, A[[a]])
      s_cov[a] <- s_cov[a] +
        as.integer(LCCD_CFG$su >= r$s_lo && LCCD_CFG$su <= r$s_hi)
      s_wid[a] <- s_wid[a] + (r$s_hi - r$s_lo)
      b_cov[a, ] <- b_cov[a, ] +
        as.integer(LCCD_CFG$beta >= r$lo & LCCD_CFG$beta <= r$hi)
      b_wid[a, ] <- b_wid[a, ] + (r$hi - r$lo)
    }
  }
  list(n_seed = n_seed, s_cov = s_cov, s_width = s_wid / n_seed,
       b_cov = b_cov, b_width = b_wid / n_seed)
}

test_that("a weight rule and a placement rule reach different halves of the read", {
  skip_on_cran()
  d  <- LCCD_SIM(8101L, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr, LCCD_CFG$ntr,
                 LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
  sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
  dm <- outer_grid_dump(recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG))
  A  <- lccd_arms(dm)
  R  <- lapply(A, function(a) lccd_arm_read(dm, a))

  # Both rules have something to say on this grid: 16 of the 256 cells are
  # interior on all four axes and carry a centred stencil, and no axis of any
  # of them declines.
  bm <- .joint_local_ccd_box_mass(dm$joint_grid, dm$log_marginal,
                                  dm$axis_names, dm$axis_tags)
  bc <- .joint_local_ccd_barycentre(dm$joint_grid, dm$log_marginal,
                                    dm$axis_names, dm$axis_tags)
  expect_identical(sum(bm$computed), 16L)
  expect_identical(sum(bc$computed), 16L)
  expect_identical(bm$n_axes_declined, 0L)
  expect_identical(bc$n_axes_declined, 0L)

  # A placement rule moves grid coordinates and `.nested_fixed_moments()` reads
  # none, so the fixed half of the read is identical under it -- bit for bit,
  # not to a tolerance. The pair is the mass rule's fixed read for the same
  # reason. Half the four-arm table is therefore a derivation.
  expect_identical(R$location$lo, R$shipped$lo)
  expect_identical(R$location$hi, R$shipped$hi)
  expect_identical(R$pair$lo, R$mass$lo)
  expect_identical(R$pair$hi, R$mass$hi)
  # The converse holds one step further out. A weight rule leaves the
  # coordinates, so it reaches the hyperparameter axis only through the
  # softmax, and a weighted quantile over a discrete atom set can land on the
  # same coordinates under a redistribution this size -- it does on this seed,
  # which rewrites 0.1142 of the outer mass. What it moves on every seed is the
  # coefficient block it marginalizes. The placement rule reaches the axis
  # directly and contracts it to a third.
  w0 <- dm$weights / sum(dm$weights)
  w1 <- A$mass$w / sum(A$mass$w)
  expect_gt(0.5 * sum(abs(w1 - w0)), 0.1)
  expect_false(identical(R$mass$lo, R$shipped$lo))
  expect_lt(R$location$s_hi - R$location$s_lo,
            0.8 * (R$shipped$s_hi - R$shipped$s_lo))

  # On a locally refined grid the spliced design nodes have no axis neighbours,
  # so only the base grid's own interior cells keep a centred stencil. That is
  # why the arms are scored on the tensor base rather than on top of #320's
  # refinement.
  fr <- recov_fit_joint_local_ccd(d, sg, "gaussian", LCCD_CFG)
  dr <- outer_grid_dump(fr)
  br <- .joint_local_ccd_box_mass(dr$joint_grid, dr$log_marginal,
                                  dr$axis_names, dr$axis_tags)
  expect_gt(nrow(dr$joint_grid), nrow(dm$joint_grid))
  expect_identical(sum(br$computed), 16L)
})

# MEASURED, 200 seeds (seed offset 8100, seeds 1-200), one fit per seed per
# resolution, all four arms read off that fit. sigma_1 is the fit's own 95%
# interval on block 1's SD, true value 0.7; the coefficients are read at 1.96
# mixture SDs, nominal 0.95. Coverage counts of 200 and mean interval width:
#
#                  4 levels (256)    5 levels (625)    6 levels (1296)
#   sigma_1
#     shipped      200   1.0593      200   0.8787      200   0.6399
#     mass         200   1.1006      200   0.9091      200   0.6952
#     location     128   0.5092      129   0.5390      118   0.3079
#     pair         172   0.4775      174   0.4967      151   0.3563
#   intercept
#     shipped      191   0.6389      193   0.6283      190   0.6071
#     mass         190   0.6047      192   0.6052      190   0.6046
#   slope
#     shipped      191   0.08533     191   0.08534     191   0.08534
#     mass         191   0.08540     191   0.08538     191   0.08535
#
# COVERAGE ALONE IS THE WRONG STATISTIC HERE and the table is read two-sided:
# the shipped arm is at or above nominal everywhere, so an arm earns promotion
# only by being SHARPER without dropping below nominal. That framing is what
# gcol33/tulpa#320 established on this same fixture, where refinement sharpened
# sigma_1 fourfold while still covering 150 of 150 and what it removed was
# conservative-side slack.
#
# POWER. At 200 seeds the arms are paired off one solve, so the comparison is
# the discordant pairs and not two independent rates. An exact two-sided sign
# test needs six one-directional swaps to reject at 0.05, which is 3.0% of 200:
# power is 0.557 against a 3-point deficit, 0.938 against 5 points and 1.000
# against 10. Below about two points (power 0.213) nothing is resolvable, and
# the seed count was chosen against that rather than against the 15-point gap
# the pre-gcol33/tulpa#332 fixture appeared to carry. The fixed-effect half sits
# under that floor by construction: the mass rule moves the intercept's standard
# error 5.1% and the slope's 0.09%, and the observed discordance is 1 seed of
# 200 on the intercept and 0 on the slope at every resolution, so 400 fixed
# trials cannot separate the arms and only the WIDTH there is a measurement. The
# sigma_1 half is not close to that floor: location is -72 / +0 discordant at
# four levels, -71 / +0 at five and -82 / +0 at six, and the pair -28 / -26 /
# -49, every swap in the same direction, exact p below 1e-7 in all six.
#
# WHAT THE PLACEMENT RULE DOES is contract the atom set rather than correct it.
# Its sigma_1 width ratio against the shipped read is 0.4776 / 0.5840 / 0.4465
# at the three resolutions -- flat, not decaying -- because each cell's atom
# moves a share of its own box (up to 0.95 here) and that share does not shrink
# as the boxes do. A discretisation correction becomes inert on a finer grid;
# this one applies the same collapse at every resolution, and the weighted
# quantile it feeds depends on the atoms tiling the axis, which a uniform
# inward pull breaks. At four levels it lands at 0.5092 against the 1296-cell
# grid's own 0.6399, i.e. narrower than the converged answer, which is why its
# 72 misses are 71 high and 1 low.
#
# THAT IS ALSO WHY GRID ACCURACY RANKED IT FIRST. Per-seed absolute distance to
# the same seed's 1296-cell width: at four levels sigma_1 goes 0.4194 shipped
# against 0.3576 location, so the placement rule IS closer to the finer grid's
# read and still loses 72 seeds of coverage. The two error directions do not
# cost the same -- the shipped arm overshoots the converged width by 0.42 and
# pays nothing, the placement rule undershoots by 0.13 and pays 36 points --
# so a rule selected on distance to a finer grid can be selected against on
# calibration. That is the question gcol33/tulpa#331 was opened to answer.
#
# THE MASS RULE is the only arm that holds coverage, and it is not a sharpening.
# It never moves a sigma_1 trial (0 discordant at all three resolutions) and
# WIDENS that interval by 4.1% / 4.2% / 9.8%, on an axis already covering
# 200 of 200. What it does buy is the intercept: per-seed absolute distance to
# the 1296-cell interval width goes 0.04289 -> 0.03442 at four levels and
# 0.04478 -> 0.03049 at five, and its own width is 0.6047 / 0.6052 / 0.6046
# against the converged 0.6071 -- flat in the resolution, which is what a
# discretisation correction should look like -- for one trial of 200. So the
# rule is coverage-safe and its sign differs between the two reads, sharpening
# the coefficient and blunting the hyperparameter, at every resolution. None of
# the three is promoted to a default on this measurement.
test_that("only the mass rule keeps the outer grid's coverage, and it does not sharpen it", {
  skip_if_not_slow()
  n_seed <- 40L
  for (lev in c(4L, 5L)) {
    R <- lccd_arm_sweep(lev, n_seed)
    lab <- sprintf("%d levels", lev)

    # The coverage floor, which is what a candidate has to clear before its
    # width is worth reading. The shipped read and the mass rule cover every
    # seed's sigma_1 at 200 seeds; the floor is three binomial standard errors
    # of this gate's own 40 below that.
    expect_gte(R$s_cov[["shipped"]] / n_seed, 0.95, label = lab)
    expect_gte(R$s_cov[["mass"]] / n_seed, 0.95, label = lab)
    # Moving the atoms to their barycentres costs a third of it. Measured
    # 0.640 / 0.645 / 0.590 at the three resolutions, so 0.85 is more than
    # three of this gate's standard errors above the weakest.
    expect_lt(R$s_cov[["location"]] / n_seed, 0.85, label = lab)
    # The pair recovers most of what the placement lost and still misses:
    # measured 0.860 and 0.870, against a shipped arm that misses nothing.
    expect_lt(R$s_cov[["pair"]], R$s_cov[["shipped"]], label = lab)

    # Both placement arms narrow sigma_1 below the 1296-cell grid's own 0.6399,
    # which is the sense in which the narrowing is not slack being removed.
    expect_lt(R$s_width[["location"]], 0.7 * R$s_width[["shipped"]], label = lab)
    expect_lt(R$s_width[["pair"]], 0.7 * R$s_width[["shipped"]], label = lab)
    # And the arm that does hold coverage buys no sharpness on this axis.
    expect_gt(R$s_width[["mass"]], R$s_width[["shipped"]], label = lab)

    # The fixed half: the placement rule cannot reach it at all, and the mass
    # rule reaches the intercept (5.1% / 3.4% narrower) and not the slope
    # (0.09% / 0.04%), the within-group contrast gcol33/tulpa#325 found barely
    # reads this grid.
    expect_identical(R$b_width["location", ], R$b_width["shipped", ], label = lab)
    expect_identical(R$b_cov["location", ], R$b_cov["shipped", ], label = lab)
    expect_lt(R$b_width["mass", 1L], 0.99 * R$b_width["shipped", 1L], label = lab)
    expect_lt(abs(R$b_width["mass", 2L] / R$b_width["shipped", 2L] - 1), 5e-3,
              label = lab)
    # Coverage there is unmoved: 1 discordant seed of 200 on the intercept and
    # 0 on the slope, so the floor is set where a real collapse would show.
    expect_gte(min(R$b_cov["mass", ]) / n_seed, 0.85, label = lab)
  }
})

# --- the residual-scale convention every gaussian fixture here rests on ------
#
# `recov_draw_y()` hands `cfg$phi` straight to `rnorm()` as an SD, and every
# fitter hands the same number straight to a direct door. That is correct only
# while the door reads it as an SD, so the convention is asserted against the
# ENGINE rather than taken from the documentation -- if either side moves, this
# fails here with the reason on it instead of silently rescaling every gaussian
# fixture in the file the way gcol33/tulpa#332 did.
#
# The closed form makes the two conventions distinguishable rather than merely
# different: a gaussian arm's fixed block is X'X / phi^2, so a slope's per-cell
# standard error is phi / sqrt(Sxx) under the SD convention and sqrt(phi / Sxx)
# under the variance one. Three values of phi separate them, since the two
# agree at phi = 1.
test_that("both nested doors read a gaussian phi as the residual SD", {
  skip_on_cran()
  set.seed(99L)
  N <- 600L
  G <- 30L
  grp <- sample.int(G, N, replace = TRUE)
  x <- rnorm(N)
  X <- cbind(1, x)
  d <- list(y = as.numeric(X %*% c(-0.2, 0.7)) + rnorm(N, 0, 0.5), X = X,
            region = grp, regions = list(grp), nr = G, N = N)
  Sxx <- sum((x - mean(x))^2)
  # An RE pinned near zero, so the fixed block at the first grid cell is the
  # ordinary least-squares one and its standard error has the closed form above.
  sgt <- exp(seq(log(0.01), log(0.05), length.out = 4L))
  cfg <- list(ntr = 1L, phi = NA_real_)
  for (phi in c(0.25, 0.5, 1.0)) {
    cfg$phi <- phi
    for (f in list(recov_fit_single(d, sgt, "gaussian", cfg),
                   recov_fit_joint(d, sgt, "gaussian", cfg))) {
      se <- sqrt(diag(solve(f$grid_hessians[[1L]])))
      expect_equal(se[[2L]], phi / sqrt(Sxx), tolerance = 1e-3)
    }
  }
})
