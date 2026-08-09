# Inner-Laplace skew CORRECTION (gcol33/tulpa#302): the consumer of gamma_3.
#
# test-inner-skew.R validates the cubic term itself -- that it is the quantity
# the derivation defines and that it tracks an independently integrated exact
# posterior skewness. This file validates what is now DONE with it: the
# Cornish-Fisher marginal quantiles `summary()` / `confint()` report when
# `control$skew_correct` is on.
#
# The arbiter is the same kind as everywhere else in this engine: an
# independently computed exact answer, not a shape check. The exact posterior
# quantiles of an intercept-only binomial-logit model are available by direct
# quadrature (the construction test-inner-skew.R already uses for the skewness),
# so the corrected endpoints are held against those, and the credible intervals
# are held against nominal coverage over many simulated data sets.
#
# WHAT THE CORRECTION CANNOT DO, measured rather than assumed. gamma_3 is a
# LOWER BOUND on the true skewness (test-inner-skew.R pins the ratio), so a
# marginal corrected from it moves PART of the way. And the correction is
# skewness-only: the location term (Rue, Martino & Chopin 2009's gamma^(1)) is
# not computed by this engine, so a fit whose Laplace mode is biased keeps that
# bias. Both show up below as an improvement that is real and partial.

# --------------------------------------------------------------------------- #
# (1) Structural: the Cornish-Fisher kernel and its gates                     #
# --------------------------------------------------------------------------- #

test_that("the Cornish-Fisher quantile is the series it claims to be", {
  z <- stats::qnorm(c(0.025, 0.25, 0.5, 0.75, 0.975))
  g <- 0.4
  out <- tulpa:::cpp_cornish_fisher_quantile(mu = 1.5, sigma = 2, gamma3 = g,
                                             z = z, max_abs_gamma3 = 1)
  expect_true(out$applied)
  expect_equal(as.numeric(out$q), 1.5 + 2 * (z + (g / 6) * (z^2 - 1)))

  # Zero skewness is the Gaussian quantile exactly, not approximately: the
  # correction must be inert where the inner Laplace is exact (a gaussian-family
  # coefficient reads gamma_3 == 0 by construction).
  out0 <- tulpa:::cpp_cornish_fisher_quantile(mu = -0.3, sigma = 1.7, gamma3 = 0,
                                              z = z, max_abs_gamma3 = 1)
  expect_identical(as.numeric(out0$q), -0.3 + 1.7 * z)
  expect_true(out0$applied)

  # The correction is a pure relocation at a symmetric level: both endpoints
  # move by the same amount, so the interval width is untouched.
  q <- as.numeric(out$q)
  expect_equal(q[5] - q[1], 2 * (z[5] - z[1]))
})

test_that("the correction declines rather than extrapolating, and says so per index", {
  z <- stats::qnorm(c(0.025, 0.975))
  # An `unreliable` band (|gamma_3| at or past the cutoff), a non-computable
  # gamma_3, and a degenerate scale each fall back to the Gaussian quantiles --
  # which are still REPORTED, so the table is complete and the flag says which
  # rows they are.
  out <- tulpa:::cpp_cornish_fisher_quantile(
    mu = c(0, 0, 0, 0), sigma = c(1, 1, 1, 0),
    gamma3 = c(0.9, 1.0, NaN, 0.5), z = z, max_abs_gamma3 = 1)
  expect_identical(as.logical(out$applied), c(TRUE, FALSE, FALSE, FALSE))
  expect_identical(out$q[2, ], z)          # band cutoff
  expect_identical(out$q[3, ], z)          # gamma_3 not computable
  expect_identical(out$q[4, ], c(0, 0))    # degenerate scale
  expect_false(isTRUE(all.equal(out$q[1, ], z)))

  # The map w(z) = z + (g/6)(z^2 - 1) is a quantile function only where it
  # increases, dw/dz = 1 + (g/3) z > 0. At |g| < 1 that holds out to |z| < 3, so
  # an extreme level with an otherwise eligible gamma_3 declines rather than
  # returning a crossed interval.
  wide <- stats::qnorm(c(1e-5, 1 - 1e-5))       # |z| ~ 4.26
  far <- tulpa:::cpp_cornish_fisher_quantile(mu = 0, sigma = 1, gamma3 = 0.9,
                                             z = wide, max_abs_gamma3 = 1)
  expect_false(as.logical(far$applied))
  expect_equal(as.numeric(far$q), wide)
  # Every reported interval is ordered, corrected or not.
  expect_lt(far$q[1, 1], far$q[1, 2])
})

test_that(".nl_skew_marginal is inert when the correction is switched off", {
  probs <- c(0.025, 0.975)
  mu <- c(0.4, -1.2); sd <- c(0.5, 0.9); g <- c(0.6, -0.7)
  off <- .nl_skew_marginal(mu, sd, g, probs, enabled = FALSE)
  expect_identical(off$applied, c(FALSE, FALSE))
  expect_identical(off$q, matrix(mu, 2L, 2L) + outer(sd, stats::qnorm(probs)))
  on <- .nl_skew_marginal(mu, sd, g, probs, enabled = TRUE)
  expect_identical(on$applied, c(TRUE, TRUE))
  expect_false(isTRUE(all.equal(on$q, off$q)))
})

test_that(".nl_skew_by_fixed maps probed indices onto coefficients, never onto zero", {
  # The default probe is 1:p, so gamma3[j] is coefficient j.
  f <- list(inner_skew = c(0.3, -0.2), inner_skew_idx = 1:2)
  expect_identical(.nl_skew_by_fixed(f, 2L), c(0.3, -0.2))
  # A fit whose control$skew_idx probed something else leaves the unprobed
  # coefficients NA -- "not scored" is not "no skew".
  f2 <- list(inner_skew = c(0.5), inner_skew_idx = 2L)
  expect_identical(.nl_skew_by_fixed(f2, 3L), c(NA_real_, 0.5, NA_real_))
  # A latent index past the fixed block is not a coefficient and is dropped.
  f3 <- list(inner_skew = c(0.5, 0.9), inner_skew_idx = c(1L, 40L))
  expect_identical(.nl_skew_by_fixed(f3, 2L), c(0.5, NA_real_))
  expect_identical(.nl_skew_by_fixed(list(), 2L), c(NA_real_, NA_real_))
})

test_that("a fit with no skew_correction record reports Gaussian quantiles", {
  sc <- .nl_skew_correction(list(), 2L)
  expect_false(sc$enabled)
  expect_identical(sc$eligible, c(FALSE, FALSE))
})

# --------------------------------------------------------------------------- #
# (2) Recovery: the corrected endpoints move TOWARD the exact quantiles        #
# --------------------------------------------------------------------------- #

# Exact posterior quantiles of an intercept-only binomial-logit model by direct
# quadrature -- the quantile-side extension of `.exact_intercept_skew()` in
# test-inner-skew.R, with the same weakly-informative N(0, sigma_beta^2) prior
# the C++ kernel applies.
.exact_intercept_quantiles <- function(N, S, probs, sigma_beta = 100) {
  lp <- function(e) S * e - N * log1p(exp(e)) - 0.5 * (e / sigma_beta)^2
  mode <- stats::optimize(lp, c(-40, 40), maximum = TRUE)$maximum
  grid <- seq(mode - 20, mode + 20, length.out = 200001L)
  w <- exp(lp(grid) - max(lp(grid)))
  w <- w / sum(w)
  cdf <- cumsum(w)
  keep <- !duplicated(cdf)
  stats::approx(cdf[keep], grid[keep], xout = probs)$y
}

# The engine's inner-Laplace marginal for that model: mode, marginal sd and
# gamma_3 at the probed intercept.
.skew_intercept_fit <- function(N, S) {
  y <- c(rep(1, S), rep(0, N - S))
  f <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, N), X = matrix(1, N, 1),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "binomial", compute_skew = TRUE, skew_idx = 1L)
  list(mu = f$mode[1L], sd = f$inner_is_sigma[1L], g3 = f$inner_skew[1L])
}

test_that("the skew-corrected marginal beats the Gaussian one against exact quantiles", {
  skip_on_cran()
  probs <- c(0.025, 0.975)
  # Rare-event binomials of decreasing size: gamma_3 grows and so does the
  # Gaussian marginal's error. Every case is inside the band the correction is
  # gated to, so every case is actually corrected.
  cases <- list(c(N = 100, S = 3), c(N = 60, S = 5), c(N = 40, S = 4),
                c(N = 20, S = 2))
  err_g <- 0; err_c <- 0
  for (cs in cases) {
    N <- cs[["N"]]; S <- cs[["S"]]
    fit <- .skew_intercept_fit(N, S)
    ex  <- .exact_intercept_quantiles(N, S, probs)
    mg  <- .nl_skew_marginal(fit$mu, fit$sd, fit$g3, probs, enabled = TRUE)
    expect_true(mg$applied)
    qg <- fit$mu + fit$sd * stats::qnorm(probs)
    qc <- as.numeric(mg$q)
    # Both endpoints improve, in every case -- not an average that hides one
    # endpoint getting worse.
    expect_lt(abs(qc[1] - ex[1]), abs(qg[1] - ex[1]))
    expect_lt(abs(qc[2] - ex[2]), abs(qg[2] - ex[2]))
    err_g <- err_g + sum(abs(qg - ex))
    err_c <- err_c + sum(abs(qc - ex))
  }
  # Measured total absolute endpoint error over these four cases: Gaussian
  # 2.4931, corrected 1.3837 -- a 44.5% reduction. The correction is PARTIAL by
  # construction: gamma_3 undershoots the true skewness (0.875 to 0.943 of the
  # exact quadrature value on these very cases) and the location term is not
  # computed, so the gate is a substantial reduction, not agreement.
  expect_lt(err_c, 0.7 * err_g)
  expect_gt(err_c, 0.4 * err_g)
})

test_that("the correction is inert where the inner Laplace is already exact", {
  skip_on_cran()
  # A gaussian-family coefficient has gamma_3 identically 0, so the corrected
  # marginal must be the Gaussian one bit for bit -- the correction may not
  # perturb a fit it has nothing to say about.
  set.seed(517)
  n <- 300L
  x <- rnorm(n)
  y <- 1 + 0.5 * x + rnorm(n, 0, 1)
  f <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, n), X = cbind(1, x),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "gaussian", compute_skew = TRUE, skew_idx = as.integer(1:2))
  probs <- c(0.025, 0.975)
  sd <- f$inner_is_sigma
  on  <- .nl_skew_marginal(f$mode[1:2], sd, f$inner_skew, probs, enabled = TRUE)
  off <- .nl_skew_marginal(f$mode[1:2], sd, f$inner_skew, probs, enabled = FALSE)
  expect_identical(on$q, off$q)
  expect_identical(on$applied, c(TRUE, TRUE))
})

# The CI-coverage gate for the correction lives with the other coverage gates,
# in test-nested-laplace-recovery.R, and reuses that file's recovery harness.

# --------------------------------------------------------------------------- #
# (3) End to end through the front door, and bit-for-bit inert when off        #
# --------------------------------------------------------------------------- #

.skew_corr_fixture <- function(seed = 1L, nr = 12L, spr = 4L,
                               beta = c(-2.5, 0.8), sigma_u = 0.7) {
  set.seed(seed)
  N <- nr * spr
  region <- rep(seq_len(nr), each = spr)
  x <- rnorm(N)
  X <- cbind(1, x)
  u <- rnorm(nr, 0, sigma_u)
  eta <- as.numeric(X %*% beta) + u[region]
  list(y = stats::rbinom(N, 1, stats::plogis(eta)), X = X,
       region = as.integer(region), nr = nr, N = N)
}

.skew_corr_fit <- function(d, skew_correct) {
  prior <- list(list(type = "iid", obs_idx = d$region, n_units = d$nr,
                     sigma_grid = exp(seq(log(0.2), log(1.5), length.out = 7))))
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$N), X = d$X, prior = prior,
    family = "binomial", phi = 1,
    control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                   keep_grid_hessians = TRUE, diagnose_k = FALSE,
                   progress = FALSE, skew_correct = skew_correct)))
}

test_that("tulpa_nested_laplace() records the correction and confint() applies it", {
  skip_on_cran()
  d <- .skew_corr_fixture()
  f <- .skew_corr_fit(d, TRUE)

  sc <- f$skew_correction
  expect_true(sc$enabled)
  expect_length(sc$gamma3, 2L)
  expect_true(all(is.finite(sc$gamma3)))
  expect_identical(sc$band, vapply(sc$gamma3, .tulpa_gamma3_band, character(1)))

  # The gate is the COMBINED inner band (gcol33/tulpa#346), so eligibility is
  # read off `band_combined` -- the worse of gamma_3's band and the importance
  # k-hat's -- and every coefficient says which score decided it. On this
  # fixture the intercept's k-hat is 0.72 at material efficiency while its
  # gamma_3 is a moderate -0.57, so the two gates disagree and the combined one
  # declines: the gamma_3-only gate this replaced corrected it.
  expect_identical(sc$band_combined, .subspace_bands(f)$band)
  expect_identical(sc$eligible, sc$reason == "eligible")
  expect_true(all(sc$reason %in% .SKEW_CORRECT_REASONS))
  expect_identical(sc$reason, c("inner_k_unreliable", "eligible"))
  expect_identical(sc$eligible, c(FALSE, TRUE))
  expect_gte(sc$pareto_k[1], .nl_diag("k_usable"))
  # The whole-fit verdict the eligibility was decided under travels with it.
  expect_true(grepl("inner", sc$reliability))

  ci <- confint(f)
  applied <- attr(ci, "skew_applied")
  expect_identical(unname(applied), c(FALSE, TRUE))
  expect_identical(names(applied), rownames(ci))
  # summary() reads the same table and carries the same record.
  expect_identical(unname(attr(summary(f), "skew_applied")), c(FALSE, TRUE))

  # The bounds are the Cornish-Fisher ones at this fit's own gamma_3 and its
  # own reported centre and standard error -- not merely "different" -- on the
  # eligible coefficient, and the Gaussian ones on the declined one.
  tab <- .fit_fixed_table(f)
  z <- stats::qnorm(c(0.025, 0.975))
  for (j in 1:2) {
    g <- if (sc$eligible[j]) sc$gamma3[j] else 0
    w <- z + (g / 6) * (z^2 - 1)
    expect_equal(as.numeric(ci[j, ]),
                 tab$estimate[j] + tab$std.error[j] * w)
  }
  # Selecting a subset carries the record with it.
  ci1 <- confint(f, parm = rownames(ci)[2])
  expect_identical(unname(attr(ci1, "skew_applied")), TRUE)
})

test_that("switching the correction off leaves the fit bit for bit what it was", {
  skip_on_cran()
  d  <- .skew_corr_fixture()
  on  <- .skew_corr_fit(d, TRUE)
  off <- .skew_corr_fit(d, FALSE)

  # The correction is post-processing on the reported quantiles: it must not
  # touch the posterior itself. Every field the inference produced is identical.
  for (fld in c("draws", "modes", "weights", "log_marginal", "theta_grid",
                "theta_mean", "theta_sd", "grid_modes", "grid_hessians",
                "n_iter", "inner_skew", "inner_skew_idx")) {
    expect_identical(on[[fld]], off[[fld]], label = paste0("field ", fld))
  }
  # And the off fit reports the interval the outer grid's own Gaussian mixture
  # carries (gcol33/tulpa#336). It used to report `estimate +/- z se` off the
  # single Gaussian matching the marginalized moments; those moments are
  # unchanged, so the two reads differ only in the quantile step, which is
  # nonlinear and does not survive the collapse.
  tab <- .fit_fixed_table(off)
  expect_identical(attr(tab, "interval_source"), "mixture_cdf")
  mom <- .nested_fixed_moments(off)
  p   <- nrow(tab)
  ref <- .nl_gauss_mixture_summary(mom$mu[, seq_len(p), drop = FALSE],
                                   mom$var[, seq_len(p), drop = FALSE],
                                   mom$w, probs = c(0.025, 0.975))
  expect_equal(tab$conf.low,  ref$quantiles[, 1L], tolerance = 1e-12)
  expect_equal(tab$conf.high, ref$quantiles[, 2L], tolerance = 1e-12)
  z <- stats::qnorm(0.975)
  expect_false(isTRUE(all.equal(tab$conf.low,
                                tab$estimate - z * tab$std.error)))
  expect_identical(unname(attr(confint(off), "skew_applied")), c(FALSE, FALSE))
  expect_false(off$skew_correction$enabled)

  # The two fits differ ONLY in the reported bounds.
  expect_false(isTRUE(all.equal(unname(confint(on)), unname(confint(off)))))
  expect_identical(coef(on), coef(off))
  expect_identical(vcov(on), vcov(off))
})

test_that("the correction declines a coefficient the importance k-hat flags", {
  skip_on_cran()
  # The gate is per coefficient and reads the combined band, so a fit whose two
  # coefficients disagree carries one corrected and one Gaussian row, and the
  # record names the score responsible for each. `.nl_skew_gamma3_eligible()` is
  # what the quantile path consumes, so a declined coefficient reaches
  # `.nl_skew_marginal()` as NA rather than being noted and corrected anyway.
  sc <- list(enabled = TRUE, gamma3 = c(0.4, -0.3, 0.2),
             eligible = c(TRUE, FALSE, TRUE))
  expect_identical(.nl_skew_gamma3_eligible(sc), c(0.4, NA_real_, 0.2))
  # A record with no eligibility vector at all (a fit predating it) is read at
  # face value rather than silently blanked.
  expect_identical(.nl_skew_gamma3_eligible(list(gamma3 = c(0.4, -0.3))),
                   c(0.4, -0.3))

  probs <- c(0.025, 0.975)
  mg <- .nl_skew_marginal(rep(0, 3), rep(1, 3),
                          .nl_skew_gamma3_eligible(sc), probs, enabled = TRUE)
  expect_identical(mg$applied, c(TRUE, FALSE, TRUE))
  expect_identical(mg$q[2, ], stats::qnorm(probs))
})

test_that("an unreliable band keeps the Gaussian quantiles at the front door", {
  skip_on_cran()
  # The band gate is what stops a leading-order expansion being extrapolated
  # past its regime. Forced here by tightening the cutoff rather than hunting
  # for a fixture, so the gate itself is what is tested.
  d <- .skew_corr_fixture()
  f <- .skew_corr_fit(d, TRUE)
  g <- f$skew_correction$gamma3
  tight <- min(abs(g)) / 2
  mg <- .nl_skew_marginal(rep(0, 2), rep(1, 2), g, c(0.025, 0.975),
                          enabled = TRUE, max_abs_gamma3 = tight)
  expect_identical(mg$applied, c(FALSE, FALSE))
  expect_equal(mg$q[, 1], rep(stats::qnorm(0.025), 2))
})

# --------------------------------------------------------------------------- #
# (4) The whole-marginal gate (gcol33/tulpa#346)                              #
# --------------------------------------------------------------------------- #
#
# Section 2 scores total absolute error of the 2.5% / 97.5% quantiles, and that
# is structurally blind to most of what the correction does. At a SYMMETRIC
# level pair the Cornish-Fisher term sigma (gamma_3 / 6) (z_p^2 - 1) takes the
# same value at both ends, because z_p^2 = z_{1-p}^2, so the correction is a
# pure LOCATION SHIFT of the interval with its width exactly unchanged and a
# two-point symmetric metric can measure nothing else. Away from a symmetric
# pair the term is not a shift at all: it is negative at z = 0 and positive at
# |z| > 1 for gamma_3 > 0, so the median moves one way while the tails move the
# other.
#
# This gate reads the whole CDF instead, through the gcol33/tulpa#335 harness:
# SBC uniformity against the exact simultaneous band, and paired CRPS against
# the exact posterior in a prior-predictive experiment, where the CRPS is a
# proper posterior score. The `shift only` arm -- a Gaussian relocated by
# exactly the 95%-level Cornish-Fisher offset, with no reshaping -- is the
# control that separates the two effects, and without it the gate still cannot
# tell them apart.
#
# The fixture is the rare-event binomial-logit intercept of section 2 driven
# prior-predictively, so the exact posterior is a one-dimensional quadrature and
# no reference sampler is needed.

.WM_N   <- 40L        # trials
.WM_OFF <- -4         # offset, so successes are rare and the posterior skews
.WM_SB  <- 2.0        # the prior the truth is drawn from IS the prior fitted
.WM_P   <- (seq_len(2001L) - 0.5) / 2001L
.WM_Z   <- stats::qnorm(0.975)

.wm_sim <- function(seed) {
  set.seed(seed)
  beta <- stats::rnorm(1, 0, .WM_SB)
  list(seed = seed, y = as.numeric(stats::rbinom(.WM_N, 1L,
                                                 stats::plogis(.WM_OFF + beta))),
       theta = c(beta = beta))
}

.wm_lp <- function(d, b) {
  eta <- .WM_OFF + b
  sum(d$y) * eta - .WM_N * log1p(exp(eta)) - 0.5 * (b / .WM_SB)^2
}

# The exact posterior as an inverse-CDF grid centred on the engine's own mode.
.wm_exact <- function(d, mu, sd, p = .WM_P) {
  g <- seq(mu - 14 * max(sd, 0.5), mu + 14 * max(sd, 0.5), length.out = 20001L)
  lp <- .wm_lp(d, g)
  w <- exp(lp - max(lp)); w <- w / sum(w)
  cdf <- cumsum(w) - 0.5 * w
  keep <- !duplicated(cdf) & is.finite(cdf)
  stats::approx(cdf[keep], g[keep], xout = p, rule = 2)$y
}

# The shipped Cornish-Fisher marginal as a DISTRIBUTION: the push-forward of
# N(0, 1) through the map, restricted to its monotone branch |z| < 3 / |gamma_3|
# and renormalized over it. On that branch it is the shipped quantile at every
# level, which the gate asserts rather than assumes; the discarded tail mass is
# at most 2 Phi(-3) = 0.0027 at the band edge and less inside it.
.wm_cf <- function(mu, sigma, g3, p = .WM_P,
                   max_abs_g3 = .nl_diag("gamma3_unreliable")) {
  if (!is.finite(g3) || abs(g3) >= max_abs_g3 || !is.finite(sigma) || sigma <= 0)
    return(mu + sigma * stats::qnorm(p))
  zb <- 3 / abs(g3)
  lo <- stats::pnorm(-zb); hi <- stats::pnorm(zb)
  z <- stats::qnorm(lo + p * (hi - lo))
  mu + sigma * (z + (g3 / 6) * (z^2 - 1))
}

test_that("the whole-marginal score sees what the endpoint score cannot", {
  skip_if_not_slow()

  probs <- c(0.025, 0.975)
  rows <- list()

  arms <- function(d) {
    f <- tulpa_laplace(y = d$y, n_trials = rep(1L, .WM_N),
                       X = matrix(1, .WM_N, 1), family = "binomial",
                       offset = rep(.WM_OFF, .WM_N),
                       beta_prior = list(mean = 0, sd = .WM_SB),
                       max_iter = 200L, tol = 1e-11,
                       compute_skew = TRUE, skew_idx = 1L)
    mu <- as.numeric(f$mode[1]); sd <- as.numeric(f$inner_is_sigma[1])
    g3 <- as.numeric(f$inner_skew[1])
    ex <- .wm_exact(d, mu, sd)
    shift <- sd * (g3 / 6) * (.WM_Z^2 - 1)

    # The section-2 endpoint reading of the SAME fit, from the shipped function
    # at the shipped gamma_3-only gate: what the correction was accepted on.
    exq <- .wm_exact(d, mu, sd, probs)
    qg <- mu + sd * stats::qnorm(probs)
    mg <- .nl_skew_marginal(mu, sd, g3, probs, enabled = TRUE)
    qc <- as.numeric(mg$q)
    rows[[length(rows) + 1L]] <<- data.frame(
      seed = d$seed, gamma3 = g3, sd = sd, applied = mg$applied,
      band = as.character(.subspace_bands(f)$band[1]),
      # the two endpoints' displacement, which the algebra says is one number
      move_lo = qc[1] - qg[1], move_hi = qc[2] - qg[2],
      err_g = sum(abs(qg - exq)), err_c = sum(abs(qc - exq)),
      stringsAsFactors = FALSE)

    list(exact      = list(beta = sbc_draws(ex)),
         laplace    = list(beta = sbc_normal(mu, sd)),
         shift_only = list(beta = sbc_normal(mu + shift, sd)),
         skew_cf    = list(beta = sbc_draws(.wm_cf(mu, sd, g3))))
  }

  res <- recov_sbc(.wm_sim, arms, 400L, truth = "prior_draw")
  rep <- sbc_report(res)
  cmp <- sbc_crps_compare(res, "laplace")
  dg <- do.call(rbind, rows)
  get <- function(a, col) rep[[col]][rep$arm == a]
  dlt <- function(a) cmp$delta[cmp$arm == a]
  tstat <- function(a) cmp$t[cmp$arm == a]

  # The arm's quantile IS the shipped one on the branch it is restricted to,
  # so the distribution scored below is the correction and not a lookalike.
  lv <- c(0.1, 0.5, 0.9)
  chk <- .nl_skew_marginal(0.3, 1.4, 0.6, lv, enabled = TRUE)
  expect_true(chk$applied)
  # The branch restriction renormalizes the probability scale, so level q sits
  # at (q - Phi(-z_b)) / (Phi(z_b) - Phi(-z_b)) of the arm's own grid.
  zb <- 3 / 0.6
  lo <- stats::pnorm(-zb); hi <- stats::pnorm(zb)
  expect_equal(as.numeric(chk$q), .wm_cf(0.3, 1.4, 0.6, p = (lv - lo) / (hi - lo)),
               tolerance = 1e-12)

  # (a) HARNESS SELF-CHECK. The exact posterior is the CRPS-optimal forecast in
  # a prior-predictive experiment and its PIT is uniform, so it must score best
  # and deviate least. Measured: CRPS 0.5504, KS 0.0290.
  expect_lt(get("exact", "crps"), min(get("laplace", "crps"),
                                      get("shift_only", "crps"),
                                      get("skew_cf", "crps")))
  expect_lt(get("exact", "ks"), min(get("laplace", "ks"), get("skew_cf", "ks")))

  # (b) THE ENDPOINT SCORE IS A PURE LOCATION SHIFT, on real fits and not only
  # in the algebra: both bounds move by the same number on every replicate.
  expect_equal(dg$move_lo, dg$move_hi, tolerance = 1e-12)

  # (c) AND IT IMPROVES, by more than the 44.5% section 2 records. Measured:
  # 441.42 -> 191.58, a 56.6% reduction, with both endpoints better on 396 of
  # 400 replicates. Read alone this is the correction working.
  expect_lt(sum(dg$err_c), 0.6 * sum(dg$err_g))
  expect_gt(mean(dg$err_c < dg$err_g), 0.95)

  # (d) THE WHOLE MARGINAL DISAGREES. The same fits, scored over the whole CDF:
  # the correction is a NET LOSS against the uncorrected Laplace. Measured
  # delta +0.00775 (t +3.54), KS 0.0833 -> 0.1139. This is the pin that has to
  # move if the location term (RMC 2009's gamma^(1), gcol33/tulpa#354) is ever
  # computed and the reshaping is applied about the right centre.
  expect_gt(dlt("skew_cf"), 0)
  expect_gt(tstat("skew_cf"), 2)
  expect_gt(get("skew_cf", "ks"), get("laplace", "ks"))

  # (e) THE CONTROL SEPARATES THE TWO EFFECTS. The shift alone -- the part the
  # endpoint score measures -- is a GAIN, and recovers what a correctly located
  # Gaussian achieves; the reshaping laid on top of it is what turns the gain
  # into a loss. Measured: shift only -0.01451 (t -1.43) against exact -0.01662,
  # and its PIT re-enters the simultaneous band (p 0.226 against 7.6e-06).
  expect_lt(dlt("shift_only"), 0)
  expect_gt(dlt("skew_cf"), dlt("shift_only"))
  expect_lt(get("shift_only", "crps"), get("laplace", "crps"))
  expect_gt(get("shift_only", "p_unif"), 0.05)

  # (f) THE GATE THE CORRECTION NOW READS. gamma_3 alone admits every replicate
  # here, while the combined inner band is `unreliable` on 38.3% of them, driven
  # by the importance k-hat -- the same coordinates gcol33/tulpa#304 selects for
  # exact sampling.
  expect_true(all(dg$applied))
  expect_gt(mean(dg$band == "unreliable"), 0.2)
})
