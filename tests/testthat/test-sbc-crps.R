# test-sbc-crps.R
#
# The SBC / PIT and CRPS instruments in helper-sbc.R, and the acceptance
# measurement they were built for (gcol33/tulpa#335).
#
# Three things have to be true before a calibration harness is worth reading,
# and each is a section below:
#   1. the CRPS closed forms are the score they claim to be -- held against
#      numerical integration of the definition and against the Monte-Carlo
#      kernel estimator, on mixtures chosen to break a wrong formula;
#   2. the band is SIMULTANEOUS -- its realized simultaneous coverage is
#      measured, its Kolmogorov-Smirnov member reproduces the published critical
#      value, and the pointwise band it is not is shown to cover far less;
#   3. the harness can FAIL -- a deliberately mis-scaled posterior and a
#      generator / inference convention crossing both land outside the band.
# Only then is section 4's reading of the mixture against the collapsed
# Gaussian worth anything.

# ---------------------------------------------------------------------------
# 1. CRPS
# ---------------------------------------------------------------------------

# Mixtures chosen so that a formula that dropped the cross terms, symmetrized
# the wrong index pair, or used sigma_k rather than sqrt(s2_k + s2_l) in the
# second sum would disagree with at least one of them.
CRPS_CASES <- list(
  degenerate    = list(y =  0.3, mu = 0,             var = 1,                 w = 1),
  asymmetric    = list(y = -0.4, mu = c(-1, 2),      var = c(0.25, 4),        w = c(0.7, 0.3)),
  scale_spread  = list(y =  1.1, mu = c(0, 0.2, 5),  var = c(1e-4, 1, 100),   w = c(0.5, 0.3, 0.2)),
  tail_truth    = list(y =  8.0, mu = c(-1, 1),      var = c(0.5, 0.5),       w = c(0.5, 0.5)),
  equal_means   = list(y =  0.0, mu = c(0, 0),       var = c(0.01, 9),        w = c(0.9, 0.1))
)

test_that("the mixture CRPS closed form integrates the definition", {
  for (nm in names(CRPS_CASES)) {
    cs <- CRPS_CASES[[nm]]
    dist <- sbc_mixture(cs$mu, cs$var, cs$w)
    expect_equal(sbc_crps(dist, cs$y), sbc_crps_integral(dist, cs$y),
                 tolerance = 1e-4, label = sprintf("%s CRPS", nm))
  }
})

test_that("a one-component mixture reproduces the Gaussian CRPS", {
  # sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ], z = (y - mu)/sigma.
  for (mu in c(-1, 0, 2)) for (sg in c(0.3, 1, 4)) for (y in c(-2, 0, 1.5)) {
    z <- (y - mu) / sg
    ref <- sg * (z * (2 * stats::pnorm(z) - 1) + 2 * stats::dnorm(z) - 1 / sqrt(pi))
    expect_equal(sbc_crps(sbc_normal(mu, sg), y), ref, tolerance = 1e-12)
  }
})

test_that("the discrete CRPS closed form integrates the definition", {
  # Splitting the panels at every atom makes the numerical integral exact here,
  # so this is a machine-precision identity rather than a tolerance.
  d <- sbc_discrete(c(0.2, 0.5, 1.1, 3.0), c(0.4, 0.3, 0.2, 0.1))
  for (y in c(0.2, 0.7, 3.0, -1.0)) {
    expect_equal(sbc_crps(d, y), sbc_crps_integral(d, y), tolerance = 1e-12,
                 label = sprintf("discrete CRPS at y = %g", y))
  }
})

test_that("the draws CRPS uses the sorted identity for the pairwise term", {
  set.seed(4L)
  x <- stats::rnorm(200)
  y <- 0.4
  ref <- mean(abs(x - y)) - 0.5 * mean(abs(outer(x, x, "-")))
  expect_equal(sbc_crps(sbc_draws(x), y), ref, tolerance = 1e-12)
})

test_that("the mixture CRPS matches the Monte-Carlo kernel estimator", {
  skip_on_cran()
  set.seed(99L)
  for (nm in names(CRPS_CASES)) {
    cs <- CRPS_CASES[[nm]]
    dist <- sbc_mixture(cs$mu, cs$var, cs$w)
    cf <- sbc_crps(dist, cs$y)
    mc <- sbc_crps_mc(dist, cs$y, n = 2e6L)
    # The estimator's own standard error scales with the spread of the mixture,
    # so the tolerance is relative to the score rather than absolute.
    expect_lt(abs(cf - mc) / max(1, abs(cf)), 5e-3,
              label = sprintf("%s closed form vs MC", nm))
  }
})

# ---------------------------------------------------------------------------
# 2. PIT
# ---------------------------------------------------------------------------

test_that("the mixture PIT is the mixture CDF and the fold preserves uniformity", {
  dist <- sbc_mixture(c(-1, 2), c(0.25, 4), c(0.7, 0.3))
  for (y in c(-3, -0.4, 0, 5)) {
    expect_equal(sbc_pit(dist, y),
                 0.7 * stats::pnorm(y, -1, 0.5) + 0.3 * stats::pnorm(y, 2, 2),
                 tolerance = 1e-12)
  }
  u <- seq(0.001, 0.999, length.out = 999)
  # 2|u - 1/2| maps Uniform(0,1) onto Uniform(0,1), so its ECDF against the
  # identity is the identity up to the grid spacing.
  expect_lt(max(abs(sort(sbc_fold(u)) - seq_along(u) / length(u))), 2 / length(u))
})

test_that("a discrete PIT randomizes within its atom", {
  d <- sbc_discrete(c(1, 2, 3), c(0.5, 0.3, 0.2))
  expect_equal(sbc_pit(d, 1, u = 0), 0)
  expect_equal(sbc_pit(d, 1, u = 1), 0.5)
  expect_equal(sbc_pit(d, 2, u = 0.5), 0.65)
  expect_equal(sbc_pit(d, 3, u = 1), 1)
  # A rank r out of n_ref is the same construction: (r + V) / (n_ref + 1).
  expect_equal(sbc_pit(sbc_rank(7L, 20L), NA_real_, u = 0.25), (7 + 0.25) / 21)
})

test_that("randomizing the rank is what makes the reference uniform", {
  skip_on_cran()
  # The classic silent SBC bug: reading r / n_ref against a continuous uniform.
  # At n_ref = 9 the unrandomized PIT sits on ten atoms and its ECDF departs
  # from the identity by up to 1/(n_ref + 1) = 0.1, which the simultaneous band
  # at n = 400 rejects; the randomized read is exactly uniform and passes.
  n <- 400L; L <- 9L
  set.seed(5L)
  r <- sample.int(L + 1L, n, replace = TRUE) - 1L
  band <- sbc_ecdf_band(n, 0.95)
  u_rand <- (r + stats::runif(n)) / (L + 1)
  u_raw  <- r / L
  expect_true(sbc_ecdf_inside(u_rand, band))
  expect_false(sbc_ecdf_inside(u_raw, band))
  expect_gt(sbc_ecdf_dev(u_raw), 0.08)
})

# ---------------------------------------------------------------------------
# 3. The simultaneous band
# ---------------------------------------------------------------------------

test_that("the crossing probability reproduces the cases with a closed form", {
  # A one-sided band at a constant level collapses to an event with an
  # elementary probability: h_i = t for every i is "every draw is at most t",
  # and g_i = t for every i is "every draw is at least t".
  for (n in c(5L, 10L, 25L)) for (t in c(0.3, 0.7)) {
    expect_equal(sbc_crossing_prob(rep(0, n), rep(t, n)), t^n, tolerance = 1e-12)
    expect_equal(sbc_crossing_prob(rep(t, n), rep(1, n)), (1 - t)^n,
                 tolerance = 1e-12)
  }
  n <- 10L
  expect_equal(sbc_crossing_prob(rep(0, n), rep(1, n)), 1, tolerance = 1e-12)
  # A zero-width band cannot be met by a continuous sample.
  expect_identical(sbc_crossing_prob(rep(0.5, n), rep(0.5, n)), 0)
  # Monotone in the band width.
  p <- vapply(c(0.4, 0.2, 0.1, 0.02), function(gm) {
    b <- .sbc_bounds(n, "beta", gm); sbc_crossing_prob(b$g, b$h)
  }, numeric(1))
  expect_true(all(diff(p) > 0))
})

test_that("the crossing probability reproduces a brute-force simulation", {
  skip_on_cran()
  set.seed(21L)
  n <- 12L; B <- 60000L
  U <- matrix(stats::runif(n * B), B, n)
  U <- t(apply(U, 1L, sort))
  for (spec in list(list("beta", 0.20), list("ks", 0.30))) {
    b <- .sbc_bounds(n, spec[[1]], spec[[2]])
    hit <- mean(apply(U, 1L, function(u) all(u >= b$g) && all(u <= b$h)))
    ex <- sbc_crossing_prob(b$g, b$h)
    expect_lt(abs(hit - ex), 4 * sqrt(ex * (1 - ex) / B),
              label = sprintf("%s crossing probability", spec[[1]]))
  }
})

test_that("the Kolmogorov-Smirnov band reproduces the published critical value", {
  skip_on_cran()
  # An external arbiter on the recursion: calibrating the constant-width band
  # against it must return the Kolmogorov 95% critical value, for which the
  # standard finite-n approximation is K / (sqrt(n) + 0.12 + 0.11 / sqrt(n))
  # with K = 1.358099.
  for (n in c(50L, 100L, 200L)) {
    b <- sbc_ecdf_band(n, 0.95, type = "ks")
    ref <- 1.358099 / (sqrt(n) + 0.12 + 0.11 / sqrt(n))
    expect_lt(abs(b$par / ref - 1), 1e-3,
              label = sprintf("n = %d KS critical value", n))
  }
})

test_that("a pointwise band is not a simultaneous band", {
  # The whole reason the calibration exists: holding each order statistic at 95%
  # holds all of them together at well under half that, and the calibrated band
  # is at nominal by construction.
  n <- 100L
  pw <- .sbc_bounds(n, "beta", 0.05)
  expect_lt(sbc_crossing_prob(pw$g, pw$h), 0.55)
  b <- sbc_ecdf_band(n, 0.95)
  expect_equal(b$coverage, 0.95, tolerance = 1e-5)
  expect_lt(b$par, 0.05)
})

test_that("the bands' realized simultaneous coverage is nominal", {
  skip_on_cran()
  # The arbiter for the word "simultaneous": simulate uniform samples and
  # measure how often the whole ECDF stays inside.
  set.seed(31L)
  n <- 100L; B <- 20000L
  bb <- sbc_ecdf_band(n, 0.95)
  kk <- sbc_ecdf_band(n, 0.95, type = "ks")
  U <- matrix(stats::runif(n * B), B, n)
  U <- t(apply(U, 1L, sort))
  hb <- mean(apply(U, 1L, function(u) all(u >= bb$g) && all(u <= bb$h)))
  hk <- mean(apply(U, 1L, function(u) all(u >= kk$g) && all(u <= kk$h)))
  se <- sqrt(0.95 * 0.05 / B)
  expect_lt(abs(hb - 0.95), 4 * se)
  expect_lt(abs(hk - 0.95), 4 * se)
  # And the exact simultaneous p-value is itself uniform, which is what makes
  # `p_unif` in the report comparable across arms.
  p <- apply(U[seq_len(400L), , drop = FALSE], 1L,
             function(u) sbc_ecdf_test(u)$p_value)
  expect_lt(abs(mean(p < 0.05) - 0.05), 4 * sqrt(0.05 * 0.95 / 400))
})

# ---------------------------------------------------------------------------
# 4. The driver
# ---------------------------------------------------------------------------

test_that("CRPS is refused as a posterior ranking on a fixed-truth experiment", {
  sim <- function(seed) list(theta = c(mu = 0.5))
  fit <- function(d) list(a = list(mu = sbc_normal(0.5, 1)),
                          b = list(mu = sbc_normal(0.5, 2)))
  fx <- recov_sbc(sim, fit, n_seed = 3L, truth = "fixed")
  expect_identical(attr(fx, "crps_role"), "descriptive loss (fixed truth)")
  expect_error(sbc_crps_compare(fx, "a"), "prior-predictive")
  pr <- recov_sbc(sim, fit, n_seed = 3L, truth = "prior_draw")
  expect_identical(attr(pr, "crps_role"), "proper posterior score")
  cmp <- sbc_crps_compare(pr, "a")
  expect_s3_class(cmp, "data.frame")
  expect_identical(cmp$arm, "b")
})

test_that("asking for the diagnostic leaves the ambient RNG stream where it was", {
  sim <- function(seed) { set.seed(seed); list(theta = c(mu = stats::rnorm(1))) }
  fit <- function(d) list(a = list(mu = sbc_normal(0, 1)))
  set.seed(77L)
  before <- get(".Random.seed", envir = globalenv())
  invisible(recov_sbc(sim, fit, n_seed = 4L))
  set.seed(77L)
  expect_identical(get(".Random.seed", envir = globalenv()), before)
})

test_that("the driver recovers uniformity on a conjugate model it can solve exactly", {
  skip_on_cran()
  # A normal-normal with a fully proper prior, no engine involved: theta ~
  # N(0, 1), y_i ~ N(theta, 1) for i = 1..8. The exact posterior is conjugate,
  # so the `exact` arm must be uniform and the two mis-scaled arms must not --
  # the driver's own end-to-end check, independent of anything tulpa computes.
  n <- 400L
  sim <- function(seed) {
    set.seed(seed)
    th <- stats::rnorm(1)
    list(y = stats::rnorm(8L, th, 1), theta = c(mu = th))
  }
  fit <- function(d) {
    v <- 1 / (1 + length(d$y))
    m <- v * sum(d$y)
    list(exact = list(mu = sbc_normal(m, sqrt(v))),
         wide  = list(mu = sbc_normal(m, sqrt(v) * 1.3)),
         narrow = list(mu = sbc_normal(m, sqrt(v) / 1.3)))
  }
  res <- recov_sbc(sim, fit, n_seed = n, seed_off = 900L)
  band <- sbc_ecdf_band(n, 0.95)
  u <- function(a) res$pit[res$arm == a]
  expect_true(sbc_ecdf_inside(u("exact"), band))
  expect_false(sbc_ecdf_inside(sbc_fold(u("wide")), band))
  expect_false(sbc_ecdf_inside(sbc_fold(u("narrow")), band))
  # And the proper score ranks the exact posterior first, on paired seeds.
  # Measured on these fixed seeds: t = 3.88 for the 30%-wide arm and 2.57 for
  # the 30%-narrow one, both against a baseline that is the true posterior.
  cmp <- sbc_crps_compare(res, baseline = "exact")
  expect_true(all(cmp$delta > 0))
  expect_gt(min(cmp$t), 2)
})

# ---------------------------------------------------------------------------
# 5. The engine fixture
# ---------------------------------------------------------------------------

test_that("the nested fit reproduces the exact posterior of the SBC fixture", {
  skip_on_cran()
  # Everything the acceptance measurement reads rests on this: the gaussian
  # log-likelihood is quadratic in eta, so the inner Laplace is exact and the
  # engine's per-cell modes, per-cell variances and cell weights must equal the
  # independently derived closed form.
  for (cfg in list(list(6L, 4L), list(10L, 5L))) {
    d <- sbc_sim_gaussian(1234L, nr = cfg[[1]], spr = cfg[[2]])
    f <- sbc_fit_nested(d)
    mom <- .nested_fixed_moments(f)
    E <- sbc_exact_post(d)
    lbl <- sprintf("nr = %d, spr = %d", cfg[[1]], cfg[[2]])
    expect_equal(as.numeric(f$theta_grid), d$grid, tolerance = 1e-12, label = lbl)
    expect_lt(max(abs(mom$w - E$w)), 1e-5, label = paste(lbl, "weights"))
    expect_lt(max(abs(mom$mu - E$mu)), 1e-4, label = paste(lbl, "modes"))
    expect_lt(max(abs(mom$var - E$var)), 1e-4, label = paste(lbl, "variances"))
    # The engine's log marginal differs from the exact one by a constant only,
    # which is what makes the cell weights the exact posterior on sigma.
    expect_lt(stats::sd(f$log_marginal - E$log_marg), 1e-4, label = paste(lbl, "log marginal"))
  }
})

# MEASURED, 2000 prior-predictive replicates of the nr = 6, spr = 4 gaussian
# fixture: sigma drawn from the engine's own seven-cell grid prior, beta fixed
# (exactly uniform either way, by the location-invariance argument in
# helper-sbc.R). `ks` is the Kolmogorov-Smirnov departure of the PIT sample from
# uniform and `p` the exact simultaneous p-value in the equal-local-levels
# family. The 95% band rejects at about ks = 0.030 in the middle of the range.
#
#                          raw ECDF              folded ECDF
#   arm          quantity   ks       p            ks       p
#   exact        beta1     0.0143   4.40e-01     0.0191   4.56e-01
#   mixture      beta1     0.0143   4.40e-01     0.0191   4.56e-01
#   collapsed    beta1     0.0199   1.94e-04     0.0339   2.50e-04
#   wide         beta1     0.0724   6.34e-14     0.1358   7.51e-14
#   narrow       beta1     0.0488   0            0.0870   0
#   phi_crossed  beta1     0.0228   3.58e-02     0.0370   1.45e-02
#   exact        beta2     0.0158   8.17e-01     0.0117   9.66e-01
#   mixture      beta2     0.0158   8.16e-01     0.0117   9.66e-01
#   collapsed    beta2     0.0159   8.08e-01     0.0118   9.64e-01
#   phi_crossed  beta2     0.0907   0            0.1673   0
#   mixture      sigma     0.0216   8.04e-01     0.0272   8.23e-02
#   phi_crossed  sigma     0.0888   6.61e-14     0.0692   4.20e-13
#   mixture      log_lik   0.0120   4.57e-01     0.0133   2.47e-01
#   phi_crossed  log_lik   0.1731   7.88e-14     0.0922   0
#
# THE ACCEPTANCE READING. The mixture read of the intercept is uniform and the
# collapsed Gaussian of the SAME two moments is not, at p = 1.9e-4 raw and
# 2.5e-4 folded, on seeds where a coverage indicator at two nominal levels moved
# 0 or 1 trials of 200 (gcol33/tulpa#336). The slope separates neither read,
# which is the same thing gcol33/tulpa#325 found by a different route: a
# within-group contrast barely reads the outer grid, so there is no shape there
# to discard. The full table and the CRPS half of the answer are in
# dev_notes/issue335/RESULTS.md.
#
# The gate below runs the first 300 of those seeds. It judges the reference arms
# against a 99.9% band -- sixteen in-band assertions at 95% would fail
# spuriously a third of the time, which is a gate that cannot mean anything --
# and the broken arms against the 95% band, where the power is. At 300 seeds the
# collapsed read is inside both bands: separating it takes the full run, which
# is why the number above is a comment and not an assertion.
test_that("SBC and CRPS pass the reference reads and catch the broken ones", {
  skip_if_not_slow()
  n <- 300L
  # Through the EXPORTED front door (gcol33/tulpa#380). `flat_prior` is what
  # this fixture's own uniformity argument is: beta is held fixed under the
  # nested door's flat prior, and the PIT is uniform there by the location-
  # parameter argument in section 7 of helper-sbc.R, not by having been drawn.
  fit_sbc <- sbc("prior_predictive", simulator = sbc_sim_gaussian,
                 fitter = sbc_arms_gaussian, n_sim = n, seed = 500000L,
                 flat_prior = c("beta1", "beta2"))
  res <- fit_sbc$pit
  # The door changes nothing about the experiment it runs.
  expect_identical(res, recov_sbc(sbc_sim_gaussian, sbc_arms_gaussian,
                                  n_seed = n, seed_off = 500000L,
                                  truth = "prior_draw"))
  u <- function(a, q) res$pit[res$arm == a & res$quantity == q]
  wide_band <- sbc_ecdf_band(n, 0.999)
  band <- sbc_ecdf_band(n, 0.95)

  for (a in c("exact", "mixture")) {
    for (q in c("beta1", "beta2", "sigma", "log_lik")) {
      expect_true(sbc_ecdf_inside(u(a, q), wide_band),
                  label = sprintf("%s / %s inside the 99.9%% band", a, q))
      expect_true(sbc_ecdf_inside(sbc_fold(u(a, q)), wide_band),
                  label = sprintf("%s / %s folded inside the 99.9%% band", a, q))
    }
  }
  # The engine's mixture read and the independently computed exact posterior are
  # the same posterior to 7.3e-06 in the PIT, so nothing below is the solve.
  expect_lt(max(abs(u("exact", "beta1") - u("mixture", "beta1"))), 1e-4)
  expect_lt(max(abs(u("exact", "beta2") - u("mixture", "beta2"))), 1e-4)

  # The harness can fail. A posterior mis-scaled by 25% either way is a
  # symmetric dispersion error, so the FOLDED read is what has to catch it: the
  # over-dispersed arm's raw ECDF at ks = 0.0716 on the slope sits inside the
  # 99.9% band while its folded ECDF at 0.1166 is outside, which is the reason
  # the folded rank is in the harness at all.
  for (a in c("wide", "narrow")) {
    for (q in c("beta1", "beta2")) {
      expect_false(sbc_ecdf_inside(sbc_fold(u(a, q)), band),
                   label = sprintf("%s / %s folded outside", a, q))
    }
  }
  expect_false(sbc_ecdf_inside(sbc_fold(u("wide", "beta2")), wide_band))

  # The gcol33/tulpa#332 crossing -- a fit at phi^2 where the door reads a
  # residual SD -- shows on the slope, the hyperparameter and the joint
  # log-likelihood. Its INTERCEPT stays inside both bands, which is the
  # gcol33/tulpa#325 attribution again: the intercept's posterior is carried by
  # the RE-SD grid and barely reads the residual scale. A harness with only
  # per-coefficient marginals on the intercept would have missed this fit; the
  # joint log-likelihood rank has ks = 0.202 on it.
  for (q in c("beta2", "sigma", "log_lik")) {
    expect_false(sbc_ecdf_inside(u("phi_crossed", q), band),
                 label = sprintf("phi_crossed / %s outside", q))
  }

  # The proper score. CRPS is proper in this prior-predictive experiment, so the
  # exact posterior minimizes it: the engine's read scores within 2e-05 of it
  # relatively, every deliberately broken arm pays in the mean, and the
  # collapsed read does not score better than the mixture it compresses.
  cmp <- summary(fit_sbc, baseline = "mixture")$compare
  expect_identical(cmp, sbc_crps_compare(res, baseline = "mixture"))
  b1 <- cmp[cmp$quantity == "beta1", ]
  b2 <- cmp[cmp$quantity == "beta2", ]
  expect_lt(abs(b1$delta[b1$arm == "exact"]) /
              mean(res$crps[res$quantity == "beta1"], na.rm = TRUE), 1e-4)
  expect_gt(b1$t[b1$arm == "wide"], 3)
  expect_gt(b2$t[b2$arm == "wide"], 2)
  expect_gt(b1$delta[b1$arm == "narrow"], 0)
  expect_gt(b2$delta[b2$arm == "narrow"], 0)
  expect_gt(b1$delta[b1$arm == "collapsed"], -1e-5)
  expect_gt(b2$delta[b2$arm == "collapsed"], -1e-5)

  # And the same verdicts read off the front door's own report, at its own
  # nominal level: the two reference arms inside the band on every quantity, the
  # three known-bad controls outside on at least one of theirs.
  rp <- fit_sbc$report
  ok <- rp[rp$arm %in% c("exact", "mixture"), ]
  expect_true(all(ok$inside), label = "reference arms inside the 95% band")
  expect_true(all(ok$inside_folded), label = "reference arms folded inside")
  for (a in c("wide", "narrow", "phi_crossed")) {
    bad <- rp[rp$arm == a, ]
    expect_true(any(!bad$inside | !bad$inside_folded),
                label = sprintf("%s outside the band", a))
  }
})
