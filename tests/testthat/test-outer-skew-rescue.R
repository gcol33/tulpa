# Outer Pareto-k on a collapsed grid: the skew-normal proposal rescue and the
# grid-regime classifier (gcol33/tulpa#276).
#
# The failure this closes: on a sharp hyperparameter posterior the outer grid
# collapses onto ~1 cell, the grid-mixture rescue cannot help (its few bumps
# cover worse than the Gaussian), and the k-hat is then scored against a
# SYMMETRIC Gaussian on a right-skewed variance-component marginal. The k tracks
# the collapse and the marginal's asymmetry rather than the fit, so a downstream
# bare-k threshold mis-bins healthy fits as unreliable.
#
# The safety property the rescue rests on is structural: a skew-normal has
# GAUSSIAN tails on both sides, so it can absorb ASYMMETRY but never a heavy
# TAIL. The tests below assert both directions -- a skewed marginal is repaired,
# and a heavy-tailed one is evaluated and REJECTED rather than laundered.

# --------------------------------------------------------------------------- #
# (1) The skew-normal proposal family                                          #
# --------------------------------------------------------------------------- #

test_that("the skew-normal proposal round-trips the moments it is built from", {
  set.seed(1)
  for (g in c(-0.8, -0.3, 0, 0.4, 0.9)) {
    dp <- .sn_prop_from_moments(mu = 0.5, sd = 2, skew = g)
    expect_false(is.null(dp))
    x <- .sn_prop_rand(2e5, dp)[, 1]
    m <- mean(x); s <- stats::sd(x)
    expect_equal(m, 0.5, tolerance = 0.05)
    expect_equal(s, 2,   tolerance = 0.05)
    expect_equal(mean(((x - m) / s)^3), g, tolerance = 0.06)
  }
})

test_that("the proposal log-density is a proper density and matches its sampler", {
  dp <- .sn_prop_from_moments(mu = 0.5, sd = 2, skew = 0.7)
  # Integrates to 1: the density is normalized, not merely proportional.
  I <- stats::integrate(
    function(z) exp(.sn_prop_logpdf(matrix(z, ncol = 1), dp)), -Inf, Inf)$value
  expect_equal(I, 1, tolerance = 1e-6)

  # Sampler and density are consistent: importance sampling a KNOWN target
  # through this proposal must recover that target's mean.
  set.seed(2)
  U  <- .sn_prop_rand(2e5, dp)
  lw <- stats::dnorm(U[, 1], 1.5, 0.8, log = TRUE) - .sn_prop_logpdf(U, dp)
  w  <- exp(lw - max(lw)); w <- w / sum(w)
  expect_equal(sum(w * U[, 1]), 1.5, tolerance = 0.02)
})

test_that("the multi-axis proposal is an independent product with per-axis skew", {
  set.seed(3)
  dp <- .sn_prop_from_moments(mu = c(0, 1), sd = c(1, 0.5), skew = c(0.9, -0.5))
  X <- .sn_prop_rand(2e5, dp)
  sk <- function(v) mean(((v - mean(v)) / stats::sd(v))^3)
  expect_equal(sk(X[, 1]),  0.9, tolerance = 0.06)
  expect_equal(sk(X[, 2]), -0.5, tolerance = 0.06)
  expect_equal(mean(X[, 2]), 1, tolerance = 0.02)
  expect_equal(stats::cor(X[, 1], X[, 2]), 0, tolerance = 0.02)
})

test_that("skewness beyond the skew-normal ceiling is clamped, not refused", {
  # sn_match() returns NULL past the ~0.995 ceiling; the PROPOSAL path clamps
  # inside it instead, because a proposal only has to COVER the target.
  expect_warning(expect_null(sn_match(0, 1, 3)))
  dp <- .sn_prop_from_moments(mu = 0, sd = 1, skew = 3)
  expect_false(is.null(dp))
  expect_true(all(is.finite(unlist(dp))))
})

# --------------------------------------------------------------------------- #
# (2) The rescue: repairs skew, never launders a heavy tail                    #
# --------------------------------------------------------------------------- #

# The #276 regime: a SHARP hyperparameter posterior (so the grid collapses onto
# one cell) on a WIDE default field-SD grid.
.osk_S    <- 0.3
.osk_U0   <- log(1.5)
.osk_prep <- function() {
  u_grid <- matrix(log(c(0.25, 0.5, 1.5, 3, 5)), ncol = 1,
                   dimnames = list(NULL, "sigma"))
  w <- c(1e-6, 1e-4, 1 - 1.1e-3, 1e-3, 1e-6); w <- w / sum(w)
  list(tags = "log", u_grid = u_grid, u_hat = .osk_U0,
       Su = matrix(.osk_S^2, 1, 1), cn = "sigma", d = 1L, w = w,
       proposal_source = "mode_hessian")
}
# `shape_z` is the standardized marginal's log-density; the target is that shape
# placed at U0 with scale S. The geometric grid makes log_marginal the u-space
# target with no Jacobian, matching the integrator's own weighting.
.osk_score <- function(shape_z, n = 4000L, seed = 11L) {
  prep  <- .osk_prep()
  refit <- function(theta_mat)
    shape_z((log(as.numeric(theta_mat[, 1])) - .osk_U0) / .osk_S) - log(.osk_S)
  spec <- .joint_cand_spec(prep, 1L, refit)
  # The first-pass Gaussian is scored at the tail size the DISPATCH will use,
  # not at the automatic rule's: scoring the two arms under different tail rules
  # compares the rules rather than the proposals (gcol33/tulpa#631).
  tp <- .k_outer_tail_points(n)
  set.seed(seed); g <- .k_score_gaussian(spec, n, tail_points = tp)
  set.seed(seed); d <- .k_dispatch(spec, n)
  list(gauss = g$pareto_k, k = d$best$pareto_k, src = d$source, skew = d$outer_skew)
}

test_that("a skewed marginal needs no rescue, at any budget", {
  skip_on_cran()
  # This asserted the opposite until gcol33/tulpa#631's tail rule landed: that a
  # skewness-0.9 target reads UNRELIABLE on a symmetric proposal and is repaired
  # by the skew-normal one. Measured, that reading was the BUDGET, not the
  # target. Under the automatic PSIS rule the same target climbs
  # 0.017 / 0.290 / 0.597 / 0.914 / 1.627 over 500 to 10000 draws, crossing both
  # bands, and this fixture only reached the rescue because it scores at 4000 --
  # eight times the shipped budget. At the shipped budget it reads 0.017.
  #
  # That is gcol33/tulpa#629's finding from the other side: a Gaussian proposal's
  # importance ratio stays BOUNDED on a skew-normal target (Gaussian tail one
  # side, lighter the other), so skewness alone does not inflate an outer k-hat
  # -- a heavy tail does. With the tail fraction held, the read agrees with the
  # default-budget read at every budget and no rescue is called for.
  sn9 <- .sn_prop_from_moments(0, 1, 0.9)
  shape <- function(z) .sn_prop_logpdf(matrix(z, ncol = 1), sn9)
  ks <- vapply(c(500L, 2000L, 4000L, 10000L),
               function(n) .osk_score(shape, n = n)$k, numeric(1))
  expect_true(all(ks < .nl_diag("gamma3_ok")),
              info = paste(round(ks, 3), collapse = " "))   # good band throughout
  expect_lt(max(ks) - min(ks), 0.25)                        # and budget-stable
  # No rescue: there is no bad band to rescue it out of.
  expect_false(identical(.osk_score(shape, n = 4000L)$src, "skew_normal"))
})

test_that("a symmetric heavy tail is left flagged (the rescue does not engage)", {
  skip_on_cran()
  r <- .osk_score(function(z) stats::dt(z, df = 2, log = TRUE))
  expect_lt(abs(r$skew[[1]]), 0.2)              # no asymmetry to remove
  expect_false(identical(r$src, "skew_normal"))
  expect_gte(r$k, r$gauss - 1e-9)               # k not lowered
})

test_that("a skewed HEAVY tail runs the rescue and rejects it", {
  skip_on_cran()
  # The decisive no-laundering case: skewness clears the gate, so the skew
  # proposal is actually built and scored -- and must lose, because its Gaussian
  # tails cannot cover a heavy one. "Evaluated and rejected", not "skipped".
  #
  # The tilt is `z`, not `3 z`, since gcol33/tulpa#631's held tail fraction: at
  # the stronger tilt moment matching now clears this target into the good band
  # (k-hat 0.277) and the dispatch stops before the rescue, so the case would
  # test nothing. At this tilt the Gaussian still reads 0.743 with skewness
  # 0.583, which is what puts the rescue on the table at all.
  r <- .osk_score(function(z) log(2) + stats::dt(z, df = 2, log = TRUE) +
                    stats::pnorm(z, log.p = TRUE))
  expect_gt(abs(r$skew[[1]]), 0.2)
  expect_false(identical(r$src, "skew_normal"))
  expect_gte(r$k, r$gauss - 1e-9)
})

test_that("a proposal that already fits pays nothing for the rescue", {
  skip_on_cran()
  r <- .osk_score(function(z) stats::dnorm(z, log = TRUE))
  expect_lt(r$k, 0.5)
  expect_false(identical(r$src, "skew_normal"))
  expect_null(r$skew)                           # never even estimated
})

test_that("the significance screen keeps a gaussian target off the skew proposal", {
  skip_on_cran()
  # The skewness is ESTIMATED, so a bare magnitude floor fires on noise: at 200
  # draws its standard error is ~0.17. Without the .K_DIAG_SKEW_Z screen a
  # GAUSSIAN outer target adopted the skew proposal in 18% of RNG states. The
  # reported proposal source must not depend on the seed.
  prep <- .osk_prep()
  # Slightly narrower than the proposal, so the importance ratios genuinely vary
  # (a target EQUAL to the proposal gives a constant ratio and a degenerate GPD
  # fit) while the Gaussian still covers it well.
  s_t   <- 0.85 * .osk_S
  refit <- function(theta_mat)
    stats::dnorm((log(as.numeric(theta_mat[, 1])) - .osk_U0) / s_t,
                 log = TRUE) - log(s_t)
  src <- vapply(seq_len(40), function(i) {
    set.seed(500L + i)
    .k_dispatch(.joint_cand_spec(prep, 1L, refit), 200L)$source %||% NA_character_
  }, character(1))
  expect_true(any(!is.na(src)))                 # the pass really ran
  expect_false(any(stats::na.omit(src) == "skew_normal"))
})

test_that(".skew_se is the normal-theory standard error of a sample skewness", {
  # sqrt(6n(n-1) / ((n-2)(n+1)(n+3))), approaching sqrt(6/n).
  expect_equal(.skew_se(200), sqrt(6 * 200 * 199 / (198 * 201 * 203)))
  expect_equal(.skew_se(1e6), sqrt(6 / 1e6), tolerance = 1e-4)
  expect_identical(.skew_se(3), Inf)            # too few draws to screen
  expect_identical(.skew_se(NA_real_), Inf)
})

test_that("weighted moments are read in the proposal's whitened coordinate", {
  # Draws from N(mu, s^2) whitened by that same proposal must come back with
  # whitened mean ~0, sd ~1 and skew ~0 at equal weights -- so a reported
  # skewness is a statement about the TARGET, not about the parameterization.
  set.seed(7)
  u_c <- 2.5; L <- matrix(0.4, 1, 1)
  U <- matrix(stats::rnorm(4e4, u_c, 0.4), ncol = 1)
  m <- .k_wtd_moments(U, rep(0, nrow(U)), u_c, L)
  expect_equal(m$mu[1],   0, tolerance = 0.02)
  expect_equal(m$sd[1],   1, tolerance = 0.02)
  expect_equal(m$skew[1], 0, tolerance = 0.05)
  expect_equal(m$n_eff, nrow(U), tolerance = 1e-8)   # equal weights
})

# --------------------------------------------------------------------------- #
# (3) The grid-regime classifier                                              #
# --------------------------------------------------------------------------- #

test_that("the grid regime separates a spread grid from a collapsed one", {
  gr <- cbind(sigma = c(0.25, 0.5, 1.5, 3, 5))
  mk <- function(w) list(theta_grid = gr, weights = w)

  expect_identical(.joint_pareto_grid_regime(mk(rep(0.2, 5)))$regime, "spread")

  r <- .joint_pareto_grid_regime(mk(c(0, 0, 1, 0, 0)))
  expect_identical(r$regime, "collapsed_interior")
  expect_length(r$edge_axes, 0L)
  expect_equal(r$ess_grid, 1)
})

test_that("a collapsed mode against a grid boundary is named with its side", {
  gr <- cbind(sigma = c(0.25, 0.5, 1.5, 3, 5))
  mk <- function(w) list(theta_grid = gr, weights = w)

  hi <- .joint_pareto_grid_regime(mk(c(0, 0, 0, 0, 1)))
  expect_identical(hi$regime, "collapsed_edge")
  expect_identical(hi$edge_axes, "sigma")
  expect_identical(hi$edge_sides, "upper")

  lo <- .joint_pareto_grid_regime(mk(c(1, 0, 0, 0, 0)))
  expect_identical(lo$regime, "collapsed_edge")
  expect_identical(lo$edge_sides, "lower")
})

test_that("a pinned axis is not mistaken for a grid boundary", {
  # A copy `alpha` fixed at 0 (or a one-point dispersion axis) has a single grid
  # value: it is PINNED, not "against the edge", and must not raise the
  # widen-the-grid recommendation.
  gr <- cbind(sigma = c(0.25, 0.5, 1.5), alpha = c(0, 0, 0))
  r <- .joint_pareto_grid_regime(list(theta_grid = gr, weights = c(0, 1, 0)))
  expect_identical(r$regime, "collapsed_interior")
  expect_length(r$edge_axes, 0L)
})

test_that("the regime declines rather than guessing on an unusable grid", {
  expect_null(.joint_pareto_grid_regime(list(theta_grid = NULL, weights = 1)))
  expect_null(.joint_pareto_grid_regime(
    list(theta_grid = cbind(a = 1:3), weights = c(0, 0, 0))))
})

test_that("the regime note explains a collapse and is silent on a spread grid", {
  expect_null(.tulpa_outer_regime_note(
    list(regime = "spread", edge_axes = character(0), edge_sides = character(0))))
  n1 <- .tulpa_outer_regime_note(
    list(regime = "collapsed_interior", edge_axes = character(0),
         edge_sides = character(0)))
  expect_match(n1, "not integrated")
  n2 <- .tulpa_outer_regime_note(
    list(regime = "collapsed_edge", edge_axes = "sigma", edge_sides = "upper"))
  expect_match(n2, "sigma \\(upper\\)")
  expect_match(n2, "widen")
})
