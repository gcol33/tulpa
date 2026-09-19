# Non-centered NNGP SVC field amplitude recovery (gcol33/tulpa#243, #245).
#
# The funnel this guards is only visible when the field is WEAKLY IDENTIFIED --
# the regime the issue measured on occupancy. A one-trial binomial response
# reproduces that shape inside tulpa's own family set (the Poisson fixture in
# test-svc-nuts-frontdoor.R keeps the field well identified, where centered and
# non-centered agree and the contrast says nothing).
#
# Two things had to be true before this test could discriminate. The transform
# itself (#243), and #245: the soft sum-to-zero pin, evaluated on a
# reconstructed w = L z, became a stiff rank-1 direction along the Vecchia
# cascade that a diagonal mass matrix could not precondition -- with it in
# place non-centered scored WORSE than centered here (0.04 against 0.33), the
# opposite of the fix. With the level identified by centering instead, the same
# fixture read at the time:
#
#   centered      0/400 divergent  sd ratio 0.326  cor 0.762  treedepth 6.5
#   non-centered  0/400 divergent  sd ratio 0.977  cor 0.758  treedepth 4.1
#
# That 0.33 was funnel attenuation on the centered path, not correct
# shrinkage -- the centered branch still carried only a soft sum-to-zero
# PENALTY on the raw field, leaving beta_j and mean(w_j) weakly rather than
# exactly identified. #841 bisected a later reading of this same gap (0.33
# vs an already-drifted ~1.6) to 34c9cb5b, which replaced that soft penalty
# with the exact hard mean-centring non-centered already used, on the
# argument that an NNGP field's constant direction is already proper and the
# alias just needs removing from eta, not re-penalizing. That is a real
# identifiability fix, not a regression, so the contrast this test used to
# assert (non-centered beats centered because centered is funnel-attenuated)
# no longer holds by construction -- the funnel it detected is gone. Both
# parameterizations are now checked against the same band instead.
#
# Centered then read ABOVE non-centered's own recovery here, and #842 measured
# what that is: a convergence failure, not a second identification defect. The
# two parameterizations are one model, pinned deterministically in
# test-svc-parameterization-equivalence.R, and on a well-identified fixture
# they agree to -0.002 (paired, 5 seeds, p = 0.64). On this weakly identified
# one the gap is systematic (8/8 seeds, +0.313, p = 0.006) but it is not better
# recovery: correlation with the truth is unchanged (0.511 against 0.515) while
# RMSE is worse (0.892 against 1.022), and sigma2 roughly doubles on the
# centered path on every seed. At 5x this budget Rhat on the centered sigma2
# falls to 1.04, the correlations become equal, and the gap collapses from
# +0.602 to +0.153. It is the ordinary centered funnel between the field and
# its own variance -- which is why non-centered is the default.
#
# So the centered band below is read off a chain that has NOT mixed in sigma2
# at this budget (Rhat 1.25 on the seed pinned here, against 1.02 non-centered).
# It is a regression guard on one seed, not a calibration statement, and the
# convergence cost itself is asserted rather than left implicit.

sim_svc_bernoulli <- function(n = 150L, sigma2 = 1.0, phi = 0.30,
                              a0 = 0.0, x_sd = 0.6, seed = 1L) {
  set.seed(seed)
  lon <- runif(n)
  lat <- runif(n)
  D <- as.matrix(dist(cbind(lon, lat)))
  K <- sigma2 * exp(-D / phi)
  L <- chol(K + diag(1e-8, n))
  w <- as.numeric(t(L) %*% rnorm(n))
  x <- rnorm(n, sd = x_sd)
  y <- rbinom(n, 1, plogis(a0 + (1 + w) * x))
  data.frame(lon = lon, lat = lat, x = x, y = y, w_true = w)
}

fit_svc_amp <- function(d, parameterization) {
  tulpa(y ~ x, data = d, family = "binomial",
        spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 10L,
                              parameterization = parameterization),
        mode = "exact",
        control = list(n_iter = 500L, n_warmup = 400L, seed = 7L))
}

svc_sd_ratio <- function(fit, w_true) {
  wcol <- grep("^svc_w\\[", colnames(fit$draws))
  sd(colMeans(fit$draws[, wcol, drop = FALSE])) / sd(w_true)
}

# Rhat on the field's own variance, through the shipped diagnostic rather than
# a second copy of the estimator. It is the hyperparameter the centered path
# fails to mix in, so it is the one that tells an amplitude read apart from a
# chain that has not settled.
svc_sigma2_rhat <- function(fit) {
  d <- diagnostics(fit, pars = "log_sigma2_svc[1]", measures = "rhat")
  d$rhat[[1L]]
}

test_that("non-centered SVC NUTS recovers a weakly identified field's amplitude", {
  skip_if_not_slow()
  d <- sim_svc_bernoulli(n = 150L, seed = 1L)
  fit <- fit_svc_amp(d, "noncentered")   # the default

  ratio <- svc_sd_ratio(fit, d$w_true)
  # Two-sided, same reasoning as test-gp-noncentered-amplitude.R: too small
  # flags the funnel surviving, too large flags an erroneously re-added
  # z -> w Jacobian inflating the amplitude.
  expect_gt(ratio, 0.55)
  expect_lt(ratio, 1.8)
  # The geometry fix should also leave the chain clean; the pre-#245
  # non-centered path ran at 24% divergent on the identified fixture.
  expect_lte(mean(fit$divergent), 0.05)
  # And this arm DOES mix in the field's variance at this budget (measured
  # 1.02), which is what makes its amplitude read a posterior summary.
  expect_lt(svc_sigma2_rhat(fit), 1.1)
})

test_that("centered SVC NUTS also recovers a weakly identified field's amplitude", {
  skip_if_not_slow()
  d <- sim_svc_bernoulli(n = 150L, seed = 1L)
  fit <- fit_svc_amp(d, "centered")

  ratio <- svc_sd_ratio(fit, d$w_true)
  # Same band as the non-centered test above, now that 34c9cb5b (#841) has
  # removed the funnel that used to attenuate this branch to ~0.33. The upper
  # bound is wider because this arm reads high on an unconverged sigma2 (#842,
  # measured): it catches the level funnel coming back (too small) or a runaway
  # (far too large), and nothing finer -- across seeds this band is left on
  # both sides, so it is a guard on the pinned seed, not a calibration.
  expect_gt(ratio, 0.55)
  expect_lt(ratio, 2.2)
  expect_lte(mean(fit$divergent), 0.05)
  # The measured cost of the centered funnel, recorded as a bound rather than
  # left implicit: sigma2 reaches 1.25 here against non-centered's 1.02 at the
  # same budget, and 1.04 at 5x. Widening past 1.4 means the centered path has
  # got materially worse, not that a threshold was picked generously.
  expect_lt(svc_sigma2_rhat(fit), 1.4)
})
