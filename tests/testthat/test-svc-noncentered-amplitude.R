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
# parameterizations are now checked against the same band instead. Centered
# reading *above* non-centered's own recovery on this fixture (measured
# 1.35-1.6 against non-centered's 1.0-1.3) is tracked separately in #842 --
# open question, not asserted here either way.

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
})

test_that("centered SVC NUTS also recovers a weakly identified field's amplitude", {
  skip_if_not_slow()
  d <- sim_svc_bernoulli(n = 150L, seed = 1L)
  fit <- fit_svc_amp(d, "centered")

  ratio <- svc_sd_ratio(fit, d$w_true)
  # Same band as the non-centered test above, now that 34c9cb5b (#841) has
  # removed the funnel that used to attenuate this branch to ~0.33. The
  # upper bound is wider here: centered has measured 1.35-1.6 on this
  # fixture (#842, open), against non-centered's 1.0-1.3 -- not yet
  # established as correct or as an overshoot, so this only catches the
  # funnel coming back (too small) or a runaway (far too large), not the
  # narrower question #842 tracks.
  expect_gt(ratio, 0.55)
  expect_lt(ratio, 2.2)
  expect_lte(mean(fit$divergent), 0.05)
})
