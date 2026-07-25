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
# fixture reads:
#
#   centered      0/400 divergent  sd ratio 0.326  cor 0.762  treedepth 6.5
#   non-centered  0/400 divergent  sd ratio 0.977  cor 0.758  treedepth 4.1
#
# Same correlation, three times the amplitude, shallower trees: the centered
# 0.33 is funnel attenuation, not correct shrinkage (correct shrinkage would
# cost correlation too). That is why the assertion below is a contrast and not
# just a band -- on this fixture the band alone would pass a centered fit.

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

test_that("non-centered beats centered on the amplitude the funnel attenuates", {
  skip_if_not_slow()
  d <- sim_svc_bernoulli(n = 150L, seed = 1L)
  r_nc <- svc_sd_ratio(fit_svc_amp(d, "noncentered"), d$w_true)
  r_c  <- svc_sd_ratio(fit_svc_amp(d, "centered"), d$w_true)

  # The funnel's signature on identical data and budget. Measured gap is
  # 0.977 vs 0.326; the margin here only asks that non-centered keep
  # meaningfully more of the field's spread, so a seed that moves both
  # does not flip it.
  expect_gt(r_nc, r_c + 0.2)
})
