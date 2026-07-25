# Non-centered NNGP field amplitude recovery (gcol33/tulpa#243).
#
# The centered parameterization samples the GP field jointly with its
# (sigma2, phi) hyperparameters. That is Neal's funnel, and a diagonal mass
# matrix gets stuck in the neck: the posterior-mean field is attenuated toward
# zero and sigma2 is under-estimated, regardless of warmup. #243 rewired the
# sampling path to the non-centered parameterization (sample z ~ N(0, I),
# reconstruct w = f(z, sigma2, phi)), now the default for spatial_gp().
#
# This asserts the fix on the quantity the funnel crushes: sd(field_hat)
# relative to sd(field_truth). Correlation alone did not catch the bug (the
# field was the right shape, just crushed), so the band is two-sided -- too
# small flags the funnel, too large flags an erroneously re-added z -> w
# Jacobian inflating the amplitude. The funnel is most visible in weakly
# identified data (the occupancy repro in the issue); a Poisson GLM keeps the
# field well identified, so this is a recovery guard on the non-centered path
# rather than a centered/non-centered contrast.

# One Poisson observation per unique location, driven by a dense-GP field with
# known amplitude.
sim_gp_amplitude <- function(n = 70L, sigma2 = 1.0, phi = 0.30,
                             a0 = 0.4, seed = 1L) {
  set.seed(seed)
  lon <- runif(n)
  lat <- runif(n)
  D <- as.matrix(dist(cbind(lon, lat)))
  K <- sigma2 * exp(-D / phi)
  L <- chol(K + diag(1e-8, n))
  w <- as.numeric(t(L) %*% rnorm(n))
  y <- rpois(n, exp(a0 + w))
  data.frame(lon = lon, lat = lat, y = y, w_true = w)
}

test_that("non-centered NNGP NUTS recovers the field amplitude", {
  skip_if_not_slow()
  d <- sim_gp_amplitude(n = 70L, sigma2 = 1.0, phi = 0.30, seed = 1L)

  fit <- tulpa(
    y ~ 1, data = d, family = "poisson",
    spatial = spatial_gp(~ lon + lat, nn = 10L),  # non-centered by default
    mode = "exact",
    control = list(n_iter = 500L, n_warmup = 400L, seed = 7L))
  fld <- grep("^gp_w\\[", colnames(fit$draws))
  f_hat <- colMeans(fit$draws[, fld, drop = FALSE])

  # Shape recovered.
  expect_gt(cor(f_hat, d$w_true), 0.65)
  # Amplitude recovered: the posterior-mean field keeps most of the truth's
  # spread (it shrinks somewhat, as any posterior mean does) and is not
  # inflated. The centered funnel put this ratio near 0.02-0.12 on the issue's
  # occupancy data; the two-sided band would fail there and would also fail if
  # a z -> w Jacobian were added.
  sd_ratio <- sd(f_hat) / sd(d$w_true)
  expect_gt(sd_ratio, 0.45)
  expect_lt(sd_ratio, 1.8)
})
