# Front-door exact NUTS for spatially-varying coefficients (gcol33/tulpa#158d).
# tulpa(y ~ x, spatial = spatial_svc(~ lon + lat, terms = ~ x - 1), mode =
# "exact") routes through the generic ModelData sampler, which allocates the
# per-term (sigma2, phi) hypers and the [n_svc x n_obs] NNGP field (keyed on
# data.has_svc) and samples them jointly. eta_i += sum_j X_svc[i,j] w_j(s_i),
# so the fixed slope absorbs the mean of the varying coefficient and the SVC
# field carries the spatial variation. SVC is exact-only (no nested front door).

make_svc_pois <- function(n = 120L, seed = 404L) {
  set.seed(seed)
  lon <- runif(n); lat <- runif(n)
  bsurf <- 0.9 * sin(2.8 * lon) + 0.7 * cos(2.2 * lat)   # varying part of x-slope
  x <- rnorm(n)
  d <- data.frame(lon = lon, lat = lat, x = x)
  d$y <- rpois(n, exp(0.2 + (0.8 + bsurf) * x))
  list(d = d, bsurf = bsurf)
}

test_that("svc requires an exact mode (nested/laplace refuse)", {
  s <- make_svc_pois(n = 30L)
  expect_error(
    tulpa(y ~ x, data = s$d, family = "poisson",
          spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 5L),
          mode = "structured"),
    "Spatially-varying"
  )
})

test_that("svc exact NUTS allocates the per-term field (structural)", {
  skip_if_not_slow()
  s <- make_svc_pois(n = 40L)
  fit <- tulpa(y ~ x, data = s$d, family = "poisson",
               spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 5L),
               mode = "exact",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L))
  pn <- colnames(fit$draws)
  expect_true(all(c("log_sigma2_svc[1]", "log_phi_svc[1]") %in% pn))
  expect_equal(sum(grepl("^svc_w\\[", pn)), 40L)   # one field per obs, one term
  expect_equal(ncol(fit$draws), 4L + 40L)           # 2 fixed + 2 hypers + field
})

test_that("svc exact NUTS recovers the varying-coefficient surface", {
  skip_if_not_slow()
  s <- make_svc_pois()
  fit <- tulpa(y ~ x, data = s$d, family = "poisson",
               spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 8L),
               mode = "exact",
               control = list(n_iter = 350L, n_warmup = 175L, seed = 5L))
  wcol <- grep("^svc_w\\[", colnames(fit$draws))
  fm <- colMeans(fit$draws[, wcol, drop = FALSE])   # w_1(s_i): centred varying slope
  # The SVC field recovers the spatial surface; the fixed x-coefficient absorbs
  # its (non-zero) mean, so recovery is on the centred surface.
  expect_gt(cor(fm - mean(fm), s$bsurf - mean(s$bsurf)), 0.8)
})
