# predict() kriging for an HSGP fit (#158). The field is grid-marginalised in
# C++ (cpp_hsgp_field_predict): the per-cell latent is spectral-scaled and
# projected through the Laplacian basis at the new coordinates, weighted by the
# hyperparameter-grid posterior weights. Correctness anchor: the kriged field at
# held-out coordinates recovers a known smooth surface.

test_that("HSGP predict adds a kriged field that recovers a smooth surface", {
  skip_on_cran()
  set.seed(101)
  truef <- function(lon, lat) 1.2 * sin(3 * lon) + 0.9 * cos(2.5 * lat)
  n <- 400L
  lon <- runif(n); lat <- runif(n)
  d <- data.frame(lon = lon, lat = lat, x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x + truef(lon, lat)))

  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 10))

  # New grid: the field is the difference between the full and fixed-effect
  # linear predictors.
  gg <- expand.grid(lon = seq(0.05, 0.95, length.out = 12),
                    lat = seq(0.05, 0.95, length.out = 12))
  gg$x <- 0
  eta_full  <- predict(fit, newdata = gg, type = "link", include_field = TRUE)
  eta_fixed <- predict(fit, newdata = gg, type = "link", include_field = FALSE)
  field_pred <- eta_full - eta_fixed

  expect_true(any(abs(field_pred) > 1e-6))          # a field was actually added
  field_true <- truef(gg$lon, gg$lat)
  # Centre both (the HSGP field and the intercept are confounded in level).
  cc <- cor(field_pred - mean(field_pred), field_true - mean(field_true))
  expect_gt(cc, 0.9)
})

test_that("HSGP basis rebuild at the training coords is byte-identical", {
  skip_on_cran()
  set.seed(7); n <- 120L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  d$y <- rpois(n, exp(0.4 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 8))
  cm <- fit$spatial$coords_matrix
  reb <- cpp_hsgp_basis_2d(matrix(as.numeric(cm), nrow(cm), 2), 8L,
                           as.numeric(fit$spatial[["c"]]))
  # The refactored setup (fill-basis at explicit centre/L) must reproduce the
  # basis the fit stored.
  expect_equal(reb$phi_basis, fit$prior$phi_basis, tolerance = 1e-12)
})

test_that("include_field = FALSE gives the fixed-effect prediction", {
  skip_on_cran()
  set.seed(3); n <- 150L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.6 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 8))
  gg <- data.frame(lon = c(0.2, 0.8), lat = c(0.3, 0.7), x = c(0, 0))
  pf <- predict(fit, newdata = gg, type = "link", include_field = FALSE)
  # Pure fixed-effect: X beta at x = 0 is the intercept for both rows.
  expect_equal(pf[1], pf[2], tolerance = 1e-10)
})
