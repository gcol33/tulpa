# predict() kriging for a GP / NNGP fit (#158). The field is interpolated to new
# coordinates by the NNGP conditional mean (cpp_gp_field_predict) at each new
# location's nearest training locations, marginalised over the hyperparameter
# grid. Correctness anchor: the kriged field recovers a known smooth surface at
# held-out coordinates, and the training-location field recovers it too.

test_that("GP/NNGP predict krige recovers a smooth surface at new coordinates", {
  skip_on_cran()
  set.seed(202)
  truef <- function(lon, lat) 1.3 * sin(3.2 * lon) + 1.0 * cos(2.8 * lat)
  n <- 250L
  lon <- runif(n); lat <- runif(n)
  d <- data.frame(lon = lon, lat = lat, x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x + truef(lon, lat)))

  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, nn = 10L))
  expect_identical(fit$spatial$type, "gp")

  gg <- expand.grid(lon = seq(0.05, 0.95, length.out = 12),
                    lat = seq(0.05, 0.95, length.out = 12))
  gg$x <- 0
  eta_full  <- predict(fit, newdata = gg, type = "link", include_field = TRUE)
  eta_fixed <- predict(fit, newdata = gg, type = "link", include_field = FALSE)
  field_pred <- eta_full - eta_fixed

  expect_true(any(abs(field_pred) > 1e-6))
  field_true <- truef(gg$lon, gg$lat)
  cc <- cor(field_pred - mean(field_pred), field_true - mean(field_true))
  expect_gt(cc, 0.85)
})

test_that("GP/NNGP training-location field (newdata = NULL) recovers the surface", {
  skip_on_cran()
  set.seed(202)
  truef <- function(lon, lat) 1.3 * sin(3.2 * lon) + 1.0 * cos(2.8 * lat)
  n <- 250L
  lon <- runif(n); lat <- runif(n)
  d <- data.frame(lon = lon, lat = lat, x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x + truef(lon, lat)))
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, nn = 10L))

  eta_full  <- predict(fit, type = "link", include_field = TRUE)
  eta_fixed <- predict(fit, type = "link", include_field = FALSE)
  field_tr  <- eta_full - eta_fixed
  expect_true(any(abs(field_tr) > 1e-6))
  ft <- truef(lon, lat)
  expect_gt(cor(field_tr - mean(field_tr), ft - mean(ft)), 0.85)
})
