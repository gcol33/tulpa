test_that("adaptive quadrature covers a dense HSGP reference at every order", {
  skip_if_not_slow()
  d <- read.csv(test_path("fixtures", "outer-hsgp120.csv"))
  reference <- dget(test_path("fixtures", "outer-hsgp120-reference.R"))
  fit <- suppressMessages(tulpa(y ~ x, data = d, family = "poisson",
    spatial = spatial_multiscale(~ lon + lat,
      range_local = c(.02, .15), range_regional = c(.15, 1),
      approx = "hsgp", m = 6L, scale_coords = FALSE),
    control = list(progress = FALSE, n_threads = 2L)))
  expect_outer_quadrature_reference(fit, reference)
})
