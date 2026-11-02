# spatial_spde()'s default mesh domain is the convex hull of the sites widened
# by the data-anchored range, so no observation sits in the band near the edge
# where the SPDE's Neumann boundary inflates the field variance.

test_that(".spde_default_boundary() widens the hull by the stated width", {
  sites <- cbind(c(0, 1, 1, 0, 0.5), c(0, 0, 1, 1, 0.5))
  b <- tulpa:::.spde_default_boundary(sites, 0.3)
  # Distance from each boundary vertex to the unit square, and the reach
  # beyond it along each axis.
  d_out <- sqrt(pmax(0, -b[, 1], b[, 1] - 1)^2 + pmax(0, -b[, 2], b[, 2] - 1)^2)
  expect_true(all(abs(d_out - 0.3) < 1e-12))
  expect_equal(range(b[, 1]), c(-0.3, 1.3))
  expect_null(tulpa:::.spde_default_boundary(sites, NULL))
})

skip_on_cran()

test_that("spatial_spde() builds its default mesh on the widened domain", {
  skip_if_fast()
  set.seed(1)
  d <- data.frame(lon = runif(80), lat = runif(80))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.12, 0.3))
  w <- sp$prior_range[1]                      # the data-anchored range
  v <- sp$mesh$vertices
  expect_lt(min(v[, 1]), min(d$lon) - 0.95 * w)
  expect_gt(max(v[, 2]), max(d$lat) + 0.95 * w)
  # A stated boundary is used as given.
  box <- cbind(c(-0.05, 1.05, 1.05, -0.05), c(-0.05, -0.05, 1.05, 1.05))
  sb <- spatial_spde(~ lon + lat, data = d, boundary = box, max_edge = c(0.12, 0.3))
  expect_gt(min(sb$mesh$vertices[, 1]), -0.06)
})
