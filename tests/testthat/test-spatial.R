test_that("spatial_car creates valid specification", {
  adj <- matrix(c(0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0), 4, 4)
  spec <- spatial_car(adjacency = adj, level = "group", group_var = "site")
  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "car")
  expect_equal(spec$n_spatial, 4)
})

test_that("spatial_bym2 creates valid specification", {
  adj <- matrix(c(0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0), 4, 4)
  spec <- spatial_bym2(adjacency = adj, level = "group", group_var = "site")
  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "bym2")
})

test_that("spatial_gp(approx=hsgp) creates valid specification", {
  spec <- spatial_gp(approx = "hsgp", coords = ~ x + y, m = 10, c = 1.5)
  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "hsgp")
  expect_equal(spec$m, 10)
})
