test_that("temporal_rw1 creates valid specification", {
  spec <- temporal_rw1(time_var = "year")
  expect_s3_class(spec, "tulpa_temporal")
  expect_equal(spec$type, "rw1")
})

test_that("temporal_ar1 creates valid specification", {
  spec <- temporal_ar1(time_var = "year")
  expect_s3_class(spec, "tulpa_temporal")
  expect_equal(spec$type, "ar1")
})

test_that("temporal_rw2 creates valid specification", {
  spec <- temporal_rw2(time_var = "year", cyclic = TRUE)
  expect_s3_class(spec, "tulpa_temporal")
  expect_equal(spec$type, "rw2")
  expect_true(spec$cyclic)
})
