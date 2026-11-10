test_that("the sampler lays out the HSGP space-time hyperparameters and coefficients", {
  load_cpp_fixture("st_hsgp_prior")
  expect_equal(cpp_test_st_hsgp_layout(), 25L)
  expect_equal(cpp_test_st_hsgp_layout(temporal = "rw1"), 25L)
  expect_equal(cpp_test_st_hsgp_layout(temporal = "ar1"), 26L)
})

test_that("the sampler layout refuses a multiscale HSGP field", {
  load_cpp_fixture("st_hsgp_prior")
  expect_error(cpp_test_msgp_hsgp_layout(), "no multiscale HSGP density")
})
