# An HSGP space-time interaction (ModelData::st_is_hsgp) has no sampler
# density in the engine: no door sets the flag, and the interaction is fitted
# by nested Laplace (fit_st_nested(spatial_type = "hsgp")). A consumer-built
# ModelData carrying it is refused where the engine lays out the parameters,
# rather than sampled under the Kronecker branch, whose prior reads neither
# HSGP hyperparameter and treats basis weights as sites.

test_that("the sampler layout refuses an HSGP space-time interaction", {
  load_cpp_fixture("st_hsgp_prior")
  expect_error(cpp_test_st_hsgp_layout(), "no HSGP space-time interaction")
  expect_error(cpp_test_st_hsgp_layout(temporal = "rw1"),
               "fit_st_nested")
})
