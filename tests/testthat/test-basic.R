test_that("tulpa_version returns version string", {
  expect_equal(tulpa_version(), "0.1.0")
})

test_that("tulpa package loads without error", {
  expect_true("tulpa" %in% loadedNamespaces() || requireNamespace("tulpa", quietly = TRUE))
})
