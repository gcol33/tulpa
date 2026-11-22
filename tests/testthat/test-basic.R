
test_that("tulpa package loads without error", {
  expect_true("tulpa" %in% loadedNamespaces() || requireNamespace("tulpa", quietly = TRUE))
})
