test_that("tulpa_version reports DESCRIPTION's version rather than a literal", {
  # Read DESCRIPTION directly instead of comparing the accessor to its own
  # implementation: a version restated in the source satisfies a
  # self-comparison, and drifts from DESCRIPTION at the first release that
  # forgets it.
  installed <- read.dcf(system.file("DESCRIPTION", package = "tulpa"),
                        fields = "Version")[1, "Version"]
  expect_identical(tulpa:::tulpa_version(), unname(installed))
})

test_that("tulpa package loads without error", {
  expect_true("tulpa" %in% loadedNamespaces() || requireNamespace("tulpa", quietly = TRUE))
})
