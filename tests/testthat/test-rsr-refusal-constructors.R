# spatial_rsr()'s refusal names the constructors to build a restrictable field
# with; every one it names is an exported function.

test_that("spatial_rsr() refusal names only exported constructors", {
  n <- 4L
  sp <- spatial_spde_custom(Matrix::Diagonal(n), Matrix::Diagonal(n),
                            Matrix::Diagonal(n), prior_range = c(1, 0.5),
                            prior_sigma = c(1, 0.01))
  msg <- tryCatch(spatial_rsr(sp, ~ x), error = conditionMessage)
  named <- unique(regmatches(msg, gregexpr("[a-z_]+(?=\\(\\))", msg,
                                           perl = TRUE))[[1]])
  expect_true(length(named) >= 3L)
  expect_true(all(named %in% getNamespaceExports("tulpa")))
})
