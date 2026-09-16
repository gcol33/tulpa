# tulpa_rpg() is an exported door onto the Polson-Scott-Windle Polya-Gamma
# sampler cpp_rpg() drives (gcol33/tulpa#826). Every PG-Gibbs fitter in this
# package reaches the same kernel through the compiled Rcpp::export directly;
# this door exists for consumer packages fitting their own PG-Gibbs models.

test_that("tulpa_rpg() is exported and matches cpp_rpg() bit for bit", {
  expect_true("tulpa_rpg" %in% getNamespaceExports("tulpa"))
  b <- c(1L, 1L, 3L, 1L, 5L)
  z <- c(-2, 0, 0.7, 1.5, -0.3)
  set.seed(11); ref <- cpp_rpg(b, z)
  set.seed(11); got <- tulpa_rpg(b, z)
  expect_identical(got, ref)
})

test_that("tulpa_rpg() coerces b and z the way cpp_rpg() requires", {
  set.seed(5); ref <- cpp_rpg(as.integer(c(1, 1, 1)), as.numeric(c(0L, 1L, -1L)))
  set.seed(5); got <- tulpa_rpg(c(1, 1, 1), c(0L, 1L, -1L))
  expect_identical(got, ref)
  expect_length(got, 3L)
  expect_true(all(got > 0))
})
