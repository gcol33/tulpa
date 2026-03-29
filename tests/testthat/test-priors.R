test_that("tulpa_priors creates valid prior specification", {
  p <- tulpa_priors()
  expect_s3_class(p, "tulpa_priors")
})

test_that("prior_normal creates valid prior", {
  p <- prior_normal(mean = 0, sd = 2.5)
  expect_true(is.list(p))
  expect_equal(p$mean, 0)
  expect_equal(p$sd, 2.5)
})

test_that("prior_pc creates valid PC prior", {
  p <- prior_pc(U = 1.0, alpha = 0.01)
  expect_true(is.list(p))
  expect_equal(p$U, 1.0)
  expect_equal(p$alpha, 0.01)
})
