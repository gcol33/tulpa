# Temporal prior kernels at the edges of their rank and their names.

test_that("rw2_quadratic_form returns 0 for T < 3", {
  load_cpp_fixture("test_ratio_temporal")
  expect_equal(cpp_test_rw2_quadratic_form(c(1, 2), FALSE), 0)
  expect_equal(cpp_test_rw2_quadratic_form(c(1), FALSE), 0)
})

test_that("the intrinsic arms' augmentation carries both the quadratic and the rank", {
  load_cpp_fixture("test_ratio_temporal")
  # Q -> Q + 11'/n gives the constant direction the field's own precision tau,
  # which adds (sum phi)^2 / n to the quadratic form AND one to the rank. A
  # path that took one and not the other would make the tau-marginal wrong.
  sigma2 <- 0.7
  phi <- c(1, 2, 3)  # sum = 6

  quad <- (2 - 1)^2 + (3 - 2)^2                      # 2
  quad_aug <- quad + 6^2 / 3                         # 14
  rank <- 2                                          # rw1_rank(3, FALSE)

  plain <- cpp_test_ms_rw_log_lik(phi, sigma2, 1L, FALSE, FALSE)
  expect_equal(plain,
               -0.5 * quad / sigma2 - 0.5 * rank * log(2 * pi * sigma2),
               tolerance = 1e-10)

  augmented <- cpp_test_ms_rw_log_lik(phi, sigma2, 1L, FALSE, TRUE)
  expect_equal(augmented,
               -0.5 * quad_aug / sigma2 - 0.5 * (rank + 1) * log(2 * pi * sigma2),
               tolerance = 1e-10)

  # On the sum-to-zero subspace the quadratic is untouched, so the whole
  # difference is the one extra rank the full-rank normalizer carries.
  centred <- c(-1, 0, 1)
  expect_equal(
    cpp_test_ms_rw_log_lik(centred, sigma2, 1L, FALSE, TRUE) -
      cpp_test_ms_rw_log_lik(centred, sigma2, 1L, FALSE, FALSE),
    -0.5 * log(2 * pi * sigma2),
    tolerance = 1e-10)
})

test_that("a non-cyclic RW2 stays deficient by one after the augmentation", {
  load_cpp_fixture("test_ratio_temporal")
  # The augmentation fills the constant direction and nothing else. A non-cyclic
  # RW2 also annihilates a linear ramp, so its rank goes T-2 -> T-1, not to T;
  # passing the field length there would bias the variance component.
  sigma2 <- 1.3
  phi <- c(0.2, -0.5, 1.1, 0.4)
  n <- length(phi)

  d2 <- diff(diff(phi))
  quad_aug <- sum(d2^2) + sum(phi)^2 / n

  expect_equal(
    cpp_test_ms_rw_log_lik(phi, sigma2, 2L, FALSE, TRUE),
    -0.5 * quad_aug / sigma2 - 0.5 * (n - 1) * log(2 * pi * sigma2),
    tolerance = 1e-10)
})

test_that("an unknown temporal structure stops rather than reading as a default", {
  load_cpp_fixture("test_ratio_temporal")
  expect_error(cpp_test_temporal_log_prior(rnorm(5), "rw3", 1, 0, FALSE),
               "temporal structure")
})

