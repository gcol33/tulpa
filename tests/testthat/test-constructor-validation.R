# Utility constructors refuse invalid input with an error naming the argument
# (gcol33/tulpa#896), rather than accepting it silently, truncating it, or
# failing later with "missing value where TRUE/FALSE needed".

test_that("the SBC predictive constructors validate their inputs", {
  expect_error(sbc_normal(0, -1), "`sd`")
  expect_error(sbc_normal(NA, 1), "`mean`")
  expect_error(sbc_normal("a", 1), "`mean`")
  expect_identical(sbc_normal(0.3, 0)$var, 0)          # a point mass is fine

  expect_error(sbc_discrete(1:3, c(-1, 1, 1)), "`probs`")
  expect_error(sbc_discrete(1:3, c(0.5, 0.5)), "one entry per element")
  expect_error(sbc_discrete(1:3, c(0, 0, 0)), "positive total mass")
  expect_error(sbc_discrete(c(1, NA, 3), c(1, 1, 1)), "`support`")
  expect_equal(sbc_discrete(3:1, c(1, 1, 2))$probs, c(0.5, 0.25, 0.25))

  expect_error(sbc_rank(150, 100), "0:n_ref")
  expect_error(sbc_rank(-3, 100), "`rank`")
  expect_error(sbc_rank(2.5, 100), "`rank`")
  expect_error(sbc_rank(3, 0), "`n_ref`")
  expect_identical(sbc_rank(100, 100)$rank, 100L)

  expect_error(sbc_mixture(c(0, 1), c(1, 1), w = c(-1, 2)), "`w`")
  expect_error(sbc_mixture(c(0, 1), c(1, 1), w = 1), "one entry per element")
  expect_error(sbc_mixture(c(0, 1), c(1, -1)), "`var`")
  expect_error(sbc_mixture(c(0, 1), 1), "`var` must have one entry")
})

test_that("the prior constructors name a missing or non-numeric argument", {
  expect_error(prior_normal(NA, 1), "`mean` must be a single finite number")
  expect_error(prior_normal("a", 1), "`mean`")
  expect_error(prior_normal(0, NA), "`sd` must be a single positive number")
  expect_error(prior_normal(0, -1), "`sd`")
  expect_error(prior_normal(c(0, 1), 1), "`mean`")
  expect_error(prior_half_normal(NA), "`sd`")
  expect_error(prior_half_cauchy(NA), "`scale`")
  expect_error(prior_gamma(NA, 1), "`shape`")
  expect_error(prior_exponential(NA), "`rate`")
  expect_error(prior_beta(1, NA), "`beta`")
  expect_error(prior_pc(U = NA), "`U`")
  expect_error(prior_pc(alpha = NA), "`alpha`")
  expect_error(prior_pc(alpha = 1), "`alpha`")
  expect_s3_class(prior_normal(0, 2.5), "tulpa_prior_normal")
  expect_equal(prior_pc(1, 0.01)$rate, -log(0.01))
})

test_that("grid constructors refuse a fractional size instead of truncating it", {
  expect_error(ccd_grid(2.7), "`k` must be a single whole number")
  expect_error(ccd_grid(0), "`k`")
  expect_error(ccd_grid(NA), "`k`")
  expect_error(ccd_grid(2, f_0 = -1), "`f_0`")
  expect_equal(ncol(ccd_grid(3)$z), 3L)

  expect_error(tulpa_grid_axis("field_sd", n = 2.5),
               "`n` must be a single whole number")
  expect_error(tulpa_grid_axis("field_sd", n = 0), "`n`")
  expect_length(tulpa_grid_axis("field_sd", n = 4), 4L)
})

test_that("tulpa_check_control refuses a knob given twice", {
  expect_error(
    tulpa_check_control(list(max_iter = 1, max_iter = 2),
                        c("max_iter", "tol"), "my_fit"),
    "given more than once in my_fit\\(\\): 'max_iter'")
  expect_null(tulpa_check_control(list(max_iter = 1, tol = 1e-6),
                                  c("max_iter", "tol"), "my_fit"))
})
