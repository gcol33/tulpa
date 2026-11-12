test_that("GP coefficients retain native coordinates across term counts", {
  d <- data.frame(day = rep(seq_len(5L), each = 4L),
    x = sin(seq_len(20L)), z = cos(seq_len(20L)), v = sin(.3 * seq_len(20L)))
  X <- model.matrix(~ x + z + v, d)
  for (n_terms in seq_len(3L)) {
    temporal <- tulpa:::validate_tvc(temporal_tvc("day",
      terms = 1L + seq_len(n_terms), structure = "gp"), d, X)
    spec <- tulpa:::.tvc_sampler_spec(temporal, X)
    r <- tulpa:::cpp_test_field_coordinates(y = .2 + .5 * d$x, X = X,
      tvc_spec = spec, shift = .3)
    expect_length(r$blocks, 0L)
    expect_equal(r$x, r$q, tolerance = 1e-12)
    expect_equal(r$jacobian, 0)
    expect_equal(r$inverse, r$q, tolerance = 1e-12)
    expect_equal(r$gradient, r$numerical, tolerance = 1e-5)
    expect_equal(r$prior_gradient, r$numerical_prior, tolerance = 1e-5)
    expect_true(r$verified)
  }
})
