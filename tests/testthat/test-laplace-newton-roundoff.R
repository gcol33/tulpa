test_that("near-mode Newton steps tolerate accumulated objective rounding", {
  for (offset in c(-2e6, 2e6)) {
    out <- tulpa:::cpp_line_search_probe(1e-11, -0.5e-11,
                                         obj_offset = offset, trial_error = -3e-7)
    expect_identical(out$step, 1)
    expect_identical(out$n_evals, 1L)
    expect_equal(out$obj, offset - 3e-7, tolerance = 1e-14)
  }
})

test_that("the rounding band requires a near-mode ascent proposal", {
  for (slope in c(-1e-11, 1e-5)) {
    out <- tulpa:::cpp_line_search_probe(slope, -slope,
                                         obj_offset = -2e6, trial_error = -3e-7)
    expect_lt(out$step, 1)
    expect_gt(out$n_evals, 1L)
  }
  out <- tulpa:::cpp_line_search_probe(1e-11, -0.5,
                                       obj_offset = -2e6)
  expect_lt(out$step, 1)
  expect_gt(out$n_evals, 1L)
})

test_that("a non-finite objective remains a barrier at the rounding limit", {
  for (error in c(-Inf, Inf, NaN)) {
    out <- tulpa:::cpp_line_search_probe(1e-11, -0.5e-11,
                                         obj_offset = -2e6, trial_error = error)
    expect_identical(out$step, 0)
    expect_identical(out$obj, -2e6)
  }
})

test_that("the decrement stop is invariant to latent coordinate units", {
  delta <- c(1e-4, -2e-4)
  grad <- c(2e-22, -1e-22)
  for (units in list(c(1, 1), c(1e6, 1e-6), c(1e-6, 1e6))) {
    expect_true(tulpa:::cpp_newton_converged_probe(
      delta * units, grad / units, 1, tol = 1e-12))
    expect_false(tulpa:::cpp_newton_converged_probe(
      delta * units, grad / units, 0, tol = 1e-12))
  }
  expect_false(tulpa:::cpp_newton_converged_probe(
    c(1e-4, -2e-4), c(-2e-22, 1e-22), 1, tol = 1e-12))
})
