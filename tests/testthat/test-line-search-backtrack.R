test_that("full step is accepted directly when it does not overshoot", {
  # phi(a) = a - 0.25 a^2: optimum at a* = 2, so a = 1 already improves.
  res <- tulpa:::cpp_line_search_probe(slope = 1.0, c = -0.25)
  expect_equal(res$step, 1.0)
  expect_equal(res$n_evals, 1L)   # one eval, accepted, no backtrack
})

test_that("interpolation lands near the line optimum in one backtrack", {
  # phi(a) = a - 2 a^2: optimum a* = -slope/(2c) = 0.25. The full step a = 1
  # gives phi(1) = -1 < 0 (overshoot), so one quadratic interpolation should
  # jump straight to a* (inside the [0.1, 0.5] safeguard of a = 1).
  res <- tulpa:::cpp_line_search_probe(slope = 1.0, c = -2.0)
  expect_equal(res$step, 0.25, tolerance = 1e-8)
  expect_equal(res$n_evals, 2L)   # a = 1 rejected, a = 0.25 accepted
  expect_gte(res$obj, -1e-8)      # accepted objective does not decrease
})

test_that("interpolation uses fewer sweeps than halving on a strong overshoot", {
  # phi(a) = a - 8 a^2: optimum a* = 0.0625. Pure halving would test
  # 1, 0.5, 0.25, 0.125 (4 evals) before accepting; the safeguarded
  # interpolation clamps to 0.1 x the step and accepts on the second eval.
  res <- tulpa:::cpp_line_search_probe(slope = 1.0, c = -8.0)
  expect_lte(res$n_evals, 3L)
  expect_gte(res$obj, -1e-8)
  expect_gt(res$step, 0.0)
})

test_that("non-ascent slope falls back to halving safely", {
  # slope = 0 disables interpolation (the guard requires slope > 0); the
  # search must still terminate via the halving fallback / MAX_HALVING cap
  # without producing a non-finite step.
  res <- tulpa:::cpp_line_search_probe(slope = 0.0, c = -1.0)
  expect_true(is.finite(res$step))
  expect_gt(res$n_evals, 0L)
})

test_that("convergence reads the Newton proposal, not the damped step", {
  # A proposal of 1e-6 is nowhere near a 1e-12 tolerance, whatever scale the line
  # search settled on. Reading the TAKEN step would call the third case converged
  # -- 1e-6 x 2^-20 is 9.5e-13 -- turning a search that damped hard into a mode.
  delta <- 1e-6
  grad  <- 1e-3          # decrement 1e-9, under the near-mode gate
  for (ss in c(1, 0.5, 2^-20)) {
    expect_false(
      tulpa:::cpp_newton_converged_probe(delta, grad, ss, tol = 1e-12),
      info = sprintf("step_scale = %g", ss))
  }

  # A proposal genuinely under the tolerance converges at any accepted scale.
  for (ss in c(1, 0.5, 2^-20)) {
    expect_true(
      tulpa:::cpp_newton_converged_probe(1e-14, grad, ss, tol = 1e-12),
      info = sprintf("step_scale = %g", ss))
  }

  # A search that accepted nothing is a stall, never a mode, however small the
  # proposal it could not move along.
  expect_false(tulpa:::cpp_newton_converged_probe(1e-14, grad, 0, tol = 1e-12))
})
