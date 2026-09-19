# The VI stopping rule (ConvergenceChecker, src/vi_optimizer.h).
#
# This, not `vi_max_iter`, is what ends a VI fit: the budget is a ceiling the
# run rarely reaches. The rule used to test a single iteration's gain against
# |ELBO|, but an ELBO carries an arbitrary additive constant, so that made the
# threshold a property of where the log-density's normalizing terms happened to
# sit -- at ELBO ~ -742 any gain under 7.4 nats per iteration counted as no
# improvement (gcol33/tulpa#821). It now compares the ELBO gain across the
# patience window against the span the run has covered, both of them
# differences.

test_that("the stopping rule is invariant to an ELBO offset", {
  # The property the old rule failed. An ELBO's additive constant is not part
  # of the model, so shifting a whole run by one must not move the stop.
  set.seed(11)
  base <- -3000 + 2800 * (1 - exp(-seq_len(400) / 60)) + rnorm(400, 0, 1.5)
  ref <- cpp_vi_convergence_replay(base)
  for (shift in c(-1e5, -1000, 0, 1000, 1e5)) {
    got <- cpp_vi_convergence_replay(base + shift)
    expect_equal(got$iteration, ref$iteration,
                 info = sprintf("shift = %g", shift))
    expect_equal(got$reason, ref$reason, info = sprintf("shift = %g", shift))
  }
})

test_that("the stopping rule is invariant to an ELBO rescaling", {
  # Gain and threshold are both differences of the same sequence, so a common
  # positive scale cancels out of the comparison.
  set.seed(12)
  base <- -500 + 450 * (1 - exp(-seq_len(400) / 50)) + rnorm(400, 0, 0.4)
  ref <- cpp_vi_convergence_replay(base)
  for (s in c(0.01, 1, 100)) {
    expect_equal(cpp_vi_convergence_replay(s * base)$iteration, ref$iteration,
                 info = sprintf("scale = %g", s))
  }
})

test_that("a run still climbing is not stopped", {
  # A steady climb of 1 nat per iteration on a run whose span is ~400: the old
  # rule stopped this at iteration 50 for any |ELBO| above 100, because 1 nat
  # is under 1% of it.
  climb <- -800 + seq_len(400)
  res <- cpp_vi_convergence_replay(climb, patience = 50)
  expect_false(res$stopped)

  # ... at every offset, including one where the old test's threshold would
  # have been far above the per-iteration gain.
  for (shift in c(0, 1e4, -1e4)) {
    expect_false(cpp_vi_convergence_replay(climb + shift)$stopped,
                 info = sprintf("shift = %g", shift))
  }
})

test_that("a flat run stops once the window is full", {
  flat <- rep(-123.456, 200)
  res <- cpp_vi_convergence_replay(flat, patience = 50)
  expect_true(res$stopped)
  expect_equal(res$reason, "patience")
  expect_equal(res$iteration, 50L)
})

test_that("a falling run stops", {
  res <- cpp_vi_convergence_replay(-100 - seq_len(300), patience = 40)
  expect_true(res$stopped)
  expect_equal(res$reason, "patience")
})

test_that("tol_rel_elbo and patience move the stop in the expected direction", {
  set.seed(13)
  s <- -2000 + 1900 * (1 - exp(-seq_len(2000) / 120)) + rnorm(2000, 0, 1)
  loose <- cpp_vi_convergence_replay(s, tol_rel_elbo = 0.05, patience = 50)
  tight <- cpp_vi_convergence_replay(s, tol_rel_elbo = 0,    patience = 50)
  expect_lt(loose$iteration, tight$iteration)

  short <- cpp_vi_convergence_replay(s, tol_rel_elbo = 0, patience = 20)
  long  <- cpp_vi_convergence_replay(s, tol_rel_elbo = 0, patience = 200)
  expect_lt(short$iteration, long$iteration)
})

test_that("a small gradient norm stops the run whatever the ELBO does", {
  res <- cpp_vi_convergence_replay(-100 + seq_len(300), grad_norm = 1e-9)
  expect_true(res$stopped)
  expect_equal(res$reason, "gradient_norm")
  expect_equal(res$iteration, 1L)
})

test_that("a VI fit reports the iteration it stopped at and the rule", {
  skip_on_cran()
  set.seed(4)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.3 + 0.5 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "vi",
               control = list(vi_max_iter = 400L, seed = 1L))
  expect_true(is.numeric(fit$vi_iterations))
  expect_true(fit$converged_reason %in%
                c("patience", "gradient_norm", "max_iter"))
  # The budget is not what bound, which is the fact `converged = TRUE` alone
  # could not express.
  expect_lte(fit$vi_iterations, 400L)
})

test_that("the stopping-rule knobs reach the optimizer", {
  skip_on_cran()
  set.seed(4)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.3 + 0.5 * d$x))
  run <- function(...) tulpa(y ~ x, data = d, family = "poisson", mode = "vi",
                             control = utils::modifyList(
                               list(vi_max_iter = 3000L, seed = 1L), list(...)))
  short <- run(vi_patience = 10L)
  long  <- run(vi_patience = 300L, vi_tol_rel_elbo = 0)
  expect_lt(short$vi_iterations, long$vi_iterations)
})

test_that("the control layer rejects a stopping rule it cannot run", {
  d <- data.frame(x = rnorm(30), y = rpois(30, 2))
  expect_error(tulpa(y ~ x, data = d, family = "poisson", mode = "vi",
                     control = list(vi_patience = 1L)),
               "vi_patience")
  expect_error(tulpa(y ~ x, data = d, family = "poisson", mode = "vi",
                     control = list(vi_tol_rel_elbo = -1)),
               "vi_tol_rel_elbo")
  expect_error(tulpa(y ~ x, data = d, family = "poisson", mode = "vi",
                     control = list(vi_tol_grad = -1)),
               "vi_tol_grad")
})
