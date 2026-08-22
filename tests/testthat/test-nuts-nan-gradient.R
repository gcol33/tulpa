# A non-finite gradient is a divergence, not an acceptance (gcol33/tulpa#429).
#
# The default leapfrog op sequence ends in a Kick, so if the gradient at the
# endpoint carries a NaN the momentum does and the position does not: q was last
# written by the preceding Drift, and the log-posterior is whatever the density
# says. The leaf's state scan used to read the position vector and the
# log-posterior only, so the leaf passed as valid; the Hamiltonian it then
# computed was NaN, `NaN > delta_max` is false so the leaf was not divergent
# either, and its multinomial weight H0 - H_new was NaN.
#
# That NaN weight does not stay local. log_sum_exp of it is NaN, every later
# subtree's acceptance probability is NaN and gets zeroed by the finiteness
# guard, so no subtree can be selected for the rest of the doubling and
# sum_log_weight never recovers. The iteration keeps its initial point, is
# counted as an acceptance, and is not counted as a divergence -- a stuck draw
# that the output says nothing about.
#
# The divergence count is the diagnostic a user reads to know a fit is
# untrustworthy, so the failure being invisible is the whole of the defect.

test_that("the state scan reads the momentum, not only the position", {
  # q finite, log_prob finite, p carrying a NaN: the exact state a trailing Kick
  # on a non-finite gradient produces.
  r <- cpp_test_divergence_predicates(log_prob = 0, q = c(1, 2),
                                      p = c(1, NaN), H0 = 5, H_new = 5.1)
  expect_true(r$state_nonfinite)

  # An infinite momentum the same way, and a clean state left alone.
  expect_true(cpp_test_divergence_predicates(0, c(1, 2), c(1, Inf), 5, 5.1)$state_nonfinite)
  expect_false(cpp_test_divergence_predicates(0, c(1, 2), c(1, 2), 5, 5.1)$state_nonfinite)

  # The two arms it already had.
  expect_true(cpp_test_divergence_predicates(NaN, c(1, 2), c(1, 2), 5, 5.1)$state_nonfinite)
  expect_true(cpp_test_divergence_predicates(0, c(1, NaN), c(1, 2), 5, 5.1)$state_nonfinite)
})

test_that("a non-finite Hamiltonian is divergent", {
  # The negated comparison is what routes NaN to the divergent branch;
  # `H_new - H0 > delta_max` would not, because every comparison with NaN is
  # false.
  expect_true(cpp_test_divergence_predicates(0, 1, 1, H0 = 5, H_new = NaN)$hamiltonian_divergent)
  expect_true(cpp_test_divergence_predicates(0, 1, 1, H0 = 5, H_new = Inf)$hamiltonian_divergent)
  expect_false(isTRUE(NaN - 5 > 1000))   # what the old form evaluated

  # The ordinary arms are unchanged.
  expect_false(cpp_test_divergence_predicates(0, 1, 1, 5, 5.1)$hamiltonian_divergent)
  expect_true(cpp_test_divergence_predicates(0, 1, 1, 5, 5 + 1001)$hamiltonian_divergent)
  # An energy DROP is never a divergence, however large.
  expect_false(cpp_test_divergence_predicates(0, 1, 1, 5, 5 - 1e6)$hamiltonian_divergent)
})

test_that("a chain whose gradient is NaN reports every iteration divergent", {
  # LikelihoodSpec::gradient_fn is a model package's hand-coded full gradient
  # and writes the fused log-posterior itself, so it is the supported way to put
  # the sampler in the state above: a finite log-posterior with a non-finite
  # gradient. The target is a standard normal, so the two arms differ in one
  # planted entry and nothing else.
  nan   <- cpp_test_nan_gradient_nuts(plant_nan = TRUE)
  clean <- cpp_test_nan_gradient_nuts(plant_nan = FALSE)

  expect_identical(nan$n_divergent, nan$n_samples)
  # The chain is stuck, which is the honest outcome -- what must not happen is
  # it being stuck AND reported clean.
  expect_identical(nan$n_moved, 0L)
  expect_true(all(is.finite(nan$draws)))

  # The control: the same model with an intact gradient samples and diverges on
  # nothing, so the count is not saturated by the harness.
  expect_identical(clean$n_divergent, 0L)
  expect_gt(clean$n_moved, 0L)
})
