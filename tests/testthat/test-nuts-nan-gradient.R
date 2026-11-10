skip_on_cran_budget()

test_that("the state scan checks position, momentum and log density", {
  load_cpp_fixture("nan_gradient_nuts")
  for (bad in c(NaN, Inf)) {
    expect_true(cpp_test_divergence_predicates(0, c(1, 2), c(1, bad), 5, 5.1)$state_nonfinite)
  }
  expect_false(cpp_test_divergence_predicates(0, c(1, 2), c(1, 2), 5, 5.1)$state_nonfinite)
  expect_true(cpp_test_divergence_predicates(NaN, c(1, 2), c(1, 2), 5, 5.1)$state_nonfinite)
  expect_true(cpp_test_divergence_predicates(0, c(1, NaN), c(1, 2), 5, 5.1)$state_nonfinite)
})

test_that("a non-finite Hamiltonian is divergent", {
  load_cpp_fixture("nan_gradient_nuts")
  expect_true(cpp_test_divergence_predicates(0, 1, 1, 5, NaN)$hamiltonian_divergent)
  expect_true(cpp_test_divergence_predicates(0, 1, 1, 5, Inf)$hamiltonian_divergent)
  expect_false(isTRUE(NaN - 5 > 1000))
  expect_false(cpp_test_divergence_predicates(0, 1, 1, 5, 5.1)$hamiltonian_divergent)
  expect_true(cpp_test_divergence_predicates(0, 1, 1, 5, 5 + 1001)$hamiltonian_divergent)
  expect_false(cpp_test_divergence_predicates(0, 1, 1, 5, 5 - 1e6)$hamiltonian_divergent)
})

test_that("initial gradient validation precedes the step search", {
  load_cpp_fixture("nan_gradient_nuts")
  expect_silent(expect_error(cpp_test_nan_gradient_nuts(TRUE),
    "non-finite gradient at the initial position, coordinate 1"))
})

test_that("a NaN gradient during a trajectory marks every iteration divergent", {
  load_cpp_fixture("nan_gradient_nuts")
  nan <- cpp_test_nan_gradient_nuts(TRUE, epsilon_init = 0.1, nan_at_start = FALSE)
  clean <- cpp_test_nan_gradient_nuts(FALSE)
  expect_identical(nan$n_divergent, nan$n_samples)
  expect_identical(nan$n_moved, 0L)
  expect_true(all(is.finite(nan$draws)))
  expect_identical(clean$n_divergent, 0L)
  expect_gt(clean$n_moved, 0L)
})

test_that("direct chain entry validates a start with either gate state", {
  load_cpp_fixture("nan_gradient_nuts")
  for (bypass in c(FALSE, TRUE)) {
    for (epsilon in c(0, 0.1)) {
      expect_silent(expect_error(
        cpp_test_nan_gradient_nuts(TRUE, bypass_gate = bypass, epsilon_init = epsilon),
        "non-finite gradient at the initial position, coordinate 1 \\(chain 1\\)"))
    }
  }
  expect_silent(ok <- cpp_test_nan_gradient_nuts(FALSE, bypass_gate = FALSE))
  expect_identical(ok, cpp_test_nan_gradient_nuts(FALSE, bypass_gate = TRUE))
  expect_gt(ok$n_moved, 0L)
})
