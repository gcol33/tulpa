# Out-of-pattern Hessian writes (gcol33/tulpa#249).
#
# A scatter and the pattern builder that precedes it must enumerate the same
# (row, col) set. Where they disagree the surplus writes are discarded, leaving a
# Hessian that is too small in one entry: finite, positive definite, correctly
# shaped, and wrong. No shape / finiteness / PD check sees it, so the detection
# channel is the drop counter in src/hessian_pattern_guard.h.

# A small SPD lower-triangle-nonzero matrix: dense enough that omitting one entry
# leaves a well-formed pattern, sparse enough that the omission is unambiguous.
guard_matrix <- function() {
  A <- matrix(0, 4, 4)
  diag(A) <- c(2.5, 3.0, 2.0, 4.0)
  A[2, 1] <- A[1, 2] <- 0.7
  A[3, 2] <- A[2, 3] <- -0.4
  A[4, 1] <- A[1, 4] <- 0.9
  A
}

test_that("a complete pattern stores every contribution and drops nothing", {
  A <- guard_matrix()
  res <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = -1L, raise = FALSE)

  expect_equal(res$dropped, 0)
  # 4 diagonal + 3 off-diagonal lower-triangle nonzeros.
  expect_equal(res$nnz, 7L)
  expect_equal(res$stored_sum,
               sum(diag(A)) + A[2, 1] + A[3, 2] + A[4, 1],
               tolerance = 1e-12)
})

test_that("an entry missing from the pattern is counted, not silently dropped", {
  A <- guard_matrix()
  full <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = -1L, raise = FALSE)

  # Omit each lower-triangle nonzero in turn: every one must be detected.
  for (e in seq_len(full$nnz) - 1L) {
    res <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = e, raise = FALSE)
    expect_equal(res$dropped, 1,
                 info = sprintf("omitted pattern entry %d", e))
    expect_equal(res$nnz, full$nnz - 1L)
  }
})

test_that("the discarded contribution is invisible in the stored matrix", {
  # The point of the counter: what remains passes every structural check while
  # being wrong by exactly the dropped value.
  A <- guard_matrix()
  full <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = -1L, raise = FALSE)
  # Entry 0 in (col, row) traversal order is A[1, 1].
  holed <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = 0L, raise = FALSE)

  expect_equal(full$stored_sum - holed$stored_sum, A[1, 1], tolerance = 1e-12)
  expect_true(is.finite(holed$stored_sum))
})

test_that("the guard raises on a fit that discarded a contribution", {
  A <- guard_matrix()

  expect_error(
    tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = 0L, raise = TRUE),
    "outside the registered sparsity pattern"
  )
  expect_silent(tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = -1L, raise = TRUE))
})

test_that("an unresolved cached slot counts only a nonzero contribution", {
  # Index caches resolve whole cross products up front and legitimately hold -1
  # for pairs no observation touches; those slots are written with a structural
  # zero, which changes nothing whether it lands or not.
  A <- guard_matrix()

  zero_write <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = -1L,
                                               raise = FALSE, slot_val = 0)
  expect_equal(zero_write$dropped, 0)

  live_write <- tulpa:::cpp_test_hessian_pattern_guard(A, omit_entry = -1L,
                                               raise = FALSE, slot_val = 1.5)
  expect_equal(live_write$dropped, 1)
})

test_that("real fits scatter entirely inside their registered pattern", {
  # The acceptance condition for #249: with the check armed, the kernels that
  # broke this invariant before (#241 areal components, #242 weighted entries)
  # must run clean. A drop anywhere below raises from the solve driver.
  skip_on_cran()
  set.seed(4249)

  n <- 40
  d <- data.frame(
    x = rnorm(n),
    g = factor(rep(seq_len(8), each = 5))
  )
  d$y <- rpois(n, exp(0.4 + 0.3 * d$x))

  expect_no_error(
    fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                 mode = "laplace")
  )
  expect_s3_class(fit, "tulpa_fit")
})
