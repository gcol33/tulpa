# Serial against threaded, on the three OpenMP shapes the engine's hot loops
# take: a per-row reduction into a shared accumulator, a likelihood reduction,
# and an independent per-element write. A reduction that is not associative in
# its accumulation order, or a race on a shared write, shows up here as a
# thread-count-dependent answer.

test_that("a parallel dot-product reduction is independent of the thread count", {
  set.seed(5)
  X <- matrix(rnorm(200 * 8), 200, 8)
  b <- rnorm(8)

  ref <- cpp_test_parallel_dot_products(X, b, 1L)
  expect_equal(ref$results, as.numeric(X %*% b), tolerance = 1e-12)
  expect_equal(ref$total_sum, sum(as.numeric(X %*% b)), tolerance = 1e-10)

  for (nt in c(2L, 4L, 8L)) {
    got <- cpp_test_parallel_dot_products(X, b, nt)
    # Per-row results are written independently: identical, not merely close.
    expect_identical(got$results, ref$results, info = paste("threads", nt))
    # The reduction accumulates in a thread-dependent order, so it agrees to
    # floating-point summation error rather than bit for bit.
    expect_equal(got$total_sum, ref$total_sum, tolerance = 1e-10,
                 info = paste("threads", nt))
  }
})

test_that("a parallel likelihood reduction is independent of the thread count", {
  set.seed(6)
  n <- 500L
  mu <- runif(n, 0.2, 6)
  y <- rpois(n, mu)

  ref <- cpp_test_parallel_likelihood(y, mu, 1L)
  expect_equal(ref$log_lik, sum(dpois(y, mu, log = TRUE)), tolerance = 1e-10)
  for (nt in c(2L, 4L, 8L)) {
    got <- cpp_test_parallel_likelihood(y, mu, nt)
    expect_equal(got$log_lik, ref$log_lik, tolerance = 1e-10,
                 info = paste("threads", nt))
    expect_equal(got$n_threads_requested, nt)
  }
})

test_that("independent per-element writes are identical at any thread count", {
  n <- 1000L
  ref <- cpp_test_parallel_independent(n, 1L)
  i <- seq_len(n) - 1
  expect_equal(ref, sin(i * 0.1) * cos(i * 0.2) + i, tolerance = 1e-12)
  for (nt in c(2L, 4L, 8L)) {
    expect_identical(cpp_test_parallel_independent(n, nt), ref,
                     info = paste("threads", nt))
  }
})
