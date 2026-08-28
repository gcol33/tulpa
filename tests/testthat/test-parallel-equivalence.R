# Serial against threaded, on the OpenMP shapes the engine's hot loops take:
# a summed reduction, and an independent per-element write.
#
# The summed reduction is probed through BOTH constructs, because the engine
# ships only one of them (gcol33/tulpa#610). `tulpa_parallel_sum()` cuts the
# range into `team` contiguous chunks by index arithmetic and adds the chunk
# totals in chunk order, so its answer is a function of (team, n) alone; a raw
# `reduction(+:)` clause leaves both the partition and the combination order to
# the runtime. Every hot loop was moved onto the first, and the fixtures here
# were left on the second, so the two failures this file produced on Windows
# arm64 were on a construct that ships nowhere -- and nothing in the report
# could say so, because the two were never computed side by side.
#
# What each column separates, when a platform disagrees:
#
#   results / terms   independent per-element writes. A difference here is the
#                     per-observation arithmetic, not the summation -- for the
#                     likelihood probe that means the libm call in the body.
#   sum_shipped       the engine's construct. A difference here is what a fit
#                     would actually see.
#   sum_omp_red       the runtime's. A difference the shipped sum does not
#                     show is the OpenMP reduction, and reaches no fit.
#
# The reduction tolerance is the summation bound the values themselves imply,
# not a fixed constant: the difference between two orderings of the same `n`
# doubles is bounded by about `(n - 1) * eps * sum(|x|)`, so a platform outside
# that is doing something other than reordering. On these fixtures the bound is
# 2.6e-11 and 1.0e-10 absolute, both TIGHTER than the 1e-10 RELATIVE tolerance
# that failed, so this is a stricter reading of the same probes and not a
# loosened one.
#
# `sum_shipped` carries the verdict and `sum_omp_red` is REPORTED, because on
# Windows arm64 the reduction clause is measurably broken and nothing the
# package computes goes through it. The r-universe R-release build read
# `log_lik = -669.2` against a true -898.6, and the missing 229.417 is, to the
# four figures the report printed, exactly the last chunk of a 4-thread static
# split (chunk sums -221.97 / -219.51 / -227.70 / -229.44; dropping the fourth
# gives -669.18). One thread's private copy was not added in. The iterations
# themselves ran -- the same run passed the per-element `results` identity at
# every thread count -- so it is the combination the runtime performs at the
# end of the region, which is the one step `tulpa_parallel_sum()` does not ask
# the runtime for: it writes its slots from the loop body and adds them in a
# serial loop afterwards. It is also intermittent, failing both probes in one
# build and one in the next, which is what a race looks like.
#
# So a red suite there would be a third-party runtime defect on a construct no
# fit can reach, which is the noise that made this hard to read in the first
# place. The probe stays, and warns with the numbers.

# Headroom over `(n - 1) * eps * sum(|x|)`, which is a bound rather than a
# fitted constant.
reorder_bound <- function(x) 4 * length(x) * .Machine$double.eps * sum(abs(x))

# Requested / resolved / actual team size, and the observed difference, in the
# message -- so a run on a platform with no local machine reports what it saw
# instead of only that it disagreed.
thread_info <- function(got, nt, diff, bound) {
  sprintf(paste0("requested %d, resolved team %d, actual team %d, openmp %s; ",
                 "diff %.6e against bound %.6e"),
          nt, got$n_threads_team, got$n_threads_used, got$openmp, diff, bound)
}

# The retired construct's reading. A warning rather than a failure: it reaches
# no fit, and on Windows arm64 it is the runtime that is wrong.
report_omp_reduction <- function(got, ref, nt, bound) {
  d <- got$sum_omp_red - ref$sum_omp_red
  if (abs(d) >= bound) {
    warning("OpenMP reduction(+:) past the summation bound on this platform; ",
            "the shipped tulpa_parallel_sum is unaffected. ",
            thread_info(got, nt, d, bound), call. = FALSE)
  }
  invisible(d)
}

test_that("a parallel dot-product reduction is independent of the thread count", {
  set.seed(5)
  X <- matrix(rnorm(200 * 8), 200, 8)
  b <- rnorm(8)

  ref <- cpp_test_parallel_dot_products(X, b, 1L)
  expect_equal(ref$results, as.numeric(X %*% b), tolerance = 1e-12)
  expect_equal(ref$total_sum, sum(as.numeric(X %*% b)), tolerance = 1e-10)
  # One thread takes the plain loop on both constructs, so they are the same
  # sum in the same order.
  expect_identical(ref$sum_shipped, ref$sum_omp_red)
  expect_identical(ref$n_threads_used, 1L)

  bound <- reorder_bound(ref$results)
  for (nt in c(2L, 4L, 8L)) {
    got <- cpp_test_parallel_dot_products(X, b, nt)
    expect_identical(got$n_threads_requested, nt)
    # Per-row results are written independently: identical, not merely close.
    expect_identical(got$results, ref$results, info = paste("threads", nt))

    # The engine's construct. Reproducible at a team size, so a repeat call is
    # bit-identical, and across team sizes it moves only by the chunking.
    again <- cpp_test_parallel_dot_products(X, b, nt)
    expect_identical(again$sum_shipped, got$sum_shipped,
                     info = paste("threads", nt))
    d_ship <- got$sum_shipped - ref$sum_shipped
    expect_true(abs(d_ship) < bound, info = thread_info(got, nt, d_ship, bound))

    # The runtime's construct: reported, not asserted. See the file header.
    report_omp_reduction(got, ref, nt, bound)
  }
})

test_that("a parallel likelihood reduction is independent of the thread count", {
  set.seed(6)
  n <- 500L
  mu <- runif(n, 0.2, 6)
  y <- rpois(n, mu)

  ref <- cpp_test_parallel_likelihood(y, mu, 1L)
  expect_equal(ref$log_lik, sum(dpois(y, mu, log = TRUE)), tolerance = 1e-10)
  expect_equal(ref$terms, dpois(y, mu, log = TRUE), tolerance = 1e-12)
  expect_identical(ref$sum_shipped, ref$sum_omp_red)

  bound <- reorder_bound(ref$terms)
  for (nt in c(2L, 4L, 8L)) {
    got <- cpp_test_parallel_likelihood(y, mu, nt)
    expect_identical(got$n_threads_requested, nt)
    # std::lgamma writes the global `signgam`. If that global is held per
    # process on this platform's libm, the terms themselves move under
    # concurrency, and that is a different fault from a reordered sum.
    expect_identical(got$terms, ref$terms, info = paste("threads", nt))

    again <- cpp_test_parallel_likelihood(y, mu, nt)
    expect_identical(again$sum_shipped, got$sum_shipped,
                     info = paste("threads", nt))
    d_ship <- got$sum_shipped - ref$sum_shipped
    expect_true(abs(d_ship) < bound, info = thread_info(got, nt, d_ship, bound))

    report_omp_reduction(got, ref, nt, bound)
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
