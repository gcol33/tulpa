# =============================================================================
# test-nuts-progress-scope.R - the active NUTS progress reporter is a
# process-global raw pointer, and no chain may outlive its registration.
#
# `g_active_grid_progress` names a reporter owned by a local unique_ptr in the
# frame that built it. A chain that leaves by an exception destroys that
# reporter; if the global still names it, the next chain in the process reads a
# non-null pointer, declines to build its own, and ticks freed memory once per
# iteration -- a fault below R with no traceback, landing wherever the freed
# block was reused rather than where the fault was made.
#
# Two readings, because either alone can pass while the object is dead:
#   (1) the pointer itself, after a chain ended by a planted gradient exception;
#   (2) the behaviour it governs -- a later chain asking for its own heartbeat
#       file gets one, which a chain that skipped building a reporter cannot.
# =============================================================================

# Run `code` with a heartbeat file wired up, and report whether it was written.
with_heartbeat <- function(code) {
  f <- tempfile(fileext = ".eta")
  old <- options(tulpa.nl_progress = list(progress = FALSE, progress_file = f,
                                          progress_every = 1L,
                                          progress_throttle = 0))
  on.exit({ options(old); unlink(f) }, add = TRUE)
  force(code)
  file.exists(f)
}

test_that("a chain ended by an exception leaves no reporter registered", {
  expect_false(cpp_test_nuts_progress_active())

  wrote <- with_heartbeat(
    expect_error(cpp_test_nuts_gradient_throws(throw_after = 30L, K = 3L,
                                               n_iter = 40L, n_warmup = 20L),
                 "planted gradient failure"))

  # The chain got far enough to own a reporter, so the pointer really was set
  # and cleared rather than never set at all.
  expect_true(wrote)
  expect_false(cpp_test_nuts_progress_active())
})

test_that("a chain after a failed one still builds its own reporter", {
  with_heartbeat(
    expect_error(cpp_test_nuts_gradient_throws(throw_after = 30L, K = 3L,
                                               n_iter = 40L, n_warmup = 20L),
                 "planted gradient failure"))

  wrote <- with_heartbeat(
    cpp_test_nan_gradient_nuts(FALSE, K = 2L, n_iter = 6L, n_warmup = 3L))
  expect_true(wrote)
  expect_false(cpp_test_nuts_progress_active())
})
