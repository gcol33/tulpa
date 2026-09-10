# Outer-grid progress knobs are resolved by EXACT key (gcol33/tulpa#719).
#
# `progress` is a strict prefix of `progress.every`, `progress.throttle` and
# `progress.file`, and `$` on a list partial-matches. Read with `$`, a control
# list carrying exactly ONE of the three returned that knob's value for
# `progress`, and `isTRUE()` of a file path or a cadence is FALSE -- so asking
# for the heartbeat file switched the console bar off. Carrying TWO made the
# prefix ambiguous, `$` returned NULL, and the default came back: the resolved
# setting was not monotone in the number of knobs set, which is what kept it
# out of sight.
#
# The two channels are independent by design: `progress` gates the console bar,
# `progress.file` is emitted whenever non-empty. Only an explicit `progress`
# turns the bar off.

test_that("a progress.* knob alone leaves the console bar on", {
  for (ctl in list(list(progress.file = "hb.log"),
                   list(progress.every = 5L),
                   list(progress.throttle = 1),
                   list(progress.file = "a", progress.every = 5L))) {
    expect_true(.nl_progress_args(ctl)$progress,
                info = paste(deparse(ctl), collapse = ""))
  }
})

test_that("each progress.* knob still reaches its own slot", {
  r <- .nl_progress_args(list(progress.file = "hb.log", progress.every = 5L,
                              progress.throttle = 1))
  expect_identical(r$progress_file, "hb.log")
  expect_identical(r$progress_every, 5L)
  expect_identical(r$progress_throttle, 1)
})

test_that("an explicit progress = FALSE wins over any progress.* knob", {
  expect_false(.nl_progress_args(list(progress = FALSE))$progress)
  r <- .nl_progress_args(list(progress = FALSE, progress.file = "a"))
  expect_false(r$progress)
  expect_identical(r$progress_file, "a")
})

test_that("a progress.* knob suppresses the scoped-option fallback", {
  # `has_ctrl` decides whether the fit-scoped `tulpa.nl_progress` option is
  # taken instead. A control list stating any knob is a decision and must not
  # be overridden by the option.
  op <- options(tulpa.nl_progress = list(progress = FALSE, progress_every = 99L,
                                         progress_throttle = 9, progress_file = "opt"))
  on.exit(options(op), add = TRUE)
  expect_identical(.nl_progress_args(list())$progress_every, 99L)
  r <- .nl_progress_args(list(progress.file = "ctl"))
  expect_identical(r$progress_file, "ctl")
  expect_true(r$progress)
})
