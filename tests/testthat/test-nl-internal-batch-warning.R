# gcol33/tulpa#614. The soft-cap warning is advice addressed to whoever chose
# the grid. The outer Pareto-k diagnostic re-evaluates `log_marginal` at
# `control$k_samples` (default 200) importance draws by substituting them for
# the block's grid axis and re-dispatching through the ordinary fitter, so a
# caller who chose 7 nodes was warned about 200 cells and advised to "reduce
# per-block grid sizes", which does not reach the number in the message.
# `.nl_internal_batch()` is the one predicate that separates the two.
#
# gcol33/tulpa#820 (a293202f) later replaced the warning's trigger: a static
# cell-count threshold fired on the engine's own default per-block grids, so
# it now fires on MEASURED ELAPSED TIME instead (`.nl_multi_grid_warn()`,
# `R/nested_laplace.R`), with a rewritten message ("Multi-block outer grid (N
# cells) took ... to solve."). A caller-chosen grid the engine solves quickly
# no longer warns by itself -- by design -- so the "a grid the caller did
# choose still warns" side of #614's fixture now tests `.nl_multi_grid_warn()`
# directly at a synthetic elapsed time rather than a real fit, which is also
# what makes it deterministic rather than a wall-clock race.

grid_warnings <- function(expr) {
  w <- character(0)
  withCallingHandlers(
    force(expr),
    warning = function(cnd) {
      if (grepl("Multi-block outer grid (", conditionMessage(cnd), fixed = TRUE)) {
        w <<- c(w, conditionMessage(cnd))
      }
      invokeRestart("muffleWarning")
    })
  w
}

test_that(".nl_internal_batch() is off by default and restores (#614)", {
  expect_false(.nl_internal_batch())
  inner <- .nl_with_internal_batch({
    expect_true(.nl_internal_batch())
    # Nested: an internal batch that re-dispatches stays internal, and the
    # inner restore must not clear the outer flag.
    .nl_with_internal_batch(expect_true(.nl_internal_batch()))
    expect_true(.nl_internal_batch())
    "value"
  })
  expect_identical(inner, "value")
  expect_false(.nl_internal_batch())

  # It restores on an error too, so a failed diagnostic cannot leave the rest
  # of the session silenced.
  expect_error(.nl_with_internal_batch(stop("boom")), "boom")
  expect_false(.nl_internal_batch())

  # The joint side reaches the same flag through the wrapper that already
  # quiets the checkpoint and the progress bar for an internal re-dispatch.
  .joint_with_quiet_opts(expect_true(.nl_internal_batch()))
  expect_false(.nl_internal_batch())
})

test_that("the Pareto-k re-evaluation does not warn about the caller's grid (#614)", {
  skip_on_cran()
  set.seed(1)
  N <- 24L
  region <- rep(seq_len(6L), each = 4L)
  X <- cbind(1, rnorm(N))
  y <- rnorm(N) + 0.5 * X[, 2L]
  fit_with <- function(n_nodes, ...) {
    tulpa_nested_laplace(
      y = y, n_trials = rep(1L, N), X = X,
      prior = list(list(type = "iid", obs_idx = region, n_units = 6L,
                        sigma_grid = exp(seq(log(0.2), log(1.5),
                                             length.out = n_nodes)))),
      family = "gaussian", phi = 0.49,
      control = utils::modifyList(list(progress = FALSE), list(...)))
  }

  # The issue's repro: 7 caller-chosen nodes, a 200-draw diagnostic batch.
  w <- grid_warnings(fit <- fit_with(7L, diagnose_k = TRUE, k_samples = 200L))
  expect_identical(w, character(0))
  # The diagnostic really ran -- otherwise the silence proves nothing.
  expect_true(is.finite(fit$pareto_k) || !is.na(fit$pareto_k_declined))
  expect_length(fit$weights, 7L)
})

test_that(".nl_multi_grid_warn() fires on measured elapsed time, gated by .nl_internal_batch() (gcol33/tulpa#820, #614)", {
  over <- tulpa:::.NL_MULTI_GRID_WARN_SECONDS + 1

  # A grid the caller DID choose still warns once it crosses the measured
  # wall-clock threshold, and the message names its own cell count.
  w <- grid_warnings(
    tulpa:::.nl_multi_grid_warn(elapsed = over, n_cells = 60L,
                                remedy = "Reduce per-block grid sizes."))
  expect_length(w, 1L)
  expect_match(w, "60 cells", fixed = TRUE)
  expect_match(w, "Reduce per-block grid sizes.", fixed = TRUE)
  expect_false(any(grepl("200 cells", w, fixed = TRUE)))

  # Below the threshold: silent regardless of cell count.
  w0 <- grid_warnings(
    tulpa:::.nl_multi_grid_warn(elapsed = 0.01, n_cells = 60L, remedy = "x"))
  expect_identical(w0, character(0))

  # An internal re-dispatch (the Pareto-k diagnostic's own re-evaluation) is
  # exempt even when IT is the slow one -- #614's whole point.
  w2 <- grid_warnings(
    tulpa:::.nl_with_internal_batch(
      tulpa:::.nl_multi_grid_warn(elapsed = over, n_cells = 200L, remedy = "x")))
  expect_identical(w2, character(0))
})
