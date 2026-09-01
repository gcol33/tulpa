# gcol33/tulpa#614. The soft-cap cell-count warning is advice addressed to
# whoever chose the count. The outer Pareto-k diagnostic re-evaluates
# `log_marginal` at `control$k_samples` (default 200) importance draws by
# substituting them for the block's grid axis and re-dispatching through the
# ordinary fitter, so a caller who chose 7 nodes was warned about 200 cells and
# advised to "reduce per-block grid sizes", which does not reach the number in
# the message. `.nl_internal_batch()` is the one predicate that separates the
# two, and the fixtures here hold both directions of it: an internal batch is
# silent, a grid the caller really did choose still warns.

grid_warnings <- function(expr) {
  w <- character(0)
  withCallingHandlers(
    force(expr),
    warning = function(cnd) {
      if (grepl("multi-block grid has", conditionMessage(cnd), fixed = TRUE)) {
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

  # Control: a grid the caller DID choose still warns, and names its own cell
  # count rather than the diagnostic's sample size.
  w2 <- grid_warnings(fit_with(60L, diagnose_k = TRUE, k_samples = 200L))
  expect_length(w2, 1L)
  expect_match(w2, "has 60 cells")
  expect_false(any(grepl("200 cells", w2)))
})
