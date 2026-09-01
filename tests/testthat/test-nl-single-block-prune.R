# Cheap-pass outer-grid screening on the single-block nested-Laplace front door.
#
# `run_nested_laplace_grid` ranks every cell with a short warm-started inner
# Newton and skips the full solve on cells the ranking puts below a tolerance.
# The single-block entries declare the toggle, so `tulpa_nested_laplace()` can
# reach it, and five things have to hold once it can:
#
#   1. a screened fit reports the screen (or the gate's fallback), never
#      neither, so a reader can always tell which grid produced the numbers;
#   2. the pruned posterior agrees with the full one on the fixed effects, to
#      within the mass the screen dropped;
#   3. the OFF path is the path it always was, to the bit;
#   4. a tolerance outside [0, 1) and a screening depth below one step are
#      refused where the caller set them;
#   5. a screen the gate distrusts is REPLACED by the full grid, and says so.

# ICAR chain with a wide tau axis: the outer nodes sit far enough below the
# posterior mode that the screen has genuine tail cells to drop.
.nlp_fixture <- function(seed = 704L) {
  set.seed(seed)
  S <- 24L
  reps <- 6L
  nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- lengths(nb)
  field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
  idx <- rep(seq_len(S), each = reps)
  N <- length(idx)
  x <- rnorm(N)
  y <- rpois(N, exp(0.2 + 0.5 * x + 0.6 * field[idx]))
  list(
    y = as.numeric(y),
    n_trials = rep(1L, N),
    X = cbind(1, x),
    prior = list(type = "icar", n_spatial_units = S, spatial_idx = idx,
                 adj_row_ptr = as.integer(c(0L, cumsum(nn))),
                 adj_col_idx = as.integer(unlist(nb)) - 1L,
                 n_neighbors = as.integer(nn),
                 tau_grid = c(0.02, 0.1, 0.5, 1, 2, 5, 20, 100)))
}

# The placement pass and the two diagnostics are pinned off: they re-solve the
# grid on their own triggers, and a pruned grid can trip a trigger the full one
# does not, which would make the paired comparison below a comparison of two
# different grids.
.nlp_fit <- function(f, ctrl = list()) {
  tulpa_nested_laplace(
    f$y, f$n_trials, f$X, prior = f$prior, family = "poisson",
    control = utils::modifyList(
      list(auto_recenter = FALSE, diagnose_k = FALSE, diagnose_skew = FALSE),
      ctrl))
}

test_that("a screened single-block fit reports either the screen or the fallback", {
  skip_on_cran()
  f <- .nlp_fixture()
  p <- .nlp_fit(f, list(prune = TRUE))

  screened <- "prune_mask" %in% names(p)
  fell_back <- isTRUE(p$prune_fallback_triggered)
  expect_true(xor(screened, fell_back))

  if (fell_back) {
    expect_true(nzchar(p$prune_fallback_reason))
  } else {
    expect_true(all(c("prune_cheap_log_marginal", "prune_n_pruned",
                      "prune_tol") %in% names(p)))
    expect_equal(p$prune_tol, tulpa:::.nl_screen("prune_tol"))
    expect_equal(length(p$prune_mask), length(p$log_marginal))
    expect_equal(length(p$prune_cheap_log_marginal), length(p$log_marginal))
    # A pruned cell is exactly a cell the full solve never ran on.
    expect_equal(sum(p$prune_mask), p$prune_n_pruned)
    expect_true(all(!is.finite(p$log_marginal[p$prune_mask])))
  }
})

test_that("an explicit prune_tol reaches the kernel", {
  skip_on_cran()
  f <- .nlp_fixture()
  p <- .nlp_fit(f, list(prune = TRUE, prune_tol = 1e-8))
  skip_if(isTRUE(p$prune_fallback_triggered),
          "the safety gate replaced the pruned grid with the full one")
  expect_equal(p$prune_tol, 1e-8)
})

test_that("a pruned fit agrees with the full grid on the fixed effects", {
  skip_on_cran()
  f <- .nlp_fixture()
  full <- .nlp_fit(f)
  pruned <- .nlp_fit(f, list(prune = TRUE))
  skip_if(isTRUE(pruned$prune_fallback_triggered),
          "the safety gate replaced the pruned grid with the full one")

  # The wide tau axis is what makes this a real comparison: without a dropped
  # cell the two fits are the same solve.
  expect_gt(pruned$prune_n_pruned, 0L)

  # What the screen removes is cells below prune_tol of the posterior, so the
  # estimates move by at most the mass it dropped. This is a dropped-mass
  # bound, not machine precision.
  expect_lt(max(abs(coef(pruned) - coef(full))), 0.02)
  expect_lt(max(abs(pruned$theta_mean - full$theta_mean)) /
              max(1, max(abs(full$theta_mean))), 0.05)
})

test_that("prune = FALSE is the default path, unchanged", {
  skip_on_cran()
  f <- .nlp_fixture()
  a <- .nlp_fit(f)
  b <- .nlp_fit(f, list(prune = FALSE))
  # An explicit tolerance with the toggle off is still the full grid: the
  # toggle, not the tolerance, is what turns screening on.
  d <- .nlp_fit(f, list(prune = FALSE, prune_tol = 0.5))

  for (nm in c("log_marginal", "weights", "theta_grid", "theta_mean",
               "theta_sd", "modes", "n_iter")) {
    expect_equal(b[[nm]], a[[nm]], tolerance = 0)
    expect_equal(d[[nm]], a[[nm]], tolerance = 0)
  }
  expect_false("prune_mask" %in% names(a))
  expect_false("prune_mask" %in% names(b))
  expect_false("prune_mask" %in% names(d))
})

test_that("the screening knobs are validated where the caller set them", {
  f <- .nlp_fixture()

  for (bad in list(1, 1.5, -0.1, NA_real_, Inf, c(0.1, 0.2), "x")) {
    expect_error(.nlp_fit(f, list(prune = TRUE, prune_tol = bad)), "prune_tol")
  }
  # Validated whatever `prune` says, so the message names the argument rather
  # than surfacing at the next fit that happens to switch screening on.
  expect_error(.nlp_fit(f, list(prune_tol = 2)), "prune_tol")

  for (bad in list(0L, 0.5, -1L, NA_integer_, Inf, c(1L, 2L), "x")) {
    expect_error(.nlp_fit(f, list(screen_iters = bad)), "screen_iters")
  }

  expect_error(.nlp_fit(f, list(prune_tolerance = 1e-3)), "Unknown control knob")
})

test_that(".nl_prune_gate replaces a screen it distrusts with the full grid", {
  screened <- list(log_marginal = c(-10, -3, -4),
                   prune_mask = c(TRUE, FALSE, FALSE),
                   prune_n_pruned = 1L,
                   prune_argmax_disagree = TRUE,
                   prune_cheap_full_gap = 0.1)
  full <- list(log_marginal = c(-9, -3, -4))
  calls <- 0L
  resolve_full <- function() {
    calls <<- calls + 1L
    full
  }

  # Nothing was screened at a zero tolerance, so the result is handed back
  # untouched and the full grid is never solved a second time.
  expect_identical(tulpa:::.nl_prune_gate(screened, 0, resolve_full), screened)
  expect_identical(calls, 0L)

  out <- NULL
  expect_warning(
    out <- tulpa:::.nl_prune_gate(screened, 1e-3, resolve_full),
    "cheap-pass prune is unreliable")
  expect_identical(calls, 1L)
  expect_true(out$prune_fallback_triggered)
  expect_match(out$prune_fallback_reason, "argmax")
  expect_false("prune_mask" %in% names(out))
  expect_equal(out$log_marginal, full$log_marginal)
})

test_that(".nl_prune_gate keeps a screen it trusts", {
  screened <- list(log_marginal = c(-10, -3, -4),
                   prune_mask = c(TRUE, FALSE, FALSE),
                   prune_n_pruned = 1L,
                   prune_argmax_disagree = FALSE,
                   prune_cheap_full_gap = 0.1)
  expect_identical(
    tulpa:::.nl_prune_gate(screened, 1e-3,
                           function() stop("the full grid must not be solved")),
    screened)
})

# The per-cell fixed-effect retention was written against a grid where every
# cell is solved, and gcol33/tulpa#639 made the other combination reachable on
# this front door: a pruned cell holds no precision, so
# `.nl_attach_grid_hessians()` used to hand its empty CSC to
# `Matrix::sparseMatrix()` and abort the whole fit. The slot has to stay BLANK
# rather than be dropped -- assigning NULL removes the element, and a list
# shorter than `weights` fails `.nested_fixed_moments()`'s length check and NAs
# the entire coefficient table (gcol33/tulpa#345).
test_that("keep_grid_hessians survives a prune that drops cells", {
  skip_on_cran()
  f <- .nlp_fixture()
  p <- .nlp_fit(f, list(prune = TRUE, prune_tol = 1e-3,
                        keep_grid_hessians = TRUE))
  skip_if(isTRUE(p$prune_fallback_triggered),
          "the gate replaced the screen, so no cell was pruned")

  n_grid <- length(p$weights)
  expect_length(p$grid_hessians, n_grid)
  expect_length(p$grid_modes, n_grid)

  blank <- vapply(p$grid_hessians, is.null, logical(1))
  # A cell without a retained precision is exactly a cell the screen dropped,
  # and a dropped cell carries log_marginal = -Inf, hence zero weight -- so
  # nothing the report reads was in one.
  expect_true(any(blank))
  expect_identical(which(blank), which(p$weights == 0))
  expect_identical(which(blank), which(vapply(p$grid_modes, is.null, logical(1))))

  # The report is available and whole: the retention declines nothing, and the
  # mass is 1 because the blank cells held none.
  ci <- confint(p)
  expect_true(all(is.finite(as.matrix(ci))))
  expect_true(is.na(attr(ci, "interval_declined")))
  expect_equal(attr(ci, "retained_mass"), 1)

  # Against the unpruned grid: the difference is the mass the screen dropped,
  # not an artifact of the retention.
  full <- .nlp_fit(f, list(prune = FALSE, keep_grid_hessians = TRUE))
  expect_equal(unname(coef(p)), unname(coef(full)), tolerance = 1e-3)
  expect_equal(unname(as.matrix(ci)), unname(as.matrix(confint(full))),
               tolerance = 1e-3)
})

test_that("a multi-block prior refuses the knobs its entry does not declare", {
  f <- .nlp_fixture()
  multi <- f
  multi$prior <- list(f$prior)
  expect_error(.nlp_fit(multi, list(screen_iters = 3L)), "single-block")
  expect_error(.nlp_fit(multi, list(fitted_var = FALSE)), "single-block")
})
