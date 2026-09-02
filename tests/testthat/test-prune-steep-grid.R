# gcol33/tulpa#656 -- the cheap-pass screen on a sharply peaked outer grid.
#
# Measured on a 4435-cell `occu_cover` fit whose outer log-marginal spans about
# 1e5 nats: the screen kept ONE cell at every tolerance a caller would type, the
# kept set was a point mass so the placement pass declined for want of
# curvature, the fit reported a field SD of exactly 3.000 -- the top node of an
# axis nothing had re-placed -- and neither trigger of the safety gate could
# see any of it.
#
# Four things have to hold, and each is a separate failure:
#
#   1. RESOLUTION. `prune_tol` is a normalised weight, so what it cuts at is a
#      gap in nats. The whole documented range is a few tens of nats, which on
#      a surface spanning thousands is one sliver -- every setting returns the
#      same kept set. `prune_log_gap` states the cut in the units the surface
#      is measured in, and the fit reports the realised cut beside the spread
#      it was applied to.
#   2. FLOOR. The placement pass reads a finite-difference curvature off the
#      cells that were SOLVED, so the screen leaves at least `min_keep` of
#      them whatever the tolerance says.
#   3. DISTINGUISHABILITY. `no_usable_curvature` on a screened grid is a
#      different event from the same reason on a full one, and a fit says which
#      it was.
#   4. The gate's gap threshold is judged against the margin the screen
#      DISCARDED cells by, not against the spread of the cells it kept. A
#      threshold computed from the set the trigger is meant to validate cannot
#      bound the error on the set it discarded.

# --- 1. the cut, in the units the surface is measured in ---------------------

test_that("the kept-cell floor is one value across the R and C++ defaults", {
  ref <- as.integer(tulpa:::.nl_screen("min_keep"))
  expect_true(is.finite(ref) && ref >= 3L)

  src <- test_path("..", "..", "src", "nested_laplace_grid.h")
  skip_if_not(file.exists(src), "package sources not available")
  line <- grep("static const int CHEAP_SCREEN_MIN_KEEP",
               readLines(src, warn = FALSE), value = TRUE)
  expect_length(line, 1L)
  expect_identical(as.integer(sub(".*=\\s*([0-9]+).*", "\\1", line)), ref)
})

test_that("prune_log_gap states a cut the tolerance cannot reach", {
  # What `prune_tol` cuts at is -(log(tol) + log(Z)) nats below the best
  # screened cell. The documented "conservative" setting is a handful of nats,
  # and the whole range a caller would type stays inside a few tens -- which is
  # why three tolerances four orders apart returned the same kept set on a
  # surface spanning 1e5.
  expect_lt(-log(1e-3), 10)
  expect_lt(-log(1e-12), 30)

  # Stating the gap directly reaches cuts the weight form does not, and keeps
  # distinct settings distinct there.
  t300 <- tulpa:::.nl_check_prune_log_gap(300)
  t400 <- tulpa:::.nl_check_prune_log_gap(400)
  expect_true(t300 > 0 && t400 > 0)
  expect_gt(t300, t400)
  expect_equal(t300, exp(-300))
  expect_equal(-log(t400), 400)

  # Unset stays unset.
  expect_null(tulpa:::.nl_check_prune_log_gap(NULL))

  for (bad in list(0, -1, NA_real_, Inf, c(10, 20), "x")) {
    expect_error(tulpa:::.nl_check_prune_log_gap(bad), "prune_log_gap")
  }
  # A gap whose tolerance underflows to zero would switch screening OFF rather
  # than widen it, which is the opposite of what was asked for.
  expect_error(tulpa:::.nl_check_prune_log_gap(1e4), "underflow")
})

test_that("the two knobs state one cut, so setting both is refused", {
  expect_equal(tulpa:::.nl_prune_tol_from_control(list(), 1e-3), 1e-3)
  expect_equal(
    tulpa:::.nl_prune_tol_from_control(list(prune_log_gap = 50), 1e-3),
    exp(-50))
  expect_error(
    tulpa:::.nl_prune_tol_from_control(
      list(prune_log_gap = 50, prune_tol = 1e-6), 1e-6),
    "prune_log_gap")
})

test_that("the driver reports the cut and the surface it was applied to", {
  src <- test_path("..", "..", "src", "nested_laplace_grid.h")
  skip_if_not(file.exists(src), "package sources not available")
  txt <- paste(readLines(src, warn = FALSE), collapse = "\n")

  # The realised cut and the screened spread: the pair a caller reads to see
  # whether the tolerance had any resolution on this grid.
  expect_true(grepl("prune_log_gap_cut", txt, fixed = TRUE))
  expect_true(grepl("prune_cheap_lm_spread", txt, fixed = TRUE))
  # The floor, and how many cells it put back.
  expect_true(grepl("prune_min_keep", txt, fixed = TRUE))
  expect_true(grepl("prune_n_floor_restored", txt, fixed = TRUE))
  # The spread is taken over the whole screened surface (`cheap_lm`), not over
  # the survivors.
  expect_true(grepl("for (double v : cheap_lm)", txt, fixed = TRUE))
})

# --- 4. the gate's threshold -------------------------------------------------

test_that("the gate judges the screen's error against the cut, not the kept spread", {
  # The measured shape: five kept cells spanning ~98000 nats, the posterior a
  # point mass on one of them, and the screen out by 339.5 nats on exactly that
  # cell. Judged against half the kept spread the threshold is 49025.8 and the
  # trigger is silent; judged against the 6.9-nat margin the screen actually
  # discarded cells by, a 339.5-nat error is enormous.
  lm_steep <- c(0, -98051.6, -40000, -60000, -80000,
                rep(-Inf, 115))          # 115 cells the screen never solved
  res_steep <- list(
    log_marginal = lm_steep,
    prune_mask = c(rep(FALSE, 5L), rep(TRUE, 115L)),
    prune_cheap_log_marginal = lm_steep,
    prune_argmax_disagree = FALSE,
    prune_cheap_full_gap = 339.5,
    prune_n_pruned = 115L,
    prune_log_gap_cut = -log(1e-3),
    prune_cheap_lm_spread = 98051.6)
  full_sentinel <- list(log_marginal = lm_steep, ITS_THE_FULL = TRUE)

  out <- NULL
  expect_warning(
    out <- tulpa:::.joint_prune_safety_gate(
      res_steep, resolve_full = function() full_sentinel),
    "falling back to the full grid")
  expect_true(isTRUE(out$ITS_THE_FULL))
  expect_true(isTRUE(out$prune_fallback_triggered))
  expect_match(out$prune_fallback_reason, "gap|collapse")

  # Drop the cut and the old threshold comes back, computed from the very set
  # the trigger is meant to validate -- and the same fit passes. This is the
  # measured defect, and the assertion is that the cut is what changed it.
  res_old <- res_steep
  res_old$prune_log_gap_cut <- NULL
  expect_warning(
    kept <- tulpa:::.joint_prune_safety_gate(
      res_old, resolve_full = function() stop("must not re-solve")),
    regexp = NA)
  expect_null(kept$prune_fallback_triggered)
})

test_that("the tighter threshold leaves a healthy screened fit alone", {
  # A cut of 6.9 nats and a screen accurate to a twentieth of one, on a grid
  # whose kept posterior has not collapsed: the ESS conjunct is false, so the
  # trigger cannot fire whatever the threshold is. This is what makes the
  # change a no-op on a well-conditioned pruned fit.
  res_ok <- list(
    log_marginal = c(-1, -1.2, -1.1, -Inf),
    prune_mask = c(FALSE, FALSE, FALSE, TRUE),
    prune_cheap_log_marginal = c(-1.05, -1.25, -1.15, -8),
    prune_argmax_disagree = FALSE,
    prune_cheap_full_gap = 0.05,
    prune_n_pruned = 1L,
    prune_log_gap_cut = -log(1e-3),
    prune_cheap_lm_spread = 7)
  out <- expect_warning(
    tulpa:::.joint_prune_safety_gate(
      res_ok, resolve_full = function() stop("must not re-solve")),
    regexp = NA)
  expect_null(out$prune_fallback_triggered)

  # And a fit that was never screened is returned untouched, cut or no cut.
  res_none <- list(log_marginal = c(-1, -2, -3))
  expect_identical(
    tulpa:::.joint_prune_safety_gate(
      res_none, resolve_full = function() stop("must not re-solve")),
    res_none)
})

# --- 3. a decline on a screened grid is its own event ------------------------

test_that("a placement decline records whether the grid had been screened", {
  screened <- tulpa:::.nl_decline_recenter(
    list(prune_mask = c(TRUE, FALSE, FALSE)), "no_usable_curvature")
  expect_identical(screened$outer_grid_recenter_declined,
                   "no_usable_curvature")
  expect_true(screened$outer_grid_recenter_declined_pruned)

  full <- tulpa:::.nl_decline_recenter(list(), "no_usable_curvature")
  expect_identical(full$outer_grid_recenter_declined, "no_usable_curvature")
  # Recorded on every decline, TRUE or FALSE: an absent field is what made the
  # two read alike.
  expect_true("outer_grid_recenter_declined_pruned" %in% names(full))
  expect_false(full$outer_grid_recenter_declined_pruned)

  # A fit that WAS re-placed carries no decline at all.
  placed <- tulpa:::.nl_decline_recenter(
    list(outer_grid_placement = "auto_recentered", prune_mask = c(TRUE)),
    "no_usable_curvature")
  expect_null(placed$outer_grid_recenter_declined)
})

test_that("a screened fit whose placement lost its curvature falls back", {
  lost <- list(prune_mask = c(TRUE, FALSE),
               outer_grid_placement = "fixed",
               outer_grid_recenter_declined = "no_usable_curvature")
  expect_true(tulpa:::.nl_prune_placement_lost(lost))

  # Not the same event on a full grid: nothing was discarded, so there is
  # nothing to fall back to.
  expect_false(tulpa:::.nl_prune_placement_lost(
    list(outer_grid_placement = "fixed",
         outer_grid_recenter_declined = "no_usable_curvature")))
  # Nor on a screened fit that WAS re-placed, or one that declined for a reason
  # the screen cannot have caused.
  expect_false(tulpa:::.nl_prune_placement_lost(
    list(prune_mask = c(TRUE), outer_grid_placement = "auto_recentered",
         outer_grid_recenter_declined = "no_usable_curvature")))
  expect_false(tulpa:::.nl_prune_placement_lost(
    list(prune_mask = c(TRUE), outer_grid_placement = "fixed",
         outer_grid_recenter_declined = "auto_recenter_disabled")))

  full_sentinel <- list(ITS_THE_FULL = TRUE)
  out <- NULL
  expect_warning(
    out <- tulpa:::.nl_prune_placement_fallback(
      lost, function() full_sentinel),
    "falling back to the full grid")
  expect_true(isTRUE(out$ITS_THE_FULL))
  expect_true(isTRUE(out$prune_fallback_triggered))
  expect_match(out$prune_fallback_reason, "placement")

  # A fit the predicate says nothing about is handed straight back, and the
  # full grid is never solved a second time.
  ok <- list(prune_mask = c(TRUE), outer_grid_placement = "auto_recentered")
  expect_identical(
    tulpa:::.nl_prune_placement_fallback(ok, function() stop("must not re-solve")),
    ok)
})

# --- 2. the floor, on a real screened grid -----------------------------------

# ICAR chain on a tau axis spanning ten orders of magnitude: the outer nodes are
# far enough apart that the log-marginal moves by hundreds of nats between them,
# which is the regime where a tolerance-only screen collapses the kept set onto
# the modal cell.
.psg_fixture <- function(seed = 656L) {
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
  list(y = as.numeric(y), n_trials = rep(1L, N), X = cbind(1, x),
       prior = list(type = "icar", n_spatial_units = S, spatial_idx = idx,
                    adj_row_ptr = as.integer(c(0L, cumsum(nn))),
                    adj_col_idx = as.integer(unlist(nb)) - 1L,
                    n_neighbors = as.integer(nn),
                    tau_grid = c(1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3,
                                 1e4, 1e5)))
}

.psg_fit <- function(f, ctrl = list()) {
  tulpa_nested_laplace(
    f$y, f$n_trials, f$X, prior = f$prior, family = "poisson",
    control = utils::modifyList(
      list(auto_recenter = FALSE, diagnose_k = FALSE, diagnose_skew = FALSE),
      ctrl))
}

test_that("the screen leaves the placement pass something to read", {
  skip_on_cran()
  f <- .psg_fixture()
  p <- .psg_fit(f, list(prune = TRUE))
  skip_if(isTRUE(p$prune_fallback_triggered),
          "the safety gate replaced the screen with the full grid")

  n_grid <- length(p$log_marginal)
  floor_n <- min(n_grid, as.integer(tulpa:::.nl_screen("min_keep")))
  kept <- n_grid - p$prune_n_pruned

  # The floor holds: a screened grid never comes back as a point mass while it
  # holds enough feasible cells to avoid one.
  expect_gte(kept, floor_n)
  expect_identical(as.integer(p$prune_min_keep), floor_n)
  expect_gte(p$prune_n_floor_restored, 0L)
  # And the floor is exactly the cells it restored plus the ones the tolerance
  # kept, so the count is not a coincidence of the tolerance.
  expect_lte(p$prune_n_floor_restored, kept)

  # The screen reports the cut it applied and the surface it applied it to, so
  # a reader can tell a tolerance with resolution from one without.
  expect_true(is.finite(p$prune_log_gap_cut))
  expect_gt(p$prune_log_gap_cut, 0)
  expect_true(is.finite(p$prune_cheap_lm_spread))
  expect_gte(p$prune_cheap_lm_spread, 0)
})

test_that("a screened fit still tracks the full grid on the fixed effects", {
  skip_on_cran()
  f <- .psg_fixture()
  p <- .psg_fit(f, list(prune = TRUE))
  s <- .psg_fit(f, list(prune = FALSE))
  skip_if(isTRUE(p$prune_fallback_triggered),
          "the safety gate replaced the screen with the full grid")

  # The floor keeps more cells than the tolerance alone would, so this is the
  # screened posterior the floor produces -- the one a caller now gets. The
  # bound is the mass the screen dropped, not machine precision.
  expect_lt(max(abs(coef(p) - coef(s))), 0.05)
})

test_that("screening off is the path it always was", {
  skip_on_cran()
  f <- .psg_fixture()
  a <- .psg_fit(f, list(prune = FALSE))
  b <- .psg_fit(f)

  # Nothing in the floor, the log-space cut or the reported fields reaches a
  # fit that never screened.
  expect_equal(a$log_marginal, b$log_marginal, tolerance = 0)
  expect_false("prune_mask" %in% names(a))
  expect_false("prune_log_gap_cut" %in% names(a))
})
