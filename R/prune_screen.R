# Cheap-pass screening: stating the cut in nats, and what to do when the screen
# takes the placement pass down with it.
#
# The kernel's screening tolerance is a NORMALISED WEIGHT, so the quantity it
# actually cuts at is a gap in nats below the best cheap cell,
# `-(log(prune_tol) + log(Z))`. Every tolerance a caller would type -- the
# documented `<= 1e-3`, down to `1e-12` -- is 7 to 28 nats of that gap. On an
# outer surface whose log-marginal spans thousands of nats that is one sliver of
# the grid, so the same kept set comes back at every setting and the knob
# documented as controlling how aggressive the screen is controls nothing.
#
# `control$prune_log_gap` states the cut directly, in nats: keep every cell
# within this many of the best cheap cell. It reaches the kernel as
# `prune_tol = exp(-prune_log_gap)`, so the realised cut is the requested gap
# less `log(Z)` -- at most `log(n_grid)` nats narrower than asked, and the fit
# reports the realised value in `prune_log_gap_cut`. The two knobs state the
# same cut in different units, so setting both is an error rather than a silent
# ranking of one over the other.

# `NULL` when the caller set no gap; otherwise the tolerance it maps to.
.nl_check_prune_log_gap <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x <= 0) {
    stop("`prune_log_gap` must be a single finite positive number of nats ",
         "(how far below the best screened cell a cell may fall and still be ",
         "solved in full).", call. = FALSE)
  }
  tol <- exp(-as.numeric(x))
  if (!(tol > 0)) {
    stop("`prune_log_gap = ", format(x), "` is past the range the screening ",
         "tolerance can carry: exp(-prune_log_gap) underflows to zero, which ",
         "switches screening off rather than widening it. Use a smaller gap.",
         call. = FALSE)
  }
  tol
}

# Resolve the screening tolerance a fit runs at from the two knobs that state
# it. `prune_tol` is the caller's validated weight (or the registry default);
# `control$prune_log_gap`, when set, replaces it.
.nl_prune_tol_from_control <- function(control, prune_tol) {
  gap_tol <- .nl_check_prune_log_gap(control$prune_log_gap)
  if (is.null(gap_tol)) return(prune_tol)
  if (!is.null(control$prune_tol)) {
    stop("`control$prune_tol` and `control$prune_log_gap` state the same ",
         "screening cut in different units (a normalised weight against a gap ",
         "in nats). Set one.", call. = FALSE)
  }
  gap_tol
}

# The screen took the placement pass down with it: the fit was screened, the
# outer axis was never re-placed, and the reason is that no curvature could be
# read. On a full grid that reason is a posterior the finite-difference stencil
# could not resolve; on a screened one it is routinely the screen, because the
# kept set collapses onto the cells the stencil would have read. The reported
# axis is then wherever it was laid -- a grid boundary, if the mass ran to one
# -- and nothing on the fit distinguishes that from an estimate.
.nl_prune_placement_lost <- function(res) {
  if (!isTRUE(any(as.logical(res$prune_mask), na.rm = TRUE))) return(FALSE)
  if (identical(res$outer_grid_placement, "auto_recentered")) return(FALSE)
  identical(res$outer_grid_recenter_declined, "no_usable_curvature")
}

# Replace such a fit with the full-grid one, the way the cheap-pass safety gate
# replaces a screen whose ranking it distrusts. `refit` is a zero-argument thunk
# re-running the same fit with screening off.
.nl_prune_placement_fallback <- function(res, refit,
                                         fn = "tulpa_nested_laplace_joint()") {
  if (!.nl_prune_placement_lost(res)) return(res)
  warning(sprintf(
    paste0("%s: the cheap-pass screen left too few solved cells for the outer ",
           "grid placement to read a curvature, so the grid was never ",
           "re-placed and a hyperparameter reported off it is the axis as laid ",
           "rather than an estimate; falling back to the full grid. Set ",
           "control$prune = FALSE to skip the screened attempt."), fn),
    call. = FALSE)
  full <- refit()
  full$prune_fallback_triggered <- TRUE
  full$prune_fallback_reason <-
    "the screened grid left no curvature for the outer grid placement"
  full
}
