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

# The screen offset a refinement round's kernel call carries
# (`.hyper_refine_screen()`), or NULL when the call is not screened: the door
# screens nothing (`tol` 0), the round brought no screen record, or the call
# holds a single cell, which the driver never screens. `hp` maps the cells to
# the door's hyperprior record (`$lp`), NULL where the dispatcher folds its own.
.nl_refine_screen_offset <- function(cells, screen, tol, hp = NULL) {
  if (is.null(screen) || !(tol > 0) || nrow(cells) < 2L) return(NULL)
  .nl_screen_log_offset(cells, if (is.null(hp)) list() else list(hp(cells)),
                        screen = screen)
}

# The dropped-mass bound of a screened grid (`.nl_prune_dropped_mass()`) cell by
# cell. `full` and `cheap` are each cell's full and screened log posterior under
# one measure, `mask` the cells the screen dropped, `ref` the log mass of a grid
# the cells joined (-Inf for none). Every dropped cell is lifted by the worst
# screening error measured on a kept cell, `e_max = max(full - cheap)`, and the
# bound is the lifted dropped mass D as a share of the whole grid's,
#
#   B = D / (R + K + D),   D = sum_{j dropped} exp(cheap_j + e_max),
#
# with K the kept cells' mass and R the joined grid's. Leaving the dropped
# cells out replaces the grid's posterior by its restriction to the kept cells,
# and the total-variation distance between the two is exactly the dropped
# share; x / (K + x) increases in x, so B bounds it whenever D bounds the
# dropped mass. `lift` is each cell's lifted log mass (-Inf on a cell not
# dropped), `log_den` the log of R + K + D, `e_max` the error lifted by.
# `bound` is `Inf` when no kept cell carries a screen to measure the error on,
# 0 when nothing was dropped.
.nl_prune_lifted <- function(full, cheap, mask, ref = -Inf) {
  n <- length(full)
  kept <- !mask & is.finite(full)
  seen <- kept & is.finite(cheap)
  drop <- mask & is.finite(cheap)
  lift <- rep(-Inf, n)
  if (!any(seen)) {
    return(list(bound = Inf, lift = lift, log_den = NA_real_, e_max = NA_real_))
  }
  e_max <- max(full[seen] - cheap[seen])
  lift[drop] <- cheap[drop] + e_max
  log_den <- .tulpa_logsumexp(c(ref, full[kept], lift[drop]))
  list(bound = if (any(drop)) sum(exp(lift[drop] - log_den)) else 0,
       lift = lift, log_den = log_den, e_max = e_max)
}

# The dropped cells a repair solves in full when a screened grid's bound
# (`.nl_prune_lifted()` result `pb`) is past `gate`: the heaviest lifted cells
# first, as many as leave the lifted share of the rest under it, and at least
# one. Every dropped cell when no kept cell measured the screen's error.
.nl_prune_repair_take <- function(pb, gate) {
  drop <- which(is.finite(pb$lift))
  if (!is.finite(pb$bound)) return(drop)
  o <- drop[order(pb$lift[drop], decreasing = TRUE)]
  rest <- rev(cumsum(rev(exp(pb$lift[o] - pb$log_den))))
  o[seq_len(max(1L, sum(rest > gate)))]
}

# The cheap-pass record of a grid whose cells were screened in more than one
# call -- the declared grid, then each refinement round against the grid it
# joined -- read over the merged grid: the count dropped, the offset every cell
# is ranked with under the grid's final measure (`log_hyperprior` plus
# `log_quad`), and the share of the posterior the dropped cells may carry
# (`.nl_prune_dropped_mass()`). Each call bounded its own dropped mass against
# the grid it joined; this is the bound on the fit the caller receives.
.nl_prune_merged_record <- function(res) {
  if (!any(nzchar(res$refining_axis %||% ""))) return(res)
  mask <- res$prune_mask
  n <- length(res$log_marginal)
  if (is.null(mask) || length(mask) != n) return(res)
  hp <- .nl_log_hyperprior_folded(res, n)
  lq <- res$log_quad
  if (is.null(hp) || length(lq) != n) return(res)
  mask <- as.logical(mask)
  mask[is.na(mask)] <- FALSE
  res$prune_mask <- mask
  res$prune_n_pruned <- sum(mask)
  res$prune_screen_log_offset <- hp + lq
  res$prune_screen_log_ref <- NULL
  res$prune_dropped_mass_bound <- .nl_prune_dropped_mass(res)
  res
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
