# Wall-clock fit timing for the nested-Laplace front-door fitters
#. A single source of truth shared by tulpa_nested_laplace,
# tulpa_nested_laplace_joint(), and the multi-block joint dispatch, so every
# fit carries `fit$timing` -- a named numeric of elapsed seconds -- without each
# fitter re-deriving the bookkeeping.

# Accumulating phase timer. `mark(name)` charges the wall-clock elapsed since
# the previous mark to the named bucket; calling `mark(name)` again later adds
# to the same bucket, so phases that the driver interleaves (e.g. inner-grid
# solves split across a refinement and a consistency pass) accumulate cleanly
# under one label. `timing()` returns `c(total = ..., <buckets in first-touch
# order>)` in seconds, with `total` the independent start-to-now wall-clock.
#
# Wall-clock (proc.time()[["elapsed"]]) is deliberate: the inner grid runs
# multithreaded, so summed CPU time would overstate it -- the user asks how long
# the fit took on the wall, not how many core-seconds it burned.
.tulpa_timer <- function() {
  start   <- proc.time()[["elapsed"]]
  last    <- start
  buckets <- character(0)
  vals    <- numeric(0)
  mark <- function(name) {
    now   <- proc.time()[["elapsed"]]
    delta <- now - last
    last <<- now
    i <- match(name, buckets)
    if (is.na(i)) {
      buckets <<- c(buckets, name)
      vals    <<- c(vals, delta)
    } else {
      vals[i] <<- vals[i] + delta
    }
    invisible(NULL)
  }
  timing <- function() {
    total <- proc.time()[["elapsed"]] - start
    out   <- c(total, vals)
    names(out) <- c("total", buckets)
    out
  }
  list(mark = mark, timing = timing)
}

# Human-readable wall-clock duration: "5h 25m", "2h 09m", "3m 12s", "4.2s".
# Minutes/seconds zero-pad to two digits once an hour/minute leads, so the
# columns line up in a print summary.
.format_duration <- function(seconds) {
  if (length(seconds) != 1L || !is.finite(seconds) || seconds < 0) {
    return(NA_character_)
  }
  s   <- round(seconds)
  h   <- s %/% 3600L
  m   <- (s %% 3600L) %/% 60L
  sec <- s %% 60L
  if (h > 0L) return(sprintf("%dh %02dm", h, m))
  if (m > 0L) return(sprintf("%dm %02ds", m, sec))
  sprintf("%.1fs", seconds)
}

# One-line timing summary for a fit's print method: "fit in 5h 25m (grid 2h
# 09m)". The grid clause is appended only when a `grid` bucket is present.
# Returns NULL when no usable timing is attached, so the caller can skip the
# line entirely.
.timing_summary_line <- function(timing) {
  if (is.null(timing) || !is.numeric(timing) || is.null(names(timing)) ||
      !("total" %in% names(timing))) {
    return(NULL)
  }
  total <- .format_duration(timing[["total"]])
  if (is.na(total)) return(NULL)
  grid <- if ("grid" %in% names(timing)) {
    g <- .format_duration(timing[["grid"]])
    if (is.na(g)) "" else sprintf(" (grid %s)", g)
  } else {
    ""
  }
  sprintf("fit in %s%s", total, grid)
}

#' Print method for nested-Laplace fits
#'
#' Compact one-screen summary of a [tulpa_nested_laplace()] or
#' [tulpa_nested_laplace_joint()] fit: the hyperparameters integrated over, the
#' outer-grid size, the outer Pareto-\eqn{\hat{k}} accuracy diagnostic when
#' present, and the wall-clock timing line (`"fit in 5h 25m (grid 2h 09m)"`)
#' when `fit$timing` is attached. Inherited by the single-block joint and
#' multi-block joint subclasses.
#'
#' @param x A `tulpa_nested_laplace` fit (or a joint subclass).
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.tulpa_nested_laplace <- function(x, ...) {
  is_joint <- inherits(x, "tulpa_nested_laplace_joint")
  is_multi <- inherits(x, "tulpa_nested_laplace_joint_multi")
  header <- if (is_multi) {
    "tulpa joint nested-Laplace fit (multi-block)"
  } else if (is_joint) {
    "tulpa joint nested-Laplace fit"
  } else {
    "tulpa nested-Laplace fit"
  }
  cat(header, "\n")

  nms <- x$theta_names %||% names(x$theta_mean)
  if (!is.null(nms) && length(nms) > 0L) {
    cat("  hyperparameters:", paste(nms, collapse = ", "))
  } else {
    cat("  hyperparameters: (none)")
  }
  n_cells <- if (!is.null(x$theta_grid)) {
    if (is.matrix(x$theta_grid)) nrow(x$theta_grid) else length(x$theta_grid)
  } else {
    length(x$log_marginal)
  }
  if (length(n_cells) == 1L && n_cells > 0L) {
    cat(sprintf("  (grid: %d cells)", n_cells))
  }
  cat("\n")

  if (!is.null(x$pareto_k) && length(x$pareto_k) == 1L && !is.na(x$pareto_k)) {
    verdict <- if (x$pareto_k < 0.7) "reliable" else "escalate to Gibbs debias"
    cat(sprintf("  outer pareto-k: %.2f (%s)\n", x$pareto_k, verdict))
  }

  line <- .timing_summary_line(x$timing)
  if (!is.null(line)) cat("  ", line, "\n", sep = "")

  invisible(x)
}
