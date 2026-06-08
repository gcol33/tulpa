# progress_iter.R
# Shared iteration progress + ETA reporter for tulpa's R-side fitting loops
# (EM-Laplace here; community EM in tulpaObs reuses it via tulpa:::).
#
# It mirrors the C++ tulpa_progress::GridProgress wire format so a detached
# reader sees the SAME heartbeat file whether the loop that produced it was a
# C++ outer grid, a C++ NUTS sampler, or an R EM loop: one overwritten line
# holding "<done> <total> <elapsed_s> <eta_s>".
#
# Two independently gated channels, matching GridProgress:
#   * console bar -- the noisy channel, emitted when `progress` is set;
#   * heartbeat file -- written whenever `progress_file` is non-empty,
#     independent of `progress`. A file write survives a detached
#     Start-Process / nohup stdout buffer where a console flush does not, so it
#     is the only liveness signal on a headless box (gcol33/tulpaObs#43).
#
# Config comes from the scoped `tulpa.nl_progress` option that tobs() sets from
# control$progress[.every/.throttle/.file]; a fitter outside tobs() that wants
# progress sets the same option.

# Human-readable duration, ASCII only (logs / redirects / CRAN). Matches
# tulpa_progress::format_secs.
.tulpa_progress_format_secs <- function(s) {
  if (!is.finite(s) || s < 0) return("?")
  if (s < 90)   return(sprintf("%.0fs", s))
  if (s < 5400) return(sprintf("%.1fmin", s / 60))
  sprintf("%.1fh", s / 3600)
}

# Build a progress reporter for a counted R loop. `total` is the iteration
# denominator (e.g. max_iter for EM); `unit` names one step in the console line
# ("iter", "species", ...). `threads` is the active outer-thread count: when
# > 1 it is appended to the console line as "| N threads" so the parallelism is
# visible live and in detached-run logs (gcol33/tulpa#88); a serial loop leaves
# it at 1 and the field is omitted, mirroring the C++ GridProgress reporter.
# Returns a list of two closures:
#   tick()   -- call once per completed step;
#   finish() -- call once after the loop (force a final 100%/heartbeat emit).
# When no channel is wanted both are zero-overhead no-ops.
.tulpa_iter_progress <- function(label, total, unit = "iter", threads = 1L) {
  opt <- getOption("tulpa.nl_progress", NULL)
  on       <- isTRUE(opt$progress)
  file     <- if (is.character(opt$progress_file)) opt$progress_file else ""
  every    <- if (length(opt$progress_every))    as.integer(opt$progress_every) else 0L
  throttle <- if (length(opt$progress_throttle)) as.numeric(opt$progress_throttle) else 2
  threads  <- if (length(threads)) as.integer(threads)[1] else 1L

  if (!on && !nzchar(file)) {
    return(list(tick = function() invisible(NULL),
                finish = function() invisible(NULL)))
  }

  total <- max(as.integer(total), 0L)
  start <- proc.time()[["elapsed"]]
  last  <- start
  done  <- 0L
  last_emit_done <- 0L

  emit <- function(force) {
    now   <- proc.time()[["elapsed"]]
    since <- now - last
    first <- done == 1L
    by_steps <- every > 0L && (done - last_emit_done) >= every
    by_time  <- throttle > 0 && since >= throttle
    if (!force && !first && !by_steps && !by_time) return(invisible(NULL))

    elapsed <- now - start
    frac <- if (total > 0) done / total else 0
    eta  <- if (frac > 0 && frac < 1) elapsed * (1 - frac) / frac else 0
    per  <- if (done > 0) elapsed / done else 0

    if (nzchar(file)) {
      con <- file(file, open = "w")
      cat(sprintf("%d %d %.1f %.1f\n", done, total, elapsed, eta), file = con)
      close(con)
    }
    if (on) {
      pct <- if (total > 0) round(frac * 100) else 0
      thr <- if (isTRUE(threads > 1L)) sprintf(" | %d threads", threads) else ""
      cat(sprintf("[%s] %d/%d %s (%d%%) | elapsed %s | ETA ~%s | %.2fs/%s%s\n",
                  label, done, total, unit, pct,
                  .tulpa_progress_format_secs(elapsed),
                  if (frac >= 1) "done" else .tulpa_progress_format_secs(eta),
                  per, unit, thr))
      flush.console()
    }
    last <<- now
    last_emit_done <<- done
    invisible(NULL)
  }

  list(
    tick   = function() { done <<- done + 1L; emit(FALSE) },
    finish = function() { emit(TRUE) }
  )
}
