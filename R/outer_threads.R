# The outer-grid thread width: what was requested, and what the run gets.
#
# `control$n_threads_outer` is a REQUEST. The grid driver clamps it to the team
# the OpenMP environment will hand out -- `omp_get_max_threads()`, the
# `OMP_THREAD_LIMIT` hard cap, and the check-farm core cap -- before it opens
# the parallel region, because the per-cell block cache sizes its slot array
# from that same number and an unclamped `num_threads(n)` clause would map every
# excess worker onto slot 0 and corrupt the shared factor. The clamp is correct.
#
# What it does not do is say so. A job script exporting `OMP_NUM_THREADS=1` to
# pin the INNER threads and asking for ten outer ones gets one, and the two runs
# are otherwise indistinguishable: the progress reporter prints its thread
# suffix only above one thread, so a clamped-to-one grid and a serial-by-design
# grid produce byte-identical lines.
#
# REPORTED, not warned. The clamp costs wall clock, not correctness, and it is
# the ordinary state under `R CMD check`, where the farm caps cores at two -- a
# warning would fire on every checked example that asks for more, and would be
# deferred to the end of the call, arriving after the hours it was meant to
# save. A message reaches the same stream at the moment the fit starts, and
# `suppressMessages()` silences it.
#
# `cpp_get_max_threads()` goes through the same resolver the driver's clamp
# does, so the two agree on the cap; the driver clamps further by the cell
# count, which is why the width it ran at is read back off the result rather
# than predicted here.
.nl_outer_threads <- function(n_requested,
                              fn = "tulpa_nested_laplace_joint()") {
  req <- suppressWarnings(as.integer(n_requested)[1L])
  if (length(req) != 1L || is.na(req)) req <- 1L
  cap <- tryCatch(as.integer(cpp_get_max_threads())[1L],
                  error = function(e) NA_integer_)
  if (length(cap) != 1L || is.na(cap) || cap < 1L) cap <- 1L
  realised <- if (req <= 0L) cap else min(req, cap)
  if (req > 1L && realised < req) {
    message(sprintf(
      paste0("%s: n_threads_outer: %d requested, running %d (capped by ",
             "omp_get_max_threads(); set OMP_NUM_THREADS >= %d to lift it)."),
      fn, req, realised, req))
  }
  list(requested = req, realised = realised)
}

# Stamp both numbers on the fit, beside the other outer-grid diagnostics. The
# request alone cannot show a clamp and the realised width alone cannot either;
# a benchmark recording a per-cell time needs the pair.
#
# The driver reports the width it actually opened the region at
# (`n_threads_outer_realised` on the kernel result), which is the environment
# cap clamped further by the cell count. Where a fit carries none -- a path that
# never entered the grid driver -- the environment-capped request stands in.
.nl_attach_outer_threads <- function(res, tw) {
  if (!is.list(res)) return(res)
  res$n_threads_outer_requested <- as.integer(tw$requested)
  realised <- res$n_threads_outer_realised
  res$n_threads_outer_realised <- as.integer(
    if (length(realised) == 1L && !is.na(realised)) realised else tw$realised)
  res
}
