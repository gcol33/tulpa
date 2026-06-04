# Inner-loop thread resolution.
#
# The per-observation inner OpenMP loops (eta assembly + sparse Hessian
# scatter) scale with physical performance cores but lose ground once they
# spill onto a hybrid CPU's efficiency cores or SMT siblings: on a 14900K the
# beta-arm inner scatter is ~4x slower at 16 threads than at 8. So the
# requested inner thread count is capped at the performance-core count by
# default, with control$n_threads_scatter as the override.

# Physical performance-core count, or NA when the topology is unknown (off
# Windows, or detection failed). Cached in C++.
.tulpa_perf_cores <- function() {
    pc <- tryCatch(cpp_performance_core_count(), error = function(e) 0L)
    if (length(pc) != 1L || is.na(pc) || pc < 1L) NA_integer_ else as.integer(pc)
}

# Resolve the inner per-observation thread count from the requested n_threads
# and the optional control$n_threads_scatter override. The cap defaults to the
# performance-core count; when n_threads_scatter is supplied it replaces that
# default. An unknown topology and no override leaves n_threads unchanged.
.tulpa_inner_threads <- function(n_threads, n_threads_scatter = NULL) {
    n_threads <- as.integer(n_threads)
    cap <- if (!is.null(n_threads_scatter)) as.integer(n_threads_scatter)
           else .tulpa_perf_cores()
    if (length(cap) != 1L || is.na(cap) || cap < 1L) return(n_threads)
    as.integer(min(n_threads, cap))
}
