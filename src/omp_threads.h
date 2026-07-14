#pragma once

// Team size for intra-chain OpenMP regions that have no R-supplied thread
// count. Respects OMP_NUM_THREADS / omp_set_num_threads() through
// omp_get_max_threads(), and the OMP_THREAD_LIMIT hard clamp through
// omp_get_thread_limit(), so environments that restrict OpenMP (check farms,
// schedulers) bound the team even when the nthreads-var is unset. Bounded
// above by the number of independent work items.

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

inline int tulpa_omp_team_size(int n_work_items) {
#ifdef _OPENMP
    int cap = omp_get_max_threads();
    int limit = omp_get_thread_limit();
    if (limit > 0) cap = std::min(cap, limit);
    return std::max(1, std::min(cap, n_work_items));
#else
    (void)n_work_items;
    return 1;
#endif
}
