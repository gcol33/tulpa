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

// Team size for a region whose caller supplies an explicit thread count:
// the requested count, clamped by the environment (OMP_THREAD_LIMIT, max
// threads) and the number of work items. Use this as a num_threads(...)
// clause instead of mutating the process-global default with
// omp_set_num_threads(). n_requested <= 0 defers to the environment cap.
inline int tulpa_omp_team_size_req(int n_requested, int n_work_items) {
    int cap = tulpa_omp_team_size(n_work_items);
    if (n_requested <= 0) return cap;
    return std::max(1, std::min(n_requested, cap));
}

// One-thread routes take a plain loop instead of a parallel region.
//
// A region entered with a team of one -- whether by num_threads(1) or by an
// `if(...)` clause that turns out false -- serialises the BODY but still enters
// libgomp, and that entry is not free: gcol33/tulpa#365 measured 7.6 us of it
// per objective evaluation on the joint path, 28% of the per-auxiliary-draw
// cost of the corrected integrated Laplace. It is paid on every line-search
// trial of every Newton solve and on every sweep of a Gibbs sampler, so a
// region reached per evaluation or per iteration branches here.
//
// The serialised region body runs in index order, which is exactly what the
// plain loop does, so the two routes are bit-identical -- the reduction
// included, since a one-thread reduction has a single private copy summed in
// the same order.
//
// The pragma lives in this function, so a caller's locals never live across a
// region in their own frame (the OpenMP worker-stack rule the gcol33/tulpa#253
// note states). `body` is called by reference and is never copied.

template <typename Body>
inline void tulpa_parallel_for(int team, int n, Body&& body) {
#ifdef _OPENMP
    if (team > 1) {
        #pragma omp parallel for schedule(static) num_threads(team)
        for (int i = 0; i < n; i++) body(i);
        return;
    }
#else
    (void)team;
#endif
    for (int i = 0; i < n; i++) body(i);
}

// Sum of `body(i)` over i in [0, n), same policy.
template <typename Body>
inline double tulpa_parallel_sum(int team, int n, Body&& body) {
    double acc = 0.0;
#ifdef _OPENMP
    if (team > 1) {
        #pragma omp parallel for reduction(+:acc) schedule(static) \
            num_threads(team)
        for (int i = 0; i < n; i++) acc += body(i);
        return acc;
    }
#else
    (void)team;
#endif
    for (int i = 0; i < n; i++) acc += body(i);
    return acc;
}
