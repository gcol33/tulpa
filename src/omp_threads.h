#pragma once

// Team size for intra-chain OpenMP regions that have no R-supplied thread
// count. Respects OMP_NUM_THREADS / omp_set_num_threads() through
// omp_get_max_threads(), and the OMP_THREAD_LIMIT hard clamp through
// omp_get_thread_limit(), so environments that restrict OpenMP (check farms,
// schedulers) bound the team even when the nthreads-var is unset. Bounded
// above by the number of independent work items.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// CRAN's check farm sets _R_CHECK_LIMIT_CORES_ and permits at most two cores in
// examples, tests and vignettes. The variable is set once for the lifetime of
// the check process, so it is read once and cached; "false" / "FALSE" / "0"
// disable the limit, matching what R itself treats as unset.
// Returns 0 when no limit applies.
inline int tulpa_omp_check_cap() {
    static const int cap = [] {
        const char* v = std::getenv("_R_CHECK_LIMIT_CORES_");
        if (v == nullptr || *v == '\0') return 0;
        if (std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0 ||
            std::strcmp(v, "0") == 0) return 0;
        return 2;
    }();
    return cap;
}

inline int tulpa_omp_team_size(int n_work_items) {
#ifdef _OPENMP
    int cap = omp_get_max_threads();
    int limit = omp_get_thread_limit();
    if (limit > 0) cap = std::min(cap, limit);
    const int check_cap = tulpa_omp_check_cap();
    if (check_cap > 0) cap = std::min(cap, check_cap);
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
// libgomp, and that entry is not free: 7.6 us of it was measured
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
// region in their own frame, which is the OpenMP worker-stack rule.
// `body` is called by reference and is never copied.

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

// Sum of `body(i)` over i in [0, n), same one-thread policy, and above one
// thread a sum that is a function of (team, n) alone.
//
// `reduction(+:)` leaves the order in which the per-thread private copies are
// combined into the shared variable unspecified, and libgomp combines them as
// each thread finishes. Floating-point addition is not associative, so the sum
// lands an ulp or two apart from one run to the next, and the Newton loop's
// convergence test and line search amplify that into ~1e-13 on a fit's
// log_marginal.
//
// So the range is cut into `team` contiguous chunks HERE, by index arithmetic,
// each chunk summed left to right into its own slot, and the slots added in
// chunk order after the region. Nothing about the answer is left to the
// runtime: not the combination order, and not the partition either, so a fit
// reproduces itself even where the runtime hands back a smaller team than the
// one requested. The parallel loop runs over the chunks rather than over the
// observations, which keeps the combined `parallel for` construct -- the cheap
// libgomp entry -- and writes each slot once, so the slots do not share a line
// under contention.
//
// The result above one thread is still not the one-thread sum: chunking the
// range imposes its own association. Bit-identity ACROSS thread counts is not
// what this buys; reproducibility AT a thread count is.
//
// The slot buffer is allocated only on the multi-thread route, so the serial
// route keeps its plain loop and allocates nothing.
template <typename Body>
inline double tulpa_parallel_sum(int team, int n, Body&& body) {
#ifdef _OPENMP
    if (team > 1) {
        std::vector<double> part(static_cast<std::size_t>(team), 0.0);
        double* slot = part.data();
        const long long nn = static_cast<long long>(n);
        const long long tt = static_cast<long long>(team);
        #pragma omp parallel for schedule(static) num_threads(team)
        for (int t = 0; t < team; t++) {
            const int lo = static_cast<int>(nn * t / tt);
            const int hi = static_cast<int>(nn * (t + 1) / tt);
            double local = 0.0;
            for (int i = lo; i < hi; i++) local += body(i);
            slot[t] = local;
        }
        double acc = 0.0;
        for (int t = 0; t < team; t++) acc += part[t];
        return acc;
    }
#else
    (void)team;
#endif
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += body(i);
    return acc;
}
