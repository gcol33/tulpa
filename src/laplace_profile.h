// laplace_profile.h
// Lightweight phase accumulator behind tulpa_profile(). It times the inner
// Newton solvers (the single-response loop laplace_newton_solve_ll, the dense
// and sparse joint loops), the nested-Laplace outer grid (one scope per cell
// solve) and the NUTS chain (warmup / sampling iterations and every gradient
// evaluation). One facility: every site uses TULPA_PROFILE_PHASE below.
//
// The accumulator is process-global and mutex-guarded so the per-cell solves
// the parallel outer grid runs on its worker threads, and the chains the
// across-chain sampler runs concurrently, all add into the same buffer; a
// thread-local buffer would leave cpp_profile_read() (called on the R main
// thread) seeing only the work that happened to run on that thread.
//
// Timing is OFF unless tulpa_profile() switches it on (cpp_profile_enable()).
// A disabled scope costs one relaxed atomic load: no clock read and no lock,
// so the gradient-evaluation scope in the sampler's leapfrog is free in an
// ordinary fit. An enabled scope fires at per-iteration granularity (one add()
// per scatter / factorize / line search / gradient / cell, never per
// observation), and no scope sits inside an OpenMP worksharing loop: a scope
// is opened around a call that may run its own parallel region, never within
// one. Nothing here throws on the timed path.
//
// Phases NEST in one place only: an ENCLOSING phase (outer_grid_cell,
// nuts_warmup, nuts_sampling; see phase_is_enclosing) contains the leaf phases
// timed inside it, so its seconds are not added to theirs when shares are
// formed.
//
// Usage from R:
//   tulpa_profile(fit_expression)
// or, by hand,
//   cpp_profile_reset(); old <- cpp_profile_enable(TRUE)
//   fit <- ...
//   times <- cpp_profile_read(); cpp_profile_enable(old)
//
// Usage from a C++ instrumentation site:
//   { TULPA_PROFILE_PHASE(tulpa::PHASE_FACTORIZE); ... }
//
// Phase ordering is fixed (see PhaseIdx); profile_read() returns a named
// vector so callers don't have to memorize indices.

#ifndef TULPA_LAPLACE_PROFILE_H
#define TULPA_LAPLACE_PROFILE_H

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>

namespace tulpa {

// One slot per phase; keep this list in sync with kPhaseNames in
// laplace_profile.cpp.
enum PhaseIdx : int {
    PHASE_PATTERN_BUILD = 0,  // one-time symbolic Hessian pattern enumeration
    PHASE_PREP          = 1,  // per-cell prep (Sigma_inv, sqrt_S, ...)
    PHASE_ETA           = 2,  // per-iter compute_eta callback at iter start
    PHASE_SCATTER       = 3,  // per-iter scatter (Gauss-Newton fill of H, grad)
    PHASE_ANALYZE       = 4,  // one-time symbolic Cholesky (first iter only)
    PHASE_FACTORIZE     = 5,  // per-iter numeric Cholesky factorize
    PHASE_SOLVE         = 6,  // per-iter triangular solve
    PHASE_LINE_SEARCH   = 7,  // per-iter line_search_backtrack (incl re-eta)
    PHASE_LOG_DET       = 8,  // final-pass log_determinant
    PHASE_LOG_LIK_PRIOR = 9,  // final-pass log_lik + log_prior + center
    PHASE_HESSIAN_EXTRACT = 10, // final-pass H^-1 blocks / fixed block / Q export
    PHASE_INNER_DIAG    = 11, // final-pass skew / inner k-hat / debias / CILA probes
    PHASE_GRADIENT      = 12, // sampler log-density gradient evaluation
    PHASE_OUTER_CELL    = 13, // ENCLOSING: one nested-Laplace outer-grid cell solve
    PHASE_NUTS_WARMUP   = 14, // ENCLOSING: one NUTS warmup iteration
    PHASE_NUTS_SAMPLING = 15, // ENCLOSING: one NUTS sampling iteration
    PHASE_COUNT         = 16
};

// Whether a phase encloses other timed phases rather than being a leaf of the
// partition. Shares are formed over the leaves only (tulpa_profile()).
constexpr bool phase_is_enclosing(int idx) {
    return idx == PHASE_OUTER_CELL || idx == PHASE_NUTS_WARMUP ||
           idx == PHASE_NUTS_SAMPLING;
}

struct PhaseAccumulator {
    std::array<double, PHASE_COUNT> us{};  // microseconds per phase
    std::array<long,   PHASE_COUNT> n{};   // call counts per phase

    void reset() {
        for (int i = 0; i < PHASE_COUNT; ++i) { us[i] = 0.0; n[i] = 0; }
    }

    void add(int idx, double us_delta) {
        if (idx >= 0 && idx < PHASE_COUNT) {
            us[idx] += us_delta;
            n[idx]  += 1;
        }
    }
};

// Whether timing is on. Read relaxed on every scope entry; written only from
// the R main thread by cpp_profile_enable(), outside any parallel region.
inline std::atomic<bool>& phase_profiling_enabled() {
    static std::atomic<bool> on{false};
    return on;
}

// Process-global accumulator shared across the outer-grid worker threads, and
// the mutex that guards every add / reset / read of it.
inline std::mutex& phase_mutex() {
    static std::mutex m;
    return m;
}
inline PhaseAccumulator& global_phase_accumulator() {
    static PhaseAccumulator acc;
    return acc;
}

struct PhaseTimer {
    int idx;
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit PhaseTimer(int i)
        : idx(i),
          on(phase_profiling_enabled().load(std::memory_order_relaxed)) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
    ~PhaseTimer() {
        if (!on) return;
        auto t1 = std::chrono::steady_clock::now();
        double us = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        ) * 1e-3;
        std::lock_guard<std::mutex> guard(phase_mutex());
        global_phase_accumulator().add(idx, us);
    }
};

#define TULPA_PROFILE_CONCAT_INNER(a, b) a##b
#define TULPA_PROFILE_CONCAT(a, b) TULPA_PROFILE_CONCAT_INNER(a, b)
#define TULPA_PROFILE_PHASE(idx) \
    ::tulpa::PhaseTimer TULPA_PROFILE_CONCAT(_phase_timer_, __LINE__)(idx)

} // namespace tulpa

#endif // TULPA_LAPLACE_PROFILE_H
