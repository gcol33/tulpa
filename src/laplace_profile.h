// laplace_profile.h
// Lightweight thread-local phase accumulator for the sparse Laplace path.
// Always-on but zero-overhead when not reset (the timer scopes always add
// to the same buffer; reading is what makes the data meaningful).
//
// Usage from R:
//   cpp_profile_reset()
//   fit <- tulpa_nested_laplace_joint(...)
//   times <- cpp_profile_read()  # named numeric vector, microseconds
//
// Usage from C++ instrumentation site:
//   { TULPA_PROFILE_PHASE(tulpa::PHASE_FACTORIZE); ... }
//
// Phase ordering is fixed (see PhaseIdx); profile_read() returns a named
// vector so callers don't have to memorize indices.

#ifndef TULPA_LAPLACE_PROFILE_H
#define TULPA_LAPLACE_PROFILE_H

#include <array>
#include <chrono>
#include <cstddef>

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
    PHASE_LINE_SEARCH   = 7,  // per-iter step_halving_update (incl re-eta)
    PHASE_LOG_DET       = 8,  // final-pass log_determinant
    PHASE_LOG_LIK_PRIOR = 9,  // final-pass log_lik + log_prior + center
    PHASE_COUNT         = 10
};

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

// Per-thread accumulator. Outer-grid parallelism (when added) would
// accumulate per-thread separately; the R-side reader currently only
// inspects the calling thread, which is fine for the single-thread
// profile sweeps Stage 2 needs.
inline PhaseAccumulator& global_phase_accumulator() {
    static thread_local PhaseAccumulator acc;
    return acc;
}

struct PhaseTimer {
    int idx;
    std::chrono::steady_clock::time_point t0;
    explicit PhaseTimer(int i)
        : idx(i), t0(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() {
        auto t1 = std::chrono::steady_clock::now();
        double us = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        ) * 1e-3;
        global_phase_accumulator().add(idx, us);
    }
};

#define TULPA_PROFILE_CONCAT_INNER(a, b) a##b
#define TULPA_PROFILE_CONCAT(a, b) TULPA_PROFILE_CONCAT_INNER(a, b)
#define TULPA_PROFILE_PHASE(idx) \
    ::tulpa::PhaseTimer TULPA_PROFILE_CONCAT(_phase_timer_, __LINE__)(idx)

} // namespace tulpa

#endif // TULPA_LAPLACE_PROFILE_H
