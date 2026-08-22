// hessian_pattern_guard.h
// Detection channel for Hessian writes that miss the registered sparsity
// pattern.
//
// A scatter and the pattern builder that precedes it must enumerate the same
// (row, col) set. Where they disagree the surplus writes are discarded, and what
// remains is a Hessian that is too small in one entry: finite, positive
// definite, correctly shaped, and wrong. It flows into the Newton step, the
// Laplace log-determinant, the marginal standard errors and the exact outer
// gradient without tripping any shape, finiteness or PD check, so the only
// symptom is a recovery test drifting -- the hardest signal here to attribute.
// Unequal / non-contiguous areal components and weighted-entry blocks are the
// two structures whose pattern is easiest to get wrong.
//
// Detection is a counter rather than a throw at the write because the scatter
// runs inside OpenMP parallel regions, where an exception escaping the
// structured block is std::terminate -- the constraint that also makes
// LaplaceResult::start_infeasible a flag. Drivers bracket a fit with
// HessianPatternGuard and raise once the parallel region has joined.
//
// Only a NONZERO discarded contribution counts. The scatter index caches resolve
// whole cross products up front -- every (beta_j, RE_g) pair of an arm, say --
// and legitimately hold -1 for pairs no observation touches. Those slots are
// written with a structural zero, which changes nothing whether it lands or not.
//
// PLACEMENT RULE: a HessianPatternGuard must NOT be declared in a function whose
// own body contains an `#pragma omp parallel` region. Put it in that function's
// CALLER instead. An object live on both sides of such a region, ended by a call
// that can throw, costs enough additional stack on the OpenMP worker threads
// (whose stacks are far smaller than the main thread's) to overflow them once a
// process has run enough fits. The guards over the outer grid therefore sit in
// the three callers of `run_nested_laplace_grid`
// (`run_multi_block_nested_laplace`, `run_multi_block_nested_laplace_joint`, and
// the sparse impl), which is identical coverage. A pragma inside a LAMBDA in the
// function body is fine -- that region belongs to the lambda's operator(), not
// to the frame holding the guard.
//
// The write path is sized so detection costs the scatter nothing. scatter_slot()
// stays trivially inlinable, the counter is a namespace-scope inline variable so
// no thread-safe-initialization guard is emitted at any use site, and the miss
// branch is an out-of-line cold call rather than an inlined atomic. This header
// also deliberately avoids <Rcpp.h>: it is included from headers that Rcpp and
// RcppEigen translation units pull in at different points, and it has no reason
// to perturb that order. Rcpp's export wrappers convert std::exception into an R
// error, so throwing one is equivalent to Rcpp::stop here.

#ifndef TULPA_HESSIAN_PATTERN_GUARD_H
#define TULPA_HESSIAN_PATTERN_GUARD_H

#include <atomic>
#include <stdexcept>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#  define TULPA_COLD __attribute__((noinline, cold))
#else
#  define TULPA_COLD
#endif

namespace tulpa {

// Process-wide count of discarded nonzero contributions, monotone over the
// process lifetime. Guards measure windows of it rather than resetting, so a
// driver nested inside another composes without disturbing it.
inline std::atomic<long long> hessian_pattern_drops{0};

// Out of line and cold: a miss never happens in a correct fit, and keeping the
// atomic out of the caller lets scatter_slot() stay a bare predicate.
TULPA_COLD inline void record_hessian_pattern_drop() {
    hessian_pattern_drops.fetch_add(1, std::memory_order_relaxed);
}

inline long long hessian_pattern_drop_count() {
    return hessian_pattern_drops.load(std::memory_order_seq_cst);
}

// Write `val` through a slot index resolved earlier against the pattern. A
// negative slot means the entry is absent; the contribution is discarded, and
// counted when it would have changed the matrix.
inline void scatter_slot(double* __restrict__ values, int slot, double val) {
    if (slot >= 0)       values[slot] += val;
    else if (val != 0.0) record_hessian_pattern_drop();
}

// Reports what a fit discarded between construction and check().
class HessianPatternGuard {
public:
    HessianPatternGuard() : start_(hessian_pattern_drop_count()) {}

    long long dropped() const { return hessian_pattern_drop_count() - start_; }

    // Raise when anything was discarded. Call outside every parallel region.
    void check(const char* where) const {
        const long long n = dropped();
        if (n > 0) throw_dropped(n, where);
    }

private:
    TULPA_COLD static void throw_dropped(long long n, const char* where) {
        throw std::runtime_error(
            "internal error: " + std::to_string(n) + " nonzero Hessian "
            "contribution(s) fell outside the registered sparsity pattern in " +
            std::string(where) + ". The pattern builder and the scatter must "
            "enumerate the same (row, col) set; a discarded contribution leaves "
            "the Hessian too small in that entry, which the Newton step, the "
            "log-determinant and the standard errors all inherit.");
    }

    long long start_;
};

}  // namespace tulpa

#endif  // TULPA_HESSIAN_PATTERN_GUARD_H
