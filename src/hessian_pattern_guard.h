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
// The invariant has broken twice: unequal / non-contiguous areal components
// (#241) and weighted-entry blocks (#242).
//
// Detection is a counter rather than a throw at the write because the scatter
// runs inside OpenMP parallel regions, where an Rcpp::stop escaping the
// structured block is std::terminate -- the constraint that also makes
// LaplaceResult::start_infeasible a flag. Drivers bracket a fit with
// HessianPatternGuard and raise once the parallel region has joined.
//
// Only a NONZERO discarded contribution counts. The scatter index caches resolve
// whole cross products up front -- every (beta_j, RE_g) pair of an arm, say --
// and legitimately hold -1 for pairs no observation touches. Those slots are
// written with a structural zero, which changes nothing whether it lands or not.

#ifndef TULPA_HESSIAN_PATTERN_GUARD_H
#define TULPA_HESSIAN_PATTERN_GUARD_H

#include <Rcpp.h>
#include <atomic>

namespace tulpa {

// Process-wide count of discarded nonzero contributions, monotone over the
// process lifetime. Guards measure windows of it rather than resetting, so a
// driver nested inside another composes without disturbing it.
inline std::atomic<long long>& hessian_pattern_drop_counter() {
    static std::atomic<long long> n{0};
    return n;
}

inline void record_hessian_pattern_drop() {
    hessian_pattern_drop_counter().fetch_add(1, std::memory_order_relaxed);
}

inline long long hessian_pattern_drop_count() {
    return hessian_pattern_drop_counter().load(std::memory_order_seq_cst);
}

// Write `val` through a slot index resolved earlier against the pattern. A
// negative slot means the entry is absent; the contribution is discarded, and
// counted when it would have changed the matrix. The hot path is the same single
// predicate the hand-written `if (slot >= 0)` guards used.
inline void scatter_slot(double* __restrict__ values, int slot, double val) {
    if (slot >= 0)         values[slot] += val;
    else if (val != 0.0)   record_hessian_pattern_drop();
}

// Reports what a fit discarded between construction and check().
class HessianPatternGuard {
public:
    HessianPatternGuard() : start_(hessian_pattern_drop_count()) {}

    long long dropped() const { return hessian_pattern_drop_count() - start_; }

    // Raise when anything was discarded. Call outside every parallel region.
    void check(const char* where) const {
        const long long n = dropped();
        if (n > 0) {
            Rcpp::stop(
                "internal error: %lld nonzero Hessian contribution(s) fell "
                "outside the registered sparsity pattern in %s. The pattern "
                "builder and the scatter must enumerate the same (row, col) "
                "set; a discarded contribution leaves the Hessian too small in "
                "that entry, which the Newton step, the log-determinant and the "
                "standard errors all inherit.",
                n, where);
        }
    }

private:
    long long start_;
};

}  // namespace tulpa

#endif  // TULPA_HESSIAN_PATTERN_GUARD_H
