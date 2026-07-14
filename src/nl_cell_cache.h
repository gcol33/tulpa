// nl_cell_cache.h
// Per-cell slot cache for LatentBlock state written by prep(k_grid) and read
// by the same cell's solve (basis_eval / dense_basis_batch / add_prior /
// log_prior).
//
// The joint nested-Laplace driver runs outer grid cells concurrently across
// n_threads_outer, and the coupled-cell scatter additionally spawns OpenMP
// tasks that idle team threads steal. A single shared buffer races (one
// cell's prep clobbers another cell's in-flight read), and a thread-id-keyed
// slot breaks under task stealing (the reading thread need not be the thread
// that ran prep). The cache is therefore keyed by the CELL:
//
//   * prep claims the calling thread's slot, fills it, and publishes it
//     under the cell id (release store);
//   * readers scan the (<= max threads) slots for their cell id (acquire
//     load) — matching any slot is correct because a given k_grid row of
//     theta_grid always produces identical data.
//
// Lifetime invariant: a cell's readers all run between its prep and the
// owning thread's next claim — the outer loop processes one cell per thread
// at a time, and the scatter taskgroup joins before the cell's solve
// returns — so a published slot outlives every reader of its cell.
//
// Slot payloads are allocated lazily on first claim by a thread, so memory
// scales with the parallelism actually used, not omp_get_max_threads().

#ifndef TULPA_NL_CELL_CACHE_H
#define TULPA_NL_CELL_CACHE_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

inline int nl_cell_cache_slots() {
#ifdef _OPENMP
    int n = omp_get_max_threads();
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

// Slot index of the calling thread, bounds-guarded for oversubscription.
inline int nl_cell_cache_self(int n_slots) {
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
#endif
    return (t >= 0 && t < n_slots) ? t : 0;
}

template <typename T>
class NlCellCache {
public:
    // `init` runs once per slot right after its payload is default-
    // constructed in place (payloads may hold non-copyable members, e.g.
    // CHOLMOD solver state); pass nullptr when default construction is
    // enough.
    explicit NlCellCache(std::function<void(T&)> init = nullptr,
                         int n_slots = nl_cell_cache_slots())
        : init_(std::move(init)),
          slots_(static_cast<std::size_t>(n_slots > 0 ? n_slots : 1)) {}

    NlCellCache(const NlCellCache&) = delete;
    NlCellCache& operator=(const NlCellCache&) = delete;

    // Claim the calling thread's slot for filling. Retracts the slot's old
    // cell id first so a scanning reader can never match a half-written
    // payload; the payload is constructed lazily on the thread's first claim.
    T& claim() {
        Slot& s = self_slot();
        s.cell.store(-1, std::memory_order_release);
        if (!s.data) {
            s.data.reset(new T());
            if (init_) init_(*s.data);
        }
        return *s.data;
    }

    // Publish the claimed slot under k_grid; pairs with the acquire in find().
    void publish(int k_grid) {
        self_slot().cell.store(k_grid, std::memory_order_release);
    }

    // Payload for cell k_grid, whichever slot holds it. Throws if no slot
    // does — that means a reader ran without a preceding prep for its cell,
    // which is a driver sequencing bug, not a user error.
    const T& find(int k_grid) const {
        for (const Slot& s : slots_) {
            if (s.cell.load(std::memory_order_acquire) == k_grid && s.data) {
                return *s.data;
            }
        }
        throw std::runtime_error(
            "internal error: nested-Laplace block cache read before prep for "
            "this grid cell");
    }

private:
    struct Slot {
        std::atomic<int>   cell{-1};
        std::unique_ptr<T> data;
    };

    Slot& self_slot() {
        return slots_[nl_cell_cache_self(static_cast<int>(slots_.size()))];
    }

    std::function<void(T&)> init_;
    std::vector<Slot>       slots_;
};

} // namespace tulpa

#endif // TULPA_NL_CELL_CACHE_H
