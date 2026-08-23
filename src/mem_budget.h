// mem_budget.h
// Memory-budget arithmetic for the outer-grid thread clamp. It lives here
// rather than in the nested-Laplace grid drivers so the "budget against free
// RAM, not installed" rule has one definition and is testable independent of a
// running fit.
#ifndef TULPA_MEM_BUDGET_H
#define TULPA_MEM_BUDGET_H

#include <algorithm>
#include <cstddef>

namespace tulpa {

// Fraction of currently-free RAM handed to the replicated per-outer-thread pool.
// The remainder is left for the grid results accumulating during the solve, for
// CHOLMOD fill-in beyond the per-thread factor estimate, and for OS headroom.
constexpr double kOuterThreadRamFraction = 0.6;

// Bytes the replicated per-outer-thread state may occupy, given the detected
// available (free + reclaimable) and total installed RAM in bytes.
//
// The fraction applies to AVAILABLE rather than installed RAM, because a
// loaded machine can have most of what is installed already committed, and a
// budget taken from the installed figure over-provisions the thread pool into
// swap. The fallbacks (half of total, then a fixed 2 GB) cover platforms and
// conditions where the availability query returns 0.
inline std::size_t outer_thread_mem_budget(std::size_t avail_bytes,
                                           std::size_t total_bytes) {
  if (avail_bytes > 0) {
    return static_cast<std::size_t>(kOuterThreadRamFraction *
                                    static_cast<double>(avail_bytes));
  }
  if (total_bytes > 0) {
    return total_bytes / 2;
  }
  return static_cast<std::size_t>(2) * 1024 * 1024 * 1024;  // 2 GB
}

// Maximum number of concurrent outer builders whose per-thread working set fits
// the budget. Always at least 1: a single cell runs best-effort even when its
// working set exceeds the budget (the caller warns in that floor case).
inline int outer_thread_cap(std::size_t budget_bytes, std::size_t per_thread_bytes) {
  if (per_thread_bytes == 0) return 1;
  return static_cast<int>(std::max<std::size_t>(1, budget_bytes / per_thread_bytes));
}

}  // namespace tulpa

#endif  // TULPA_MEM_BUDGET_H
