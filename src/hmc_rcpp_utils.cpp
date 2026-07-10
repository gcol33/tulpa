// hmc_rcpp_utils.cpp
// Small Rcpp-facing HMC utility exports that do not depend on sampler internals.

#include <Rcpp.h>
#include "sysmem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
// _WIN32_WINNT (Windows 10) is set package-wide in Makevars.win so the PCH and
// every TU expose PROCESSOR_RELATIONSHIP::EfficiencyClass. WIN32_LEAN_AND_MEAN
// / NOMINMAX keep windows.h from dragging in winsock and the min/max macros
// that clash with std:: and Eigen.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#endif

// [[Rcpp::export]]
int cpp_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// Total installed physical RAM in bytes, or 0 when it cannot be determined.
// Returned as a double: a byte count can exceed the R integer range. The
// memory-budget clamp in the nested-Laplace grid drivers consumes the C++
// total_ram_bytes() / available_ram_bytes() directly; these wrappers exist so
// the queries are observable from R (tests, diagnostics).
// [[Rcpp::export]]
double cpp_total_ram_bytes() {
  return static_cast<double>(tulpa::total_ram_bytes());
}

// Currently available (free + reclaimable) physical RAM in bytes, or 0 when it
// cannot be determined. This, not the installed total, is what the outer-grid
// thread budget is sized against so a loaded machine is not over-provisioned.
// [[Rcpp::export]]
double cpp_available_ram_bytes() {
  return static_cast<double>(tulpa::available_ram_bytes());
}

// Number of physical performance cores, or 0 when the topology cannot be
// resolved. On Intel hybrid CPUs the performance ("P") cores carry the highest
// EfficiencyClass; counting only the top class excludes the efficiency ("E")
// cores that the per-observation inner OpenMP loops oversubscribe. On a
// non-hybrid CPU every core is class 0, so this returns the physical-core
// count (the natural inner-thread cap, below the SMT-thread count). Returns 0
// off Windows, where the caller falls back to the requested thread count.
// [[Rcpp::export]]
int cpp_performance_core_count() {
#ifdef _WIN32
  static int cached = -1;
  if (cached >= 0) return cached;

  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len == 0) { cached = 0; return 0; }

  std::vector<char> buf(len);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
          &len)) {
    cached = 0;
    return 0;
  }

  // PROCESSOR_RELATIONSHIP is { BYTE Flags; BYTE EfficiencyClass; BYTE
  // Reserved[20]; ... }. Rtools' mingw winnt.h predates the named
  // EfficiencyClass field and lumps it into Reserved[21], so read offset 1
  // (Reserved[0]) -- the byte the OS fills with the efficiency class.
  auto efficiency_class = [](PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info) -> BYTE {
    return info->Processor.Reserved[0];
  };

  char* const end = buf.data() + len;
  BYTE max_eff = 0;
  for (char* p = buf.data(); p < end; ) {
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
    if (info->Relationship == RelationProcessorCore &&
        efficiency_class(info) > max_eff) {
      max_eff = efficiency_class(info);
    }
    p += info->Size;
  }

  int count = 0;
  for (char* p = buf.data(); p < end; ) {
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
    if (info->Relationship == RelationProcessorCore &&
        efficiency_class(info) == max_eff) {
      count++;
    }
    p += info->Size;
  }

  cached = count;
  return count;
#else
  return 0;
#endif
}
