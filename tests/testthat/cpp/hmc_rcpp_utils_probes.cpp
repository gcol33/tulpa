// Test probes over src/hmc_rcpp_utils.cpp, compiled against the engine by load_cpp_fixture().

#include <RcppEigen.h>
#include <Rcpp.h>
#include "sysmem.h"
#include "omp_threads.h"
#include <omp.h>
#include <windows.h>
#include <vector>

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
