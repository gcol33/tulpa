// sysmem.h
// Total physical RAM detection, isolated in its own translation unit so the
// platform headers (notably <windows.h>) never reach the Eigen/Rcpp-heavy
// sources that consume it.
#ifndef TULPA_SYSMEM_H
#define TULPA_SYSMEM_H

#include <cstddef>

namespace tulpa {

// Total installed physical RAM in bytes, or 0 if it cannot be determined.
std::size_t total_ram_bytes();

}  // namespace tulpa

#endif  // TULPA_SYSMEM_H
