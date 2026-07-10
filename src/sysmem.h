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

// Currently available (free + reclaimable) physical RAM in bytes, or 0 if it
// cannot be determined. This is what a memory budget must be sized against:
// installed RAM already committed to other processes is not usable, so
// budgeting against total_ram_bytes() over-provisions on a loaded machine.
// Windows: MEMORYSTATUSEX::ullAvailPhys; Linux: /proc/meminfo MemAvailable
// (kernel's own estimate), else sysconf(_SC_AVPHYS_PAGES); macOS: Mach VM
// free + inactive pages.
std::size_t available_ram_bytes();

}  // namespace tulpa

#endif  // TULPA_SYSMEM_H
