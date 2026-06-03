// sysmem.cpp
// Platform-specific total-RAM query. Kept in a standalone translation unit so
// <windows.h> (with its min/max and ERROR macros) stays out of the Eigen-using
// sources that only need the byte count.
#include "sysmem.h"

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#elif defined(__APPLE__)
  #include <cstdint>
  #include <sys/sysctl.h>
  #include <sys/types.h>
#else
  #include <unistd.h>
#endif

namespace tulpa {

std::size_t total_ram_bytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return static_cast<std::size_t>(status.ullTotalPhys);
  }
  return 0;
#elif defined(__APPLE__)
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  uint64_t bytes = 0;
  std::size_t len = sizeof(bytes);
  if (sysctl(mib, 2, &bytes, &len, nullptr, 0) == 0) {
    return static_cast<std::size_t>(bytes);
  }
  return 0;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
  }
  return 0;
#endif
}

}  // namespace tulpa
