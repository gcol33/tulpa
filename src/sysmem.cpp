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
  #include <mach/mach.h>
  #include <mach/mach_host.h>
  #include <mach/vm_statistics.h>
#else
  #include <unistd.h>
  #include <cstdio>
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

std::size_t available_ram_bytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return static_cast<std::size_t>(status.ullAvailPhys);
  }
  return 0;
#elif defined(__APPLE__)
  mach_port_t host = mach_host_self();
  vm_size_t page_size = 0;
  if (host_page_size(host, &page_size) != KERN_SUCCESS) {
    return 0;
  }
  vm_statistics64_data_t vm;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(host, HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm), &count)
      != KERN_SUCCESS) {
    return 0;
  }
  // Free plus inactive (file-backed, reclaimable under pressure) pages are the
  // memory a new allocation can grow into without swapping.
  const uint64_t pages =
      static_cast<uint64_t>(vm.free_count) + static_cast<uint64_t>(vm.inactive_count);
  return static_cast<std::size_t>(pages * static_cast<uint64_t>(page_size));
#else
  // Prefer the kernel's own MemAvailable estimate (accounts for reclaimable
  // page cache); fall back to the free-page count.
  if (FILE* f = std::fopen("/proc/meminfo", "r")) {
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
        found = true;
        break;
      }
    }
    std::fclose(f);
    if (found) {
      return static_cast<std::size_t>(kb) * static_cast<std::size_t>(1024);
    }
  }
  const long avail_pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (avail_pages > 0 && page_size > 0) {
    return static_cast<std::size_t>(avail_pages) * static_cast<std::size_t>(page_size);
  }
  return 0;
#endif
}

}  // namespace tulpa
