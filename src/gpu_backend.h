// gpu_backend.h
// GPU acceleration for GP computations
// Supports CUDA (NVIDIA) and OpenCL (cross-platform)
//
// STATUS: the CUDA primitives (gpu_cuda.h) are compiled in by DEFAULT and
// resolved from the driver at run time, so a build needs no CUDA SDK and a run
// with no device degrades to the CPU. TULPA_DISABLE_CUDA builds the stubs
// below instead, and it is the only switch: TULPA_ENABLE_CUDA gates nothing.
// The block above the #include further down carries why that decision is made
// in exactly one place, and cuda_backend_kind() reports which was built.
//
// cuda_batched_cholesky IS dispatched to -- batch_nngp_scatter
// (gpu_nngp_laplace.h) hands it every NNGP neighbour factorization once the
// batch reaches 50. The gpu_batched_* compositions at the end of this file have
// no caller yet. OpenCL is a placeholder: the library is probed for, and
// nothing calls it.
//
// Uses RUNTIME detection - works even if user installs CUDA/OpenCL after
// installing tulpa. No recompilation needed.
//
// On Windows: looks for nvcuda.dll (CUDA) or OpenCL.dll
// On Linux: looks for libcuda.so or libOpenCL.so
// On macOS: looks for CUDA.framework or OpenCL.framework

#ifndef TULPA_GPU_BACKEND_H
#define TULPA_GPU_BACKEND_H

#include <string>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
  #define GPU_LIB_HANDLE HMODULE
  #define GPU_LOAD_LIB(name) LoadLibraryA(name)
  #define GPU_GET_PROC(lib, name) GetProcAddress(lib, name)
  #define GPU_FREE_LIB(lib) FreeLibrary(lib)
  #define CUDA_LIB_NAME "nvcuda.dll"
  #define OPENCL_LIB_NAME "OpenCL.dll"
#else
  #include <dlfcn.h>
  #define GPU_LIB_HANDLE void*
  #define GPU_LOAD_LIB(name) dlopen(name, RTLD_LAZY)
  #define GPU_GET_PROC(lib, name) dlsym(lib, name)
  #define GPU_FREE_LIB(lib) dlclose(lib)
  #ifdef __APPLE__
    #define CUDA_LIB_NAME "/usr/local/cuda/lib/libcuda.dylib"
    #define OPENCL_LIB_NAME "/System/Library/Frameworks/OpenCL.framework/OpenCL"
  #else
    #define CUDA_LIB_NAME "libcuda.so.1"
    #define OPENCL_LIB_NAME "libOpenCL.so.1"
  #endif
#endif

namespace tulpa_gpu {

// =============================================================================
// GPU Device Information
// =============================================================================

struct GPUDeviceInfo {
  std::string name;                // driver-reported device name
  size_t memory_mb;                // total device memory, MiB
  std::string compute_capability;  // CUDA only; empty when the driver withheld it
  int device_id;
};

// `available` says a backend library loaded and answered; `device_count` and
// `devices` are what that backend enumerated, and stay 0 / empty for a backend
// this package has no query for. The two are separate questions: OpenCL loads
// and enumerates nothing.
struct GPUInfo {
  bool available;
  std::string backend;  // "cuda", "opencl", or "none"
  int device_count;
  std::vector<GPUDeviceInfo> devices;
};

// =============================================================================
// Runtime GPU Detection
// =============================================================================

// Probe for a backend library. Each probe is a LoadLibrary / FreeLibrary pair
// and its answer cannot change within a process, so it is taken once and every
// later caller reads the cached bool. The initialization of a function-local
// static is thread-safe, which is what lets the probes sit in front of entries
// a worker thread can reach.
inline bool probe_library(const char* name) {
  GPU_LIB_HANDLE lib = GPU_LOAD_LIB(name);
  if (lib == nullptr) return false;
  GPU_FREE_LIB(lib);
  return true;
}

// Is the CUDA driver library present?
inline bool try_load_cuda() {
  static const bool present = probe_library(CUDA_LIB_NAME);
  return present;
}

// Is the OpenCL library present?
inline bool try_load_opencl() {
  static const bool present = probe_library(OPENCL_LIB_NAME);
  return present;
}

// Check GPU availability at runtime (no recompilation needed)
inline bool gpu_available() {
  // Try CUDA first (usually faster), then OpenCL
  static const bool available = try_load_cuda() || try_load_opencl();
  return available;
}

}  // namespace tulpa_gpu

// The CUDA implementation is compiled in by DEFAULT, and there is exactly ONE
// definition of it in the program (gcol33/tulpa#396).
//
// It used to be opt-in behind TULPA_ENABLE_CUDA, which neither Makevars ever
// defined -- so this header's includers compiled the STUBS below, while
// gpu_nngp_laplace.h included gpu_cuda.h DIRECTLY and compiled the real ones.
// Two different definitions of the same `inline` function across translation
// units is an ODR violation: the linker keeps one COMDAT and discards the rest,
// so whether CUDA ran at all was decided by link order rather than by any
// switch, and nothing in the package could report which had been built.
//
// Compiling it in does NOT require a CUDA SDK at build time or a GPU at run
// time: gpu_cuda.h resolves the driver, cuBLAS and cuSOLVER entry points
// dynamically and every entry returns false when they are absent, which is what
// makes "use CUDA if available" expressible as a default at all. Define
// TULPA_DISABLE_CUDA to build the stubs instead -- and that is now a
// whole-program choice, because this is the only place the decision is made.
#ifndef TULPA_DISABLE_CUDA
#include "gpu_cuda.h"
namespace tulpa_gpu {
// Which implementation this program was built with. Exposed so the choice is
// OBSERVABLE rather than inferred from behaviour -- a silent either/or is what
// let the ODR violation sit unnoticed.
inline const char* cuda_backend_kind() { return "cuda"; }

// Enumerate the CUDA devices into `info`, from the SAME CudaContext the
// batched entries run against, so what is reported is the device they would
// use rather than a description of the library file having loaded. Leaves
// `device_count` / `devices` untouched and returns false when the context
// cannot be brought up or reaches no device.
inline bool cuda_fill_device_info(GPUInfo& info) {
  CudaContext* ctx_ptr = nullptr;
  try {
    ctx_ptr = &CudaContext::instance();
  } catch (...) {
    return false;
  }
  if (!ctx_ptr) return false;

  CudaContext& ctx = *ctx_ptr;
  try {
    if (!ctx.initialize()) return false;
  } catch (...) {
    return false;
  }

  const int count = ctx.get_device_count();
  if (count <= 0) return false;

  info.device_count = count;
  for (int d = 0; d < count; d++) {
    GPUDeviceInfo dev;
    dev.name = ctx.get_device_name(d);
    dev.memory_mb = ctx.get_device_memory(d) / (1024 * 1024);
    dev.compute_capability = ctx.get_compute_capability(d);
    dev.device_id = d;
    info.devices.push_back(dev);
  }
  return true;
}
}  // namespace tulpa_gpu
#else
// Stub implementations when CUDA is explicitly disabled
namespace tulpa_gpu {
inline const char* cuda_backend_kind() { return "stub"; }
inline bool cuda_fill_device_info(GPUInfo& info) {
  (void)info;
  return false;  // CUDA not compiled in
}
inline bool cuda_batched_cholesky(std::vector<std::vector<double>>& matrices, int k) {
  (void)matrices; (void)k;
  return false;  // CUDA not compiled in
}
inline bool cuda_batched_trsv(
    const std::vector<std::vector<double>>& L_matrices,
    std::vector<std::vector<double>>& b_vectors,
    int k
) {
  (void)L_matrices; (void)b_vectors; (void)k;
  return false;  // CUDA not compiled in
}
inline bool cuda_batched_trsv_transpose(
    const std::vector<std::vector<double>>& L_matrices,
    std::vector<std::vector<double>>& b_vectors,
    int k
) {
  (void)L_matrices; (void)b_vectors; (void)k;
  return false;  // CUDA not compiled in
}
}  // namespace tulpa_gpu
#endif

namespace tulpa_gpu {

// Get detailed GPU info
inline GPUInfo get_gpu_info() {
  GPUInfo info;
  info.available = false;
  info.backend = "none";
  info.device_count = 0;

  // `available` answers the same question gpu_available() does -- a backend
  // library is present -- and the device fields answer the separate one of what
  // that backend enumerated. A driver that loads but reaches no device reports
  // available with a count of zero.
  if (try_load_cuda()) {
    info.available = true;
    info.backend = "cuda";
    cuda_fill_device_info(info);
    return info;
  }

  if (try_load_opencl()) {
    // The library is present and nothing in this package calls OpenCL, so no
    // device is enumerated and `devices` stays empty.
    info.available = true;
    info.backend = "opencl";
    return info;
  }

  return info;
}

// =============================================================================
// GPU-accelerated Linear Algebra
// =============================================================================

// These functions use runtime dynamic loading to call GPU libraries.
// They return false if GPU is not available or operation fails. The CUDA entry
// makes that decision for itself -- it resolves the driver through CudaContext
// and returns false when it is absent -- so a second library probe here would
// only repeat what gpu_available() already answered.
//
// There is no OpenCL implementation, so an OpenCL-only machine gets false from
// the CUDA entry and falls back to the CPU.
//
// The k x k factor buffers all three entries pass and expect are in ONE layout:
// L in the ROW-major lower triangle with the row-major upper zero, which is
// what cuda_batched_cholesky returns and what linalg_fast.h's CPU solves read.
// See cusolver_factor_to_row_major() in gpu_cuda.h.

// Batched Cholesky decomposition on GPU
// Solves many small k x k systems in parallel
// A_batch: vector of k x k matrices (row-major, flattened)
// L_batch: output L factors, in the row-major lower-triangular layout above
// Returns false if GPU unavailable (falls back to CPU in caller)
inline bool gpu_batched_cholesky(
    const std::vector<std::vector<double>>& A_batch,
    std::vector<std::vector<double>>& L_batch,
    int k
) {
  if (!gpu_available()) {
    return false;
  }

  // Copy input to output (Cholesky is done in-place)
  L_batch = A_batch;

  return cuda_batched_cholesky(L_batch, k);
}

// Batched triangular solve: L * x = b
// L_batch carries the factor in the row-major lower-triangular layout above
// Solves for x in-place (b_batch becomes x_batch)
inline bool gpu_batched_trsv(
    const std::vector<std::vector<double>>& L_batch,
    std::vector<std::vector<double>>& b_batch,  // Modified in place to hold x
    int k
) {
  if (!gpu_available()) {
    return false;
  }

  return cuda_batched_trsv(L_batch, b_batch, k);
}

// Batched transposed triangular solve: L^T * x = b
// Used for back-substitution after a forward solve with the same L factor
// L_batch carries the factor in the row-major lower-triangular layout above
// Solves for x in-place (b_batch becomes x_batch)
inline bool gpu_batched_trsv_transpose(
    const std::vector<std::vector<double>>& L_batch,
    std::vector<std::vector<double>>& b_batch,  // Modified in place to hold x
    int k
) {
  if (!gpu_available()) {
    return false;
  }

  return cuda_batched_trsv_transpose(L_batch, b_batch, k);
}

// Combined Cholesky + solve for NNGP: solve C * alpha = c
// Returns the full inverse-times-vector product alpha = C^{-1} c via the
// three-step factorisation C = L L^T, forward solve L y = c, backward
// solve L^T alpha = y. Both triangular solves run on the GPU.
//
// The factor is handed between the three steps in the row-major lower-triangular
// layout named above, which is the one thing this composition has to get right
// and the one thing no single step can check: a mismatch returns true and gives
// finite, plausible numbers. cpp_gpu_batched_cholesky_solve() exists to score
// the result against an independent solve.
//
// C_batch: k x k SPD covariance matrices (neighbor covariances)
// c_batch: k vectors (covariances to current point)
// alpha_batch: output k vectors (kriging weights = C^{-1} c)
inline bool gpu_batched_cholesky_solve(
    const std::vector<std::vector<double>>& C_batch,
    const std::vector<std::vector<double>>& c_batch,
    std::vector<std::vector<double>>& alpha_batch,
    int k
) {
  if (!gpu_available()) {
    return false;
  }

  int batch_size = (int)C_batch.size();
  if (batch_size == 0 || c_batch.size() != C_batch.size()) {
    return false;
  }

  // Step 1: Batched Cholesky C = L L^T
  std::vector<std::vector<double>> L_batch;
  if (!gpu_batched_cholesky(C_batch, L_batch, k)) {
    return false;
  }

  // Step 2: Batched forward solve L y = c (y stored in alpha_batch)
  alpha_batch = c_batch;  // Copy c to alpha (solve in place)
  if (!gpu_batched_trsv(L_batch, alpha_batch, k)) {
    return false;
  }

  // Step 3: Batched backward solve L^T alpha = y (overwrites y with C^{-1} c)
  if (!gpu_batched_trsv_transpose(L_batch, alpha_batch, k)) {
    return false;
  }

  return true;
}

}  // namespace tulpa_gpu

#endif  // TULPA_GPU_BACKEND_H
