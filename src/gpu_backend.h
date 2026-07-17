// gpu_backend.h
// GPU acceleration for GP computations
// Supports CUDA (NVIDIA) and OpenCL (cross-platform)
//
// STATUS: opt-in roadmap backend. CPU is the default and only
// active path; the CUDA primitives (gpu_cuda.h) are compiled in only under
// TULPA_ENABLE_CUDA and are not yet dispatched to by any sampler. The
// CUDA-disabled definitions below are deliberate CPU-fallback stubs (so the
// header always compiles), and OpenCL is a placeholder. Kept as roadmap surface
// for the GPU dispatch path; see gpu_cuda.h for the batched-Cholesky targets.
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
  std::string name;
  size_t memory_mb;
  std::string compute_capability;  // CUDA only
  int device_id;
};

struct GPUInfo {
  bool available;
  std::string backend;  // "cuda", "opencl", or "none"
  int device_count;
  std::vector<GPUDeviceInfo> devices;
};

// =============================================================================
// Runtime GPU Detection
// =============================================================================

// Try to load CUDA driver library at runtime
inline bool try_load_cuda() {
  GPU_LIB_HANDLE lib = GPU_LOAD_LIB(CUDA_LIB_NAME);
  if (lib != nullptr) {
    GPU_FREE_LIB(lib);
    return true;
  }
  return false;
}

// Try to load OpenCL library at runtime
inline bool try_load_opencl() {
  GPU_LIB_HANDLE lib = GPU_LOAD_LIB(OPENCL_LIB_NAME);
  if (lib != nullptr) {
    GPU_FREE_LIB(lib);
    return true;
  }
  return false;
}

// Check GPU availability at runtime (no recompilation needed)
inline bool gpu_available() {
  static int cached = -1;  // Cache result: -1 = not checked, 0 = no, 1 = yes
  if (cached >= 0) {
    return cached == 1;
  }

  // Try CUDA first (usually faster), then OpenCL
  if (try_load_cuda() || try_load_opencl()) {
    cached = 1;
    return true;
  }

  cached = 0;
  return false;
}

// Get detailed GPU info
inline GPUInfo get_gpu_info() {
  GPUInfo info;
  info.available = false;
  info.backend = "none";
  info.device_count = 0;

  // Check CUDA
  if (try_load_cuda()) {
    info.available = true;
    info.backend = "cuda";
    // Note: To get device count/info, we'd need to actually call CUDA API
    // For now, just report that CUDA is available
    info.device_count = 1;  // Assume at least 1 if library loads

    GPUDeviceInfo dev;
    dev.name = "CUDA Device (details require CUDA initialization)";
    dev.memory_mb = 0;
    dev.compute_capability = "unknown";
    dev.device_id = 0;
    info.devices.push_back(dev);
    return info;
  }

  // Check OpenCL
  if (try_load_opencl()) {
    info.available = true;
    info.backend = "opencl";
    info.device_count = 1;  // Assume at least 1 if library loads

    GPUDeviceInfo dev;
    dev.name = "OpenCL Device (details require OpenCL initialization)";
    dev.memory_mb = 0;
    dev.compute_capability = "";
    dev.device_id = 0;
    info.devices.push_back(dev);
    return info;
  }

  return info;
}

}  // namespace tulpa_gpu

// CUDA implementation is opt-in via TULPA_ENABLE_CUDA define
// This avoids potential load-time crashes on systems with partial CUDA installations
#ifdef TULPA_ENABLE_CUDA
#include "gpu_cuda.h"
#else
// Stub implementations when CUDA is disabled
namespace tulpa_gpu {
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

// =============================================================================
// GPU-accelerated Linear Algebra
// =============================================================================

// These functions use runtime dynamic loading to call GPU libraries.
// They return false if GPU is not available or operation fails.
// No recompilation needed when user installs CUDA/OpenCL.

// Batched Cholesky decomposition on GPU
// Solves many small k x k systems in parallel
// A_batch: vector of k x k matrices (row-major, flattened)
// L_batch: output L factors (lower triangular)
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

  // Try CUDA implementation
  if (try_load_cuda()) {
    return cuda_batched_cholesky(L_batch, k);
  }

  // OpenCL not yet implemented
  return false;
}

// Batched triangular solve: L * x = b
// Solves for x in-place (b_batch becomes x_batch)
inline bool gpu_batched_trsv(
    const std::vector<std::vector<double>>& L_batch,
    std::vector<std::vector<double>>& b_batch,  // Modified in place to hold x
    int k
) {
  if (!gpu_available()) {
    return false;
  }

  // Try CUDA implementation
  if (try_load_cuda()) {
    return cuda_batched_trsv(L_batch, b_batch, k);
  }

  // OpenCL not yet implemented
  return false;
}

// Batched transposed triangular solve: L^T * x = b
// Used for back-substitution after a forward solve with the same L factor
// Solves for x in-place (b_batch becomes x_batch)
inline bool gpu_batched_trsv_transpose(
    const std::vector<std::vector<double>>& L_batch,
    std::vector<std::vector<double>>& b_batch,  // Modified in place to hold x
    int k
) {
  if (!gpu_available()) {
    return false;
  }

  // Try CUDA implementation
  if (try_load_cuda()) {
    return cuda_batched_trsv_transpose(L_batch, b_batch, k);
  }

  // OpenCL not yet implemented
  return false;
}

// Combined Cholesky + solve for NNGP: solve C * alpha = c
// Returns the full inverse-times-vector product alpha = C^{-1} c via the
// three-step factorisation C = L L^T, forward solve L y = c, backward
// solve L^T alpha = y. Both triangular solves run on the GPU.
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
