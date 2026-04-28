// gpu_cuda.h
// CUDA implementation for GPU-accelerated GP computations
// Uses dynamic loading - no CUDA SDK required at compile time
//
// TODO: Wire into tulpa sampler (ported from numdenom, not yet integrated)
//
// Minimum requirements:
// - CUDA Toolkit 11.0+ (for cusolverDnDpotrfBatched)
// - NVIDIA GPU with compute capability 3.5+
// - Driver version 450.80.02+ (Linux) or 452.39+ (Windows)
//
// Enable with: #define TULPA_ENABLE_CUDA before including gpu_backend.h

#ifndef TULPA_GPU_CUDA_H
#define TULPA_GPU_CUDA_H

#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
  #include <windows.h>
  #define CUDA_LIB HMODULE
  #define CUDA_LOAD_LIB(name) LoadLibraryA(name)
  #define CUDA_GET_PROC(lib, name) GetProcAddress(lib, name)
  #define CUDA_FREE_LIB(lib) FreeLibrary(lib)
#else
  #include <dlfcn.h>
  #define CUDA_LIB void*
  #define CUDA_LOAD_LIB(name) dlopen(name, RTLD_LAZY)
  #define CUDA_GET_PROC(lib, name) dlsym(lib, name)
  #define CUDA_FREE_LIB(lib) dlclose(lib)
#endif

namespace tulpa_gpu {

// =============================================================================
// CUDA Type Definitions (matching cuda.h without requiring SDK)
// =============================================================================

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

// CUDA error codes
typedef enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_UNKNOWN = 999
} CUresult;

// cuBLAS types
typedef struct cublasContext* cublasHandle_t;
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_EXECUTION_FAILED = 13
} cublasStatus_t;

typedef enum {
  CUBLAS_FILL_MODE_LOWER = 0,
  CUBLAS_FILL_MODE_UPPER = 1
} cublasFillMode_t;

typedef enum {
  CUBLAS_SIDE_LEFT = 0,
  CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1
} cublasOperation_t;

typedef enum {
  CUBLAS_DIAG_NON_UNIT = 0,
  CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

// cuSOLVER types
typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef enum {
  CUSOLVER_STATUS_SUCCESS = 0,
  CUSOLVER_STATUS_NOT_INITIALIZED = 1,
  CUSOLVER_STATUS_ALLOC_FAILED = 2,
  CUSOLVER_STATUS_INVALID_VALUE = 3,
  CUSOLVER_STATUS_INTERNAL_ERROR = 6
} cusolverStatus_t;

// =============================================================================
// Function Pointer Types
// =============================================================================

// CUDA Driver API
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(size_t*, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuCtxSynchronize_t)(void);

// cuBLAS
typedef cublasStatus_t (*cublasCreate_t)(cublasHandle_t*);
typedef cublasStatus_t (*cublasDestroy_t)(cublasHandle_t);
typedef cublasStatus_t (*cublasDtrsmBatched_t)(
    cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int,
    const double*, const double* const*, int,
    double* const*, int, int);

// cuSOLVER
typedef cusolverStatus_t (*cusolverDnCreate_t)(cusolverDnHandle_t*);
typedef cusolverStatus_t (*cusolverDnDestroy_t)(cusolverDnHandle_t);
typedef cusolverStatus_t (*cusolverDnDpotrfBatched_t)(
    cusolverDnHandle_t, cublasFillMode_t, int,
    double**, int, int*, int);

// =============================================================================
// CUDA Context Manager (Singleton)
// =============================================================================

class CudaContext {
private:
  bool initialized_ = false;
  bool init_failed_ = false;

  // Library handles
  CUDA_LIB cuda_lib_ = nullptr;
  CUDA_LIB cublas_lib_ = nullptr;
  CUDA_LIB cusolver_lib_ = nullptr;

  // CUDA handles
  CUcontext context_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
  cusolverDnHandle_t cusolver_handle_ = nullptr;

  // Function pointers - CUDA Driver
  cuInit_t cuInit_ = nullptr;
  cuDeviceGetCount_t cuDeviceGetCount_ = nullptr;
  cuDeviceGet_t cuDeviceGet_ = nullptr;
  cuDeviceGetName_t cuDeviceGetName_ = nullptr;
  cuDeviceTotalMem_t cuDeviceTotalMem_ = nullptr;
  cuCtxCreate_t cuCtxCreate_ = nullptr;
  cuCtxDestroy_t cuCtxDestroy_ = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent_ = nullptr;
  cuMemAlloc_t cuMemAlloc_ = nullptr;
  cuMemFree_t cuMemFree_ = nullptr;
  cuMemcpyHtoD_t cuMemcpyHtoD_ = nullptr;
  cuMemcpyDtoH_t cuMemcpyDtoH_ = nullptr;
  cuCtxSynchronize_t cuCtxSynchronize_ = nullptr;

  // Function pointers - cuBLAS
  cublasCreate_t cublasCreate_ = nullptr;
  cublasDestroy_t cublasDestroy_ = nullptr;
  cublasDtrsmBatched_t cublasDtrsmBatched_ = nullptr;

  // Function pointers - cuSOLVER
  cusolverDnCreate_t cusolverDnCreate_ = nullptr;
  cusolverDnDestroy_t cusolverDnDestroy_ = nullptr;
  cusolverDnDpotrfBatched_t cusolverDnDpotrfBatched_ = nullptr;

  CudaContext() = default;

  bool load_libraries() {
    #ifdef _WIN32
      cuda_lib_ = CUDA_LOAD_LIB("nvcuda.dll");
      cublas_lib_ = CUDA_LOAD_LIB("cublas64_12.dll");
      if (!cublas_lib_) cublas_lib_ = CUDA_LOAD_LIB("cublas64_11.dll");
      cusolver_lib_ = CUDA_LOAD_LIB("cusolver64_12.dll");
      if (!cusolver_lib_) cusolver_lib_ = CUDA_LOAD_LIB("cusolver64_11.dll");
    #else
      cuda_lib_ = CUDA_LOAD_LIB("libcuda.so.1");
      cublas_lib_ = CUDA_LOAD_LIB("libcublas.so.12");
      if (!cublas_lib_) cublas_lib_ = CUDA_LOAD_LIB("libcublas.so.11");
      cusolver_lib_ = CUDA_LOAD_LIB("libcusolver.so.12");
      if (!cusolver_lib_) cusolver_lib_ = CUDA_LOAD_LIB("libcusolver.so.11");
    #endif

    return cuda_lib_ != nullptr;
  }

  bool load_functions() {
    if (!cuda_lib_) return false;

    // Load CUDA driver functions
    cuInit_ = (cuInit_t)CUDA_GET_PROC(cuda_lib_, "cuInit");
    cuDeviceGetCount_ = (cuDeviceGetCount_t)CUDA_GET_PROC(cuda_lib_, "cuDeviceGetCount");
    cuDeviceGet_ = (cuDeviceGet_t)CUDA_GET_PROC(cuda_lib_, "cuDeviceGet");
    cuDeviceGetName_ = (cuDeviceGetName_t)CUDA_GET_PROC(cuda_lib_, "cuDeviceGetName");
    cuDeviceTotalMem_ = (cuDeviceTotalMem_t)CUDA_GET_PROC(cuda_lib_, "cuDeviceTotalMem_v2");
    cuCtxCreate_ = (cuCtxCreate_t)CUDA_GET_PROC(cuda_lib_, "cuCtxCreate_v2");
    cuCtxDestroy_ = (cuCtxDestroy_t)CUDA_GET_PROC(cuda_lib_, "cuCtxDestroy_v2");
    cuCtxSetCurrent_ = (cuCtxSetCurrent_t)CUDA_GET_PROC(cuda_lib_, "cuCtxSetCurrent");
    cuMemAlloc_ = (cuMemAlloc_t)CUDA_GET_PROC(cuda_lib_, "cuMemAlloc_v2");
    cuMemFree_ = (cuMemFree_t)CUDA_GET_PROC(cuda_lib_, "cuMemFree_v2");
    cuMemcpyHtoD_ = (cuMemcpyHtoD_t)CUDA_GET_PROC(cuda_lib_, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_ = (cuMemcpyDtoH_t)CUDA_GET_PROC(cuda_lib_, "cuMemcpyDtoH_v2");
    cuCtxSynchronize_ = (cuCtxSynchronize_t)CUDA_GET_PROC(cuda_lib_, "cuCtxSynchronize");

    if (!cuInit_ || !cuDeviceGetCount_ || !cuMemAlloc_ || !cuMemFree_ ||
        !cuMemcpyHtoD_ || !cuMemcpyDtoH_ || !cuCtxCreate_) {
      return false;
    }

    // Load cuBLAS functions (optional - batched trsm)
    if (cublas_lib_) {
      cublasCreate_ = (cublasCreate_t)CUDA_GET_PROC(cublas_lib_, "cublasCreate_v2");
      cublasDestroy_ = (cublasDestroy_t)CUDA_GET_PROC(cublas_lib_, "cublasDestroy_v2");
      cublasDtrsmBatched_ = (cublasDtrsmBatched_t)CUDA_GET_PROC(cublas_lib_, "cublasDtrsmBatched");
    }

    // Load cuSOLVER functions (optional - batched potrf)
    if (cusolver_lib_) {
      cusolverDnCreate_ = (cusolverDnCreate_t)CUDA_GET_PROC(cusolver_lib_, "cusolverDnCreate");
      cusolverDnDestroy_ = (cusolverDnDestroy_t)CUDA_GET_PROC(cusolver_lib_, "cusolverDnDestroy");
      cusolverDnDpotrfBatched_ = (cusolverDnDpotrfBatched_t)CUDA_GET_PROC(cusolver_lib_, "cusolverDnDpotrfBatched");
    }

    return true;
  }

public:
  static CudaContext& instance() {
    static CudaContext ctx;
    return ctx;
  }

  bool initialize() {
    if (initialized_) return true;
    if (init_failed_) return false;

    if (!load_libraries() || !load_functions()) {
      init_failed_ = true;
      return false;
    }

    // Initialize CUDA
    if (cuInit_(0) != CUDA_SUCCESS) {
      init_failed_ = true;
      return false;
    }

    // Check device count
    int device_count = 0;
    if (cuDeviceGetCount_(&device_count) != CUDA_SUCCESS || device_count == 0) {
      init_failed_ = true;
      return false;
    }

    // Create context on device 0
    CUdevice device;
    if (cuDeviceGet_(&device, 0) != CUDA_SUCCESS) {
      init_failed_ = true;
      return false;
    }

    if (cuCtxCreate_(&context_, 0, device) != CUDA_SUCCESS) {
      init_failed_ = true;
      return false;
    }

    // Create cuBLAS handle
    if (cublasCreate_ && cublasCreate_(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      cublas_handle_ = nullptr;
    }

    // Create cuSOLVER handle
    if (cusolverDnCreate_ && cusolverDnCreate_(&cusolver_handle_) != CUSOLVER_STATUS_SUCCESS) {
      cusolver_handle_ = nullptr;
    }

    initialized_ = true;
    return true;
  }

  ~CudaContext() {
    if (cusolver_handle_ && cusolverDnDestroy_) {
      cusolverDnDestroy_(cusolver_handle_);
    }
    if (cublas_handle_ && cublasDestroy_) {
      cublasDestroy_(cublas_handle_);
    }
    if (context_ && cuCtxDestroy_) {
      cuCtxDestroy_(context_);
    }
    if (cuda_lib_) CUDA_FREE_LIB(cuda_lib_);
    if (cublas_lib_) CUDA_FREE_LIB(cublas_lib_);
    if (cusolver_lib_) CUDA_FREE_LIB(cusolver_lib_);
  }

  bool is_initialized() const { return initialized_; }
  bool has_cublas() const { return cublas_handle_ != nullptr; }
  bool has_cusolver() const { return cusolver_handle_ != nullptr; }

  // Memory management
  CUdeviceptr alloc(size_t bytes) {
    if (!initialized_) return 0;
    CUdeviceptr ptr = 0;
    if (cuMemAlloc_(&ptr, bytes) != CUDA_SUCCESS) {
      return 0;
    }
    return ptr;
  }

  void free(CUdeviceptr ptr) {
    if (initialized_ && ptr) {
      cuMemFree_(ptr);
    }
  }

  bool copy_to_device(CUdeviceptr dst, const void* src, size_t bytes) {
    if (!initialized_) return false;
    return cuMemcpyHtoD_(dst, src, bytes) == CUDA_SUCCESS;
  }

  bool copy_to_host(void* dst, CUdeviceptr src, size_t bytes) {
    if (!initialized_) return false;
    return cuMemcpyDtoH_(dst, src, bytes) == CUDA_SUCCESS;
  }

  void synchronize() {
    if (initialized_ && cuCtxSynchronize_) {
      cuCtxSynchronize_();
    }
  }

  // Batched Cholesky decomposition
  bool batched_cholesky(double** d_A, int n, int batch_size, int* d_info) {
    if (!has_cusolver() || !cusolverDnDpotrfBatched_) {
      return false;
    }

    cusolverStatus_t status = cusolverDnDpotrfBatched_(
      cusolver_handle_,
      CUBLAS_FILL_MODE_LOWER,
      n,
      d_A,
      n,
      d_info,
      batch_size
    );

    synchronize();
    return status == CUSOLVER_STATUS_SUCCESS;
  }

  // Batched triangular solve: op(L) * X = B
  // op = CUBLAS_OP_N for forward solve (L * X = B)
  // op = CUBLAS_OP_T for backward solve (L^T * X = B)
  bool batched_trsm(double** d_L, double** d_B, int n, int nrhs, int batch_size,
                    cublasOperation_t op = CUBLAS_OP_N) {
    if (!has_cublas() || !cublasDtrsmBatched_) {
      return false;
    }

    const double alpha = 1.0;
    cublasStatus_t status = cublasDtrsmBatched_(
      cublas_handle_,
      CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER,
      op,
      CUBLAS_DIAG_NON_UNIT,
      n, nrhs,
      &alpha,
      (const double* const*)d_L, n,
      d_B, n,
      batch_size
    );

    synchronize();
    return status == CUBLAS_STATUS_SUCCESS;
  }

  // Get device info
  int get_device_count() {
    if (!initialized_ || !cuDeviceGetCount_) return 0;
    int count = 0;
    cuDeviceGetCount_(&count);
    return count;
  }

  std::string get_device_name(int device_id = 0) {
    if (!initialized_ || !cuDeviceGetName_) return "";
    CUdevice device;
    if (cuDeviceGet_(&device, device_id) != CUDA_SUCCESS) return "";
    char name[256] = {0};
    if (cuDeviceGetName_(name, 256, device) != CUDA_SUCCESS) return "";
    return std::string(name);
  }

  size_t get_device_memory(int device_id = 0) {
    if (!initialized_ || !cuDeviceTotalMem_) return 0;
    CUdevice device;
    if (cuDeviceGet_(&device, device_id) != CUDA_SUCCESS) return 0;
    size_t bytes = 0;
    if (cuDeviceTotalMem_(&bytes, device) != CUDA_SUCCESS) return 0;
    return bytes;
  }
};

// =============================================================================
// High-level GPU Operations for NNGP
// =============================================================================

// Batched Cholesky for NNGP neighbor covariance matrices
// Each matrix is k x k, we have batch_size matrices
// Returns L factors in-place (lower triangular)
inline bool cuda_batched_cholesky(
    std::vector<std::vector<double>>& matrices,  // batch_size x (k*k)
    int k
) {
  // Try to get CUDA context - return false on any failure
  CudaContext* ctx_ptr = nullptr;
  try {
    ctx_ptr = &CudaContext::instance();
  } catch (...) {
    return false;  // Singleton construction failed
  }
  if (!ctx_ptr) return false;

  CudaContext& ctx = *ctx_ptr;

  // Try initialization - may fail if libraries not available
  try {
    if (!ctx.initialize() || !ctx.has_cusolver()) {
      return false;
    }
  } catch (...) {
    return false;  // Initialization threw an exception
  }

  int batch_size = (int)matrices.size();
  if (batch_size == 0 || k <= 0) return false;

  size_t matrix_bytes = k * k * sizeof(double);

  // Allocate device memory for all matrices
  std::vector<CUdeviceptr> d_matrices(batch_size);
  std::vector<double*> d_matrix_ptrs(batch_size);

  for (int i = 0; i < batch_size; i++) {
    d_matrices[i] = ctx.alloc(matrix_bytes);
    if (!d_matrices[i]) {
      // Cleanup on failure
      for (int j = 0; j < i; j++) ctx.free(d_matrices[j]);
      return false;
    }
    d_matrix_ptrs[i] = (double*)d_matrices[i];

    // Copy matrix to device
    if (!ctx.copy_to_device(d_matrices[i], matrices[i].data(), matrix_bytes)) {
      for (int j = 0; j <= i; j++) ctx.free(d_matrices[j]);
      return false;
    }
  }

  // Allocate pointer array on device
  CUdeviceptr d_ptrs = ctx.alloc(batch_size * sizeof(double*));
  if (!d_ptrs) {
    for (int i = 0; i < batch_size; i++) ctx.free(d_matrices[i]);
    return false;
  }
  ctx.copy_to_device(d_ptrs, d_matrix_ptrs.data(), batch_size * sizeof(double*));

  // Allocate info array
  CUdeviceptr d_info = ctx.alloc(batch_size * sizeof(int));
  if (!d_info) {
    ctx.free(d_ptrs);
    for (int i = 0; i < batch_size; i++) ctx.free(d_matrices[i]);
    return false;
  }

  // Run batched Cholesky
  bool success = ctx.batched_cholesky((double**)d_ptrs, k, batch_size, (int*)d_info);

  if (success) {
    // Copy results back
    for (int i = 0; i < batch_size; i++) {
      ctx.copy_to_host(matrices[i].data(), d_matrices[i], matrix_bytes);
    }
  }

  // Cleanup
  ctx.free(d_info);
  ctx.free(d_ptrs);
  for (int i = 0; i < batch_size; i++) {
    ctx.free(d_matrices[i]);
  }

  return success;
}

// Batched triangular solve for NNGP: L * x = b
// L is lower triangular k x k, b/x are vectors of length k
inline bool cuda_batched_trsv(
    const std::vector<std::vector<double>>& L_matrices,  // batch_size x (k*k)
    std::vector<std::vector<double>>& b_vectors,         // batch_size x k (modified in place)
    int k
) {
  // Try to get CUDA context - return false on any failure
  CudaContext* ctx_ptr = nullptr;
  try {
    ctx_ptr = &CudaContext::instance();
  } catch (...) {
    return false;
  }
  if (!ctx_ptr) return false;

  CudaContext& ctx = *ctx_ptr;

  try {
    if (!ctx.initialize() || !ctx.has_cublas()) {
      return false;
    }
  } catch (...) {
    return false;
  }

  int batch_size = (int)L_matrices.size();
  if (batch_size == 0 || k <= 0 || b_vectors.size() != L_matrices.size()) {
    return false;
  }

  size_t matrix_bytes = k * k * sizeof(double);
  size_t vector_bytes = k * sizeof(double);

  // Allocate device memory
  std::vector<CUdeviceptr> d_L(batch_size), d_b(batch_size);
  std::vector<double*> d_L_ptrs(batch_size), d_b_ptrs(batch_size);

  for (int i = 0; i < batch_size; i++) {
    d_L[i] = ctx.alloc(matrix_bytes);
    d_b[i] = ctx.alloc(vector_bytes);
    if (!d_L[i] || !d_b[i]) {
      for (int j = 0; j <= i; j++) {
        if (d_L[j]) ctx.free(d_L[j]);
        if (d_b[j]) ctx.free(d_b[j]);
      }
      return false;
    }
    d_L_ptrs[i] = (double*)d_L[i];
    d_b_ptrs[i] = (double*)d_b[i];

    ctx.copy_to_device(d_L[i], L_matrices[i].data(), matrix_bytes);
    ctx.copy_to_device(d_b[i], b_vectors[i].data(), vector_bytes);
  }

  // Allocate pointer arrays
  CUdeviceptr d_L_ptr_array = ctx.alloc(batch_size * sizeof(double*));
  CUdeviceptr d_b_ptr_array = ctx.alloc(batch_size * sizeof(double*));
  ctx.copy_to_device(d_L_ptr_array, d_L_ptrs.data(), batch_size * sizeof(double*));
  ctx.copy_to_device(d_b_ptr_array, d_b_ptrs.data(), batch_size * sizeof(double*));

  // Run batched trsm (treating vectors as n x 1 matrices)
  bool success = ctx.batched_trsm((double**)d_L_ptr_array, (double**)d_b_ptr_array, k, 1, batch_size);

  if (success) {
    // Copy results back
    for (int i = 0; i < batch_size; i++) {
      ctx.copy_to_host(b_vectors[i].data(), d_b[i], vector_bytes);
    }
  }

  // Cleanup
  ctx.free(d_L_ptr_array);
  ctx.free(d_b_ptr_array);
  for (int i = 0; i < batch_size; i++) {
    ctx.free(d_L[i]);
    ctx.free(d_b[i]);
  }

  return success;
}

// Batched transposed triangular solve for NNGP: L^T * x = b
// L is lower triangular k x k (factor from Cholesky), b/x are vectors of length k
// Uses cublasDtrsmBatched with CUBLAS_OP_T to perform back-substitution
inline bool cuda_batched_trsv_transpose(
    const std::vector<std::vector<double>>& L_matrices,  // batch_size x (k*k)
    std::vector<std::vector<double>>& b_vectors,         // batch_size x k (modified in place)
    int k
) {
  // Try to get CUDA context - return false on any failure
  CudaContext* ctx_ptr = nullptr;
  try {
    ctx_ptr = &CudaContext::instance();
  } catch (...) {
    return false;
  }
  if (!ctx_ptr) return false;

  CudaContext& ctx = *ctx_ptr;

  try {
    if (!ctx.initialize() || !ctx.has_cublas()) {
      return false;
    }
  } catch (...) {
    return false;
  }

  int batch_size = (int)L_matrices.size();
  if (batch_size == 0 || k <= 0 || b_vectors.size() != L_matrices.size()) {
    return false;
  }

  size_t matrix_bytes = k * k * sizeof(double);
  size_t vector_bytes = k * sizeof(double);

  // Allocate device memory
  std::vector<CUdeviceptr> d_L(batch_size), d_b(batch_size);
  std::vector<double*> d_L_ptrs(batch_size), d_b_ptrs(batch_size);

  for (int i = 0; i < batch_size; i++) {
    d_L[i] = ctx.alloc(matrix_bytes);
    d_b[i] = ctx.alloc(vector_bytes);
    if (!d_L[i] || !d_b[i]) {
      for (int j = 0; j <= i; j++) {
        if (d_L[j]) ctx.free(d_L[j]);
        if (d_b[j]) ctx.free(d_b[j]);
      }
      return false;
    }
    d_L_ptrs[i] = (double*)d_L[i];
    d_b_ptrs[i] = (double*)d_b[i];

    ctx.copy_to_device(d_L[i], L_matrices[i].data(), matrix_bytes);
    ctx.copy_to_device(d_b[i], b_vectors[i].data(), vector_bytes);
  }

  // Allocate pointer arrays
  CUdeviceptr d_L_ptr_array = ctx.alloc(batch_size * sizeof(double*));
  CUdeviceptr d_b_ptr_array = ctx.alloc(batch_size * sizeof(double*));
  ctx.copy_to_device(d_L_ptr_array, d_L_ptrs.data(), batch_size * sizeof(double*));
  ctx.copy_to_device(d_b_ptr_array, d_b_ptrs.data(), batch_size * sizeof(double*));

  // Run batched trsm with CUBLAS_OP_T (treating vectors as n x 1 matrices)
  bool success = ctx.batched_trsm(
    (double**)d_L_ptr_array, (double**)d_b_ptr_array,
    k, 1, batch_size, CUBLAS_OP_T
  );

  if (success) {
    // Copy results back
    for (int i = 0; i < batch_size; i++) {
      ctx.copy_to_host(b_vectors[i].data(), d_b[i], vector_bytes);
    }
  }

  // Cleanup
  ctx.free(d_L_ptr_array);
  ctx.free(d_b_ptr_array);
  for (int i = 0; i < batch_size; i++) {
    ctx.free(d_L[i]);
    ctx.free(d_b[i]);
  }

  return success;
}

}  // namespace tulpa_gpu

#endif  // TULPA_GPU_CUDA_H
