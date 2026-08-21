// gpu_cuda.h
// CUDA implementation for GPU-accelerated GP computations
// Uses dynamic loading - no CUDA SDK required at compile time
//
// STATUS: production, not roadmap. cuda_batched_cholesky IS dispatched --
// batch_nngp_scatter (gpu_nngp_laplace.h) hands it every NNGP neighbour
// factorization once the batch reaches 50. This header is reached through
// gpu_backend.h, which owns the single CUDA-or-stub decision in the program.
//
// cuda_batched_trsv / cuda_batched_trsv_transpose still have no caller. They
// read the SAME buffer convention cuda_batched_cholesky produces, stated once
// for the whole file at cusolver_factor_to_row_major() below;
// gpu_batched_cholesky_solve (gpu_backend.h) is the composition of the three,
// and cpp_gpu_batched_cholesky_solve() scores that composition against an
// independent solve.
//
// Minimum requirements:
// - CUDA Toolkit 11.0+ (for cusolverDnDpotrfBatched)
// - NVIDIA GPU with compute capability 3.5+
// - Driver version 450.80.02+ (Linux) or 452.39+ (Windows)

#ifndef TULPA_GPU_CUDA_H
#define TULPA_GPU_CUDA_H

#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <mutex>

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
typedef CUresult (*cuDeviceGetAttribute_t)(int*, int, CUdevice);
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
  std::mutex init_mu_;
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
  cuDeviceGetAttribute_t cuDeviceGetAttribute_ = nullptr;
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
    cuDeviceGetAttribute_ = (cuDeviceGetAttribute_t)CUDA_GET_PROC(cuda_lib_, "cuDeviceGetAttribute");
    cuCtxCreate_ = (cuCtxCreate_t)CUDA_GET_PROC(cuda_lib_, "cuCtxCreate_v2");
    cuCtxDestroy_ = (cuCtxDestroy_t)CUDA_GET_PROC(cuda_lib_, "cuCtxDestroy_v2");
    cuCtxSetCurrent_ = (cuCtxSetCurrent_t)CUDA_GET_PROC(cuda_lib_, "cuCtxSetCurrent");
    cuMemAlloc_ = (cuMemAlloc_t)CUDA_GET_PROC(cuda_lib_, "cuMemAlloc_v2");
    cuMemFree_ = (cuMemFree_t)CUDA_GET_PROC(cuda_lib_, "cuMemFree_v2");
    cuMemcpyHtoD_ = (cuMemcpyHtoD_t)CUDA_GET_PROC(cuda_lib_, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_ = (cuMemcpyDtoH_t)CUDA_GET_PROC(cuda_lib_, "cuMemcpyDtoH_v2");
    cuCtxSynchronize_ = (cuCtxSynchronize_t)CUDA_GET_PROC(cuda_lib_, "cuCtxSynchronize");

    if (!cuInit_ || !cuDeviceGetCount_ || !cuMemAlloc_ || !cuMemFree_ ||
        !cuMemcpyHtoD_ || !cuMemcpyDtoH_ || !cuCtxCreate_ || !cuCtxSetCurrent_) {
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

  // Serialized because the outer nested-Laplace grid is an OpenMP parallel for
  // over cells and the block `prep` that reaches this runs OUTSIDE
  // `omp critical(nl_sparse_phi)`. Without the lock two threads both observe
  // `initialized_ == false`, both run `load_libraries()` / `load_functions()`
  // -- which write the library handles and function pointers this object hands
  // out -- and `cuInit_` is called twice. `instance()`'s static is already
  // thread-safe to CONSTRUCT; that says nothing about this (gcol33/tulpa#393).
  //
  // Latent rather than live at the time of writing: every entry that can build
  // an NNGP block passes `n_threads_outer = 1` as a hardcoded literal, and the
  // one entry taking it from R has no `nngp` branch in its block-spec builder.
  // It becomes live the moment either changes, which is exactly the kind of
  // change that would not think to look here.
  bool initialize() {
    std::lock_guard<std::mutex> lock(init_mu_);
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

  // Bind this object's context to the CALLING thread. The driver keeps a
  // per-thread context stack and cuCtxCreate pushes onto the stack of the
  // thread that created it, so a thread that did not run initialize() has no
  // current context: every cuMemAlloc there returns CUDA_ERROR_INVALID_CONTEXT
  // and the entry reads as an unavailable GPU. Every high-level entry binds on
  // the way in, which is what makes them callable from any thread. Cheap and
  // idempotent when the context is already current.
  bool make_current() {
    if (!initialized_ || !context_) return false;
    return cuCtxSetCurrent_(context_) == CUDA_SUCCESS;
  }

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

  // Batched triangular solve: op(A) * X = B, stated in cuBLAS's own terms.
  // cuBLAS reads A COLUMN-major with leading dimension n, and `uplo` names
  // which triangle of that reading holds the factor; the other triangle is not
  // touched. Callers holding a buffer in some other layout convert it or say
  // which triangle the column-major reading of their bytes lands in -- passing
  // the triangle the buffer does NOT fill solves against its diagonal alone and
  // still returns success.
  bool batched_trsm(double** d_A, double** d_B, int n, int nrhs, int batch_size,
                    cublasFillMode_t uplo, cublasOperation_t op) {
    if (!has_cublas() || !cublasDtrsmBatched_) {
      return false;
    }

    const double alpha = 1.0;
    cublasStatus_t status = cublasDtrsmBatched_(
      cublas_handle_,
      CUBLAS_SIDE_LEFT,
      uplo,
      op,
      CUBLAS_DIAG_NON_UNIT,
      n, nrhs,
      &alpha,
      (const double* const*)d_A, n,
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

  // "major.minor", or empty when the driver did not answer. The two constants
  // are CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR in the driver's
  // attribute enumeration, spelled out here for the same reason the CUDA types
  // above are: this file resolves the driver dynamically and does not include
  // cuda.h.
  std::string get_compute_capability(int device_id = 0) {
    static const int kAttrCcMajor = 75;
    static const int kAttrCcMinor = 76;
    if (!initialized_ || !cuDeviceGetAttribute_) return "";
    CUdevice device;
    if (cuDeviceGet_(&device, device_id) != CUDA_SUCCESS) return "";
    int major = 0, minor = 0;
    if (cuDeviceGetAttribute_(&major, kAttrCcMajor, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute_(&minor, kAttrCcMinor, device) != CUDA_SUCCESS) {
      return "";
    }
    return std::to_string(major) + "." + std::to_string(minor);
  }
};

// =============================================================================
// High-level GPU Operations for NNGP
// =============================================================================

// THE BUFFER CONVENTION every entry below passes and expects: a k x k factor
// held as L in the ROW-major lower triangle, with the row-major upper triangle
// zero.
//
// cusolverDnDpotrfBatched is COLUMN-major and is asked for
// CUBLAS_FILL_MODE_LOWER, so it writes L[r][c] (r >= c) at offset c*k + r and
// leaves the opposite triangle holding the input. Every CPU consumer of the
// result (tri_solve_lower<RowMajor> / tri_solve_lower_transpose<RowMajor> in
// linalg_fast.h, and the batch_nngp_scatter probe that scores the batch)
// indexes ROW-major, so without this move it reads the untouched input
// covariances as factor entries -- the diagonal is the only part that agrees,
// which is why such a result stays finite and plausible.
//
// Read back COLUMN-major, as cuBLAS reads it, the same bytes are L' held in the
// UPPER triangle with the strict lower zero. That is what
// cuda_batched_trsv_impl() tells cublasDtrsmBatched, and it is why the two
// triangular solves carry the op OPPOSITE to the one their name suggests.
inline void cusolver_factor_to_row_major(double* m, int k) {
  for (int r = 0; r < k; r++) {
    for (int c = 0; c < r; c++) {
      m[r * k + c] = m[c * k + r];
      m[c * k + r] = 0.0;
    }
  }
}

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
    if (!ctx.initialize() || !ctx.has_cusolver() || !ctx.make_current()) {
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
    // cusolverDnDpotrfBatched reports per-matrix status in `info`: 0 on
    // success, j > 0 when leading minor j is not positive definite. A
    // CUSOLVER_STATUS_SUCCESS return only says the launch worked, so the
    // per-matrix codes have to be read or a non-PD neighbour set comes back as
    // a partially-written factor that looks like a valid one.
    std::vector<int> info(batch_size, 0);
    if (!ctx.copy_to_host(info.data(), d_info,
                          batch_size * sizeof(int))) {
      success = false;
    } else {
      for (int i = 0; i < batch_size; i++) {
        if (info[i] != 0) { success = false; break; }
      }
    }
  }

  if (success) {
    for (int i = 0; i < batch_size; i++) {
      if (!ctx.copy_to_host(matrices[i].data(), d_matrices[i], matrix_bytes)) {
        success = false;
        break;
      }
      // Into the buffer convention stated above, matching the CPU fallback.
      cusolver_factor_to_row_major(matrices[i].data(), k);
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

// Batched triangular solve against a batch of factors in the buffer convention
// cusolver_factor_to_row_major() leaves behind. `op` is stated in cuBLAS's own
// terms, i.e. against the L' that a COLUMN-major read of those bytes gives:
// CUBLAS_OP_T solves L x = b and CUBLAS_OP_N solves L' x = b, which is the
// opposite of what each caller's name says because the buffer is transposed
// relative to how cuBLAS reads it.
//
// b_vectors is overwritten with x; the vectors go to the batched trsm as k x 1
// matrices.
inline bool cuda_batched_trsv_impl(
    const std::vector<std::vector<double>>& L_matrices,  // batch_size x (k*k)
    std::vector<std::vector<double>>& b_vectors,         // batch_size x k (modified in place)
    int k,
    cublasOperation_t op
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
    if (!ctx.initialize() || !ctx.has_cublas() || !ctx.make_current()) {
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

  // The factor fills the ROW-major lower triangle, which is the COLUMN-major
  // UPPER triangle cuBLAS has to be pointed at. Naming the lower one here would
  // hand it a strictly zero off-diagonal and solve against diag(L) alone.
  bool success = ctx.batched_trsm(
    (double**)d_L_ptr_array, (double**)d_b_ptr_array,
    k, 1, batch_size, CUBLAS_FILL_MODE_UPPER, op
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

// Batched triangular solve for NNGP: L * x = b
// L is a k x k factor in the buffer convention above, b/x are vectors of
// length k, solved in place.
inline bool cuda_batched_trsv(
    const std::vector<std::vector<double>>& L_matrices,
    std::vector<std::vector<double>>& b_vectors,
    int k
) {
  return cuda_batched_trsv_impl(L_matrices, b_vectors, k, CUBLAS_OP_T);
}

// Batched transposed triangular solve for NNGP: L^T * x = b
// The back-substitution after a forward solve with the same factor, reading the
// same buffer.
inline bool cuda_batched_trsv_transpose(
    const std::vector<std::vector<double>>& L_matrices,
    std::vector<std::vector<double>>& b_vectors,
    int k
) {
  return cuda_batched_trsv_impl(L_matrices, b_vectors, k, CUBLAS_OP_N);
}

}  // namespace tulpa_gpu

#endif  // TULPA_GPU_CUDA_H
